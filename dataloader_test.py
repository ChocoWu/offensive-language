#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from data_utils import Vocabulary, build_data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np
from model.model import Hi_Attention
from utils import *
import argparse
import time
import torch.optim as optim

def pad_sequence(sequences, batch_first, max_sent=0, max_word=0, padding_value=0):
    """
    重写了rnn中的pad_sequence
    :param sequence: [batch_size, max_sent_num, max_word_num]
    :param batch_first:
    :param pad_value: 默认为0
    :return:
    """
    # max_size = sequences[0][0].size()
    # trailing_dims = max_size[1:]
    # trailing_dims = []
    # max_sent = max([len(doc) for doc in sequences])
    # max_word = max([max([len(sent) for sent in doc]) for doc in sequences])
    #
    # if batch_first:
    #     out_dims = (len(sequences), max_sent, max_word)
    # else:
    #     out_dims = (max_sent, max_word, len(sequences))
    #
    # # out_tensor = sequences[0][0].data.new(*out_dims).fill_(padding_value)
    # out_tensor = np.full(out_dims, padding_value)
    # # print(out_tensor)
    # for i, tensors in enumerate(sequences):
    #     for j, tensor in enumerate(tensors):
    #         length = len(tensor)
    #         if batch_first:
    #             out_tensor[i, j, :length, ...] = tensor
    #         else:
    #             out_tensor[j, :length, i, ...] = tensor
    #
    # return out_tensor
    if max_sent == 0 and max_word == 0:
        max_sent = max([len(doc) for doc in sequences])
        max_word = max([max([len(sent) for sent in doc]) for doc in sequences])

    if batch_first:
        out_dims = (len(sequences), max_sent, max_word)
    else:
        out_dims = (max_sent, max_word, len(sequences))

    # out_tensor = sequences[0][0].data.new(*out_dims).fill_(padding_value)
    out_tensor = np.full(out_dims, padding_value)
    # print(out_tensor)
    for i, tensors in enumerate(sequences):
        tensors = tensors[:max_sent]
        for j, tensor in enumerate(tensors):
            length = len(tensor)
            if length >= max_word:
                if batch_first:
                    out_tensor[i, j, :max_word, ...] = tensor[:max_word]
                else:
                    out_tensor[j, :max_word, i, ...] = tensor[:max_word]
            else:
                if batch_first:
                    out_tensor[i, j, :length, ...] = tensor
                else:
                    out_tensor[j, :length, i, ...] = tensor

    return out_tensor


class myDataset(Dataset):
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels

        self.collate_fn = lambda x: [pad_sequence(d, True) for _, d in pd.DataFrame(x).iteritems()]

    def __getitem__(self, item):
        # print("self.labels[item]", self.labels[item])
        return torch.tensor(self.inputs[item], dtype = torch.long), torch.tensor(self.labels[item], dtype = torch.long)

    def __len__(self):
        return len(self.inputs)


def load_data(config, vocab=None):
    test_df = pd.read_csv(config.test_file, header = 0, names = ['face_id', 'content', 'label'])

    test_data, test_label, test_num_sent, test_num_word = build_data(test_df['content'], test_df['label'])

    if vocab is None:
        vocab = Vocabulary()
        [[vocab.add_sentence(x, y) for (x, y) in zip(data, test_label)] for data in test_data]

    test_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in test_data]
    test_label = [vocab.tag_to_id(label) for label in test_label]
    test_input = pad_sequence(test_input, True, config.max_sent, config.max_word)
    # t = torch.tensor(test_input)
    # print(t.size())
    # print(test_label)
    test_dataset = myDataset(test_input, test_label)

    return test_dataset, vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type = str, default = "./data/english/agr_en_train.csv")
    parser.add_argument("--valid_file", type = str, default = "./data/english/agr_en_dev.csv")
    parser.add_argument("--test_file", type = str, default = "./data/english/test_file.csv")
    # parser.add_argument("--tag_format", type = str, choices = ["bio", "bmes"], default = "bio")

    parser.add_argument("--save_dir", type = str, default = "./checkpoint/")
    parser.add_argument("--log_dir", type = str, default = "./log/")
    parser.add_argument("--config_path", type = str, default = "./checkpoint/config.pt")
    parser.add_argument("--continue_train", type = bool, default = False, help = "continue to train model")
    parser.add_argument("--pretrain_embedding", type = bool, default = True)
    parser.add_argument("--embedding_file", type = str, default = "./data/glove.840B.300d.txt")

    parser.add_argument("--seed", type = int, default = 123, help = "seed for random")
    parser.add_argument("--batch_size", type = int, default = 64, help = "number of batch size")
    parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
    parser.add_argument("--embedding_size", type = int, default = 300)
    parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of adam")
    parser.add_argument("--weight_decay", type = float, default = 1e-5, help = "weight decay of adam")
    parser.add_argument("--patience", type = int, default = 8)
    parser.add_argument("--freeze", type = int, default = 10)
    parser.add_argument("--num_class", type = int, default = 3)
    parser.add_argument("--dropout_prob", type = float, default = 0.5)
    parser.add_argument("--max_sent", type = int, default = 6)
    parser.add_argument("--max_word", type = int, default = 35)

    parser.add_argument("--w_hidden_size", type = int, default = 150)
    parser.add_argument("--w_num_layer", type = int, default = 1, help = "number of layers")
    parser.add_argument("--w_atten_size", type = int, default = 300)
    parser.add_argument("--w_is_bidirectional", type = bool, default = True)
    parser.add_argument("--w_dropout_prob", type = float, default = 0.5)

    parser.add_argument("--s_hidden_size", type = int, default = 150)
    parser.add_argument("--s_num_layer", type = int, default = 1, help = "number of layers")
    parser.add_argument("--s_atten_size", type = int, default = 300)
    parser.add_argument("--s_is_bidirectional", type = bool, default = True)
    parser.add_argument("--s_dropout_prob", type = float, default = 0.5)

    parser.add_argument("--use_gpu", type = bool, default = False)
    args = parser.parse_args()

    logger = get_logger(args.log_dir + "NER_Train_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))

    logger.info(args)

    seed = args.seed
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    USE_GPU = args.use_gpu

    logger.info("Loading data...")
    if args.continue_train:
        args = load_from_pickle(args.config_path)
    else:
        args.vocab = None
    dataset, vocab = load_data(args, args.vocab)
    args.vocab = vocab
    embedding = None
    args.vocab_size = vocab.n_words
    args.n_tags = vocab.n_tags

    model = Hi_Attention(args, embedding)
    criterion = nn.CrossEntropyLoss()
    optim = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    dataloader = DataLoader(dataset = dataset,
                            batch_size = 4,
                            shuffle = False)
    model.train()
    loss_array = []
    for inputs, labels in dataloader:
        if USE_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()
        mask = inputs.ne(0).byte()
        word_mask = mask.reshape(-1, mask.size(2))
        sent_mask = mask.sum(2).ne(0).byte()

        output = model(inputs, word_mask, sent_mask)
        # result = torch.max(output, 1)[1]
        loss = criterion(output, labels)
        loss = torch.mean(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_array.append(loss.cpu().item())

