#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn
import argparse
import time
import c_data_utils
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import Hi_Attention
from tqdm import tqdm
from utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval_metric(pred_label, gold_label):
    weighted_f1 = metrics.f1_score(gold_label, pred_label, average = 'weighted')
    macro_f1 = metrics.f1_score(gold_label, pred_label, average = 'macro')
    p = metrics.precision_score(gold_label, pred_label, average = 'weighted')
    r = metrics.recall_score(gold_label, pred_label, average = 'weighted')
    acc = metrics.accuracy_score(gold_label, pred_label)

    # report = classification_report(gold_label, pred_label)
    # print(report)
    #
    # confusion_matrix = confusion_matrix(gold_label, pred_label)
    # print(confusion_matrix)
    #
    # matrix_proportions = np.zeros((3, 3))
    # for i in range(0, 3):
    #     matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
    # names = ['NAG', 'CAG', 'OAG']
    # confusion_df = pd.DataFrame(matrix_proportions, index = names, columns = names)
    # plt.figure(figsize = (5, 5))
    # seaborn.heatmap(confusion_df, annot = True, annot_kws = {'size': 12}, cmap = 'gist_gray_r', cbar = False,
    #                 square = True, fmt = '.2f')
    # plt.ylabel(r'True categories', fontsize = 14)
    # plt.xlabel(r'Predicted categories', fontsize = 14)
    # plt.tick_params(labelsize = 12)
    # plt.show()

    return weighted_f1, macro_f1, p, r, acc


def calculate_weights(inputs, classes):
    """
    calculate the weights of inputs
    根据每个类别数的倒数计算其权重
    :param inputs:
    :param classes: 类别数
    :return:
    """
    weights = []
    for i in range(classes):
        weights.append(inputs.size(0)/(inputs.size(0)-inputs.ne(i).sum().item()))
    return


class Trainer(object):
    def __init__(self, config, w_embedding, c_embedding=None):
        self.config = config
        self.model = Hi_Attention(self.config, w_embedding, c_embedding)

        self.optim = optim.Adam(self.model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        if config.use_gpu:
            self.model = self.model.cuda()

    def __call__(self, train_dataset, valid_dataset, fb_test_dataset, tw_test_dataset):
        best_f1 = 0.
        best_fb_test_f1 = 0.
        best_tw_test_f1 = 0.
        patience = 0
        for epoch in range(self.config.epochs):
            if self.config.pretrain_embedding:
                if epoch < self.config.freeze:
                    self.model.embedding.weight.requires_grad = False
                else:
                    self.model.embedding.weight.requires_grad = True
            loss, during_time = self.train(train_dataset)
            logger.info("Epoch: {} Loss: {:.4f} Time: {}".format(epoch, loss, int(during_time)))

            weighted_f1, macro_f1, p, r, acc, during_time, _, _ = self.eval(valid_dataset)
            logger.info("Epoch: {} Valid Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, acc, p, r, weighted_f1, int(during_time)))

            fb_test_f1, _, fb_test_p, fb_test_r, fb_test_acc, t_during_time, _, _ = self.eval(fb_test_dataset)
            logger.info("Epoch: {} Facebook Test Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, fb_test_acc, fb_test_p, fb_test_r, fb_test_f1, int(t_during_time)))

            tw_test_f1, _, tw_test_p, tw_test_r, tw_test_acc, t_during_time, _, _ = self.eval(tw_test_dataset)
            logger.info("Epoch: {} Twitter Test Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, tw_test_acc, tw_test_p, tw_test_r, tw_test_f1, int(t_during_time)))
            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                best_fb_test_f1 = fb_test_f1
                best_tw_test_f1 = tw_test_f1
                patience = 0
                self.save()
                logger.info("F1: {:.4f} Model is saved!".format(best_f1))

            else:
                patience += 1
            if patience >= self.config.patience:
                break
        logger.info("Best Valid_F1: {:.4f}, Best Facebook Test_F1: {:.4f}, Best Twitter Test_F1: {:.4f}".format(best_f1,
                                                                                                                best_fb_test_f1,
                                                                                                                best_tw_test_f1))

    def train(self, dataset):
        start_time = time.time()
        self.model.train()

        loss_array = []

        if self.config.is_use_char:
            for w_inputs, c_inputs, labels in tqdm(dataset):
                if self.config.use_gpu:
                    w_inputs = w_inputs.cuda()
                    c_inputs = c_inputs.cuda()
                    labels = labels.cuda()
                    mask = w_inputs.ne(0).byte()
                    word_mask = mask.reshape(-1, mask.size(2))
                    sent_mask = mask.sum(2).ne(0).byte()
                    output = self.model(w_inputs, word_mask, sent_mask, c_inputs)
                    loss = self.criterion(output, labels)
                    loss = torch.mean(loss)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    loss_array.append(loss.cpu().item())
        else:
            for inputs, labels in tqdm(dataset):

                if self.config.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                mask = inputs.ne(0).byte()
                word_mask = mask.reshape(-1, mask.size(2))
                sent_mask = mask.sum(2).ne(0).byte()

                output = self.model(inputs, word_mask, sent_mask)
                # result = torch.max(output, 1)[1]
                loss = self.criterion(output, labels)
                loss = torch.mean(loss)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_array.append(loss.cpu().item())

        during_time = time.time() - start_time
        return np.mean(loss_array), during_time

    def eval(self, dataset):
        start_time = time.time()
        self.model.eval()

        pred_labels = []
        gold_labels = []

        with torch.no_grad():
            if self.config.is_use_char:
                for inputs, c_inputs, labels in tqdm(dataset):
                    if self.config.use_gpu:
                        inputs = inputs.cuda()
                        c_inputs = c_inputs.cuda()
                        labels = labels.cuda()
                    mask = inputs.ne(0).byte()
                    word_mask = mask.view(-1, mask.size(2))
                    sent_mask = mask.sum(2).ne(0).byte()
                    output = self.model(inputs, word_mask, sent_mask, c_inputs)
                    result = torch.max(output, 1)[1]
                    pred_labels.extend(result.cpu().numpy().tolist())
                    gold_labels.extend(labels.cpu().numpy().tolist())
            else:
                for inputs, labels in tqdm(dataset):
                    if self.config.use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    mask = inputs.ne(0).byte()
                    word_mask = mask.view(-1, mask.size(2))
                    sent_mask = mask.sum(2).ne(0).byte()
                    output = self.model(inputs, word_mask, sent_mask)
                    result = torch.max(output, 1)[1]
                    pred_labels.extend(result.cpu().numpy().tolist())
                    gold_labels.extend(labels.cpu().numpy().tolist())

            weighted_f1, macro_f1, p, r, acc = eval_metric(pred_labels, gold_labels)
            during_time = time.time() - start_time
            return weighted_f1, macro_f1, p, r, acc, during_time, pred_labels, gold_labels

    def save(self):
        torch.save(self.model.state_dict(), self.config.save_dir + 'model.pt')
        torch.save(self.optim.state_dict(), self.config.save_dir + 'optim.pt')

    def load(self):
        self.model.load_state_dict(torch.load(self.config.save_dir + 'model.pt'))
        self.optim.load_state_dict(torch.load(self.config.save_dir + 'optim.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type = str, default = "./data/english/agr_en_train.csv")
    parser.add_argument("--valid_file", type = str, default = "./data/english/agr_en_dev.csv")
    parser.add_argument("--fb_test_file", type = str, default = "./data/english/agr_en_fb_test.csv")
    parser.add_argument("--tw_test_file", type = str, default = "./data/english/agr_en_tw_test.csv")
    # parser.add_argument("--tag_format", type = str, choices = ["bio", "bmes"], default = "bio")

    parser.add_argument("--save_dir", type = str, default = "./checkpoint/")
    parser.add_argument("--log_dir", type = str, default = "./log/")
    parser.add_argument("--config_path", type = str, default = "./checkpoint/config.pt")
    parser.add_argument("--vocab_path", type = str, default = "./checkpoint/vocab.pt")
    parser.add_argument("--embedding_path", type = str, default = "./checkpoint/embedding.pt")
    parser.add_argument("--continue_train", type = bool, default = False, help = "continue to train model")
    parser.add_argument("--pretrain_embedding", type = bool, default = True)
    parser.add_argument("--embedding_file", type = str, default = "./data/glove.840B.300d.txt")

    parser.add_argument("--seed", type = int, default = 123, help = "seed for random")
    parser.add_argument("--batch_size", type = int, default = 150, help = "number of batch size")
    parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
    parser.add_argument("--embedding_size", type = int, default = 300)
    parser.add_argument("--lr", type = float, default = 0.004, help = "learning rate of adam")
    parser.add_argument("--weight_decay", type = float, default = 1e-5, help = "weight decay of adam")
    parser.add_argument("--patience", type = int, default = 10)
    parser.add_argument("--freeze", type = int, default = 5)
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

    parser.add_argument("--is_use_char", type = bool, default = True)
    parser.add_argument("--char_encode_type", type = str, default = 'lstm')
    parser.add_argument("--c_embedding_size", type = int, default = 50)
    parser.add_argument("--c_hidden_size", type = int, default = 20)
    parser.add_argument("--c_num_layer", type = int, default = 1)
    parser.add_argument("--c_is_bidirectional", type = bool, default = True)
    parser.add_argument("--c_dropout_prob", type = float, default = 0.5)
    parser.add_argument("--c_kernel_size", type = list, default = [2])
    parser.add_argument("--num_filter", type = int, default = 2)

    parser.add_argument("--use_gpu", type = bool, default = True)
    args = parser.parse_args()

    logger = get_logger(args.log_dir + "NER_Train_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))

    logger.info(args)

    seed = args.seed
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    logger.info("Loading data...")
    if os.path.exists(args.vocab_path):
        logger.info("vocabulary exists")
        vocab = load_from_pickle(args.vocab_path)
    else:
        logger.info("vocabulary does not exit")
        vocab = c_data_utils.build_vocab(args)
        save_to_pickle(args.vocab_path, vocab)
        logger.info("vocabulary loading is done")

    logger.info("Loading data ......")
    train_dataset = c_data_utils.load_data(args.train_file, args.max_sent, args.max_word, vocab, args.is_use_char)
    valid_dataset = c_data_utils.load_data(args.valid_file, args.max_sent, args.max_word, vocab, args.is_use_char)
    fb_test_dataset = c_data_utils.load_data(args.fb_test_file, args.max_sent, args.max_word, vocab,
                                             args.is_use_char)
    tw_test_dataset = c_data_utils.load_data(args.tw_test_file, args.max_sent, args.max_word, vocab,
                                             args.is_use_char)

    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset,
                              batch_size = args.batch_size,
                              shuffle = False)
    fb_test_loader = DataLoader(dataset = fb_test_dataset,
                                batch_size = args.batch_size,
                                shuffle = False)
    tw_test_loader = DataLoader(dataset = tw_test_dataset,
                                batch_size = args.batch_size,
                                shuffle = False)
    logger.info("Data loading is done!")

    if args.pretrain_embedding:
        logger.info("Loading embedding file...")
        if os.path.exists(args.embedding_path):
            w_embedding = load_from_pickle(args.embedding_path)
        else:
            w_embedding = c_data_utils.load_embedding(args, vocab)
            w_embedding = torch.tensor(w_embedding, dtype = torch.float)
            save_to_pickle(args.embedding_path, w_embedding)
        logger.info("Embedding loading is done!")
    else:
        w_embedding = None

    args.vocab_size = vocab.n_words
    args.n_tags = vocab.n_tags
    args.alphabet_size = vocab.n_chars

    trainer = Trainer(args, w_embedding)

    if args.continue_train:
        logger.info("Loading model...")
        trainer.load()

    save_to_pickle(args.config_path, args)
    logger.info("Configuration data is saved in {}".format(args.config_path))

    logger.info("Start training...")
    trainer(train_loader, valid_loader, fb_test_loader, tw_test_loader)
