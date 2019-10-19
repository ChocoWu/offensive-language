#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import LSTM
from .char_encoder import CharEncoderCNN, CharEncoderLSTM
import argparse
from allennlp.modules.elmo import Elmo, batch_to_ids
from utils import *


class Hi_Attention(nn.Module):
    def __init__(self, config, w_embedding=None, c_embedding=None, l_embedding=None):
        super(Hi_Attention, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.dropout_prob = config.dropout_prob
        self.num_class = config.num_class
        self.max_sent = config.max_sent
        self.max_word = config.max_word

        self.triplet = config.triplet  # 是否使用triplet loss

        self.w_num_layer = config.w_num_layer
        self.w_hidden_size = config.w_hidden_size
        self.w_atten_size = config.w_atten_size
        self.w_is_bidirectional = config.w_is_bidirectional
        self.w_dropout_prob = config.w_dropout_prob

        self.s_num_layer = config.s_num_layer
        self.s_hidden_size = config.s_hidden_size
        self.s_atten_size = config.s_atten_size
        self.s_is_bidirectional = config.s_is_bidirectional
        self.s_dropout_prob = config.s_dropout_prob

        self.use_type = config.use_type
        self.char_encode_type = config.char_encode_type
        self.alphabet_size = config.alphabet_size
        self.c_embedding_size = config.c_embedding_size
        self.c_hidden_size = config.c_hidden_size
        self.c_num_layer = config.c_num_layer
        self.c_is_bidirectional = config.c_is_bidirectional
        self.c_dropout_prob = config.c_dropout_prob
        self.c_kernel_size = config.c_kernel_size
        self.c_num_filter = config.num_filter

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # self.label_embedding = nn.Embedding(self.config.n_tags, self.embedding_size)
        if w_embedding is not None:
            self.embedding.weight = nn.Parameter(w_embedding)

        # if l_embedding is not None:
        #     self.label_embedding.weight = nn.Parameter(l_embedding)

        if self.w_is_bidirectional:
            self.w_num_directions = 2
        else:
            self.w_num_directions = 1

        if self.s_is_bidirectional:
            self.s_num_directions = 2
        else:
            self.s_num_directions = 1

        if self.char_encode_type == 'cnn':
            self.char_encode = CharEncoderCNN(self.alphabet_size, self.c_embedding_size, self.c_kernel_size,
                                              self.c_num_filter, c_embedding)
        elif self.char_encode_type == 'lstm':
            self.char_encode = CharEncoderLSTM(self.alphabet_size, self.c_embedding_size, self.c_hidden_size,
                                               self.c_num_layer, self.c_is_bidirectional, self.c_dropout_prob,
                                               c_embedding)

        if self.use_type == 'char':
            if self.c_is_bidirectional:
                self.w_input_size = self.c_hidden_size * 2 + self.embedding_size
            else:
                self.w_input_size = self.c_hidden_size + self.embedding_size
        elif self.use_type == 'elmo':
            self.w_input_size = self.embedding_size + 1024
        else:
            self.w_input_size = self.embedding_size

        self.word_atten = LSTM(self.w_input_size, self.w_hidden_size, self.w_num_layer,
                               self.w_dropout_prob, self.w_is_bidirectional, self.w_atten_size)
        self.s_input_size = self.w_num_directions * self.w_hidden_size
        self.sent_atten = LSTM(self.s_input_size, self.s_hidden_size, self.s_num_layer,
                               self.s_dropout_prob, self.s_is_bidirectional, self.s_atten_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.s_hidden_size * self.s_num_directions, 2)
        self.elmo = Elmo(config.options_file, config.weights_file, 1)

    def forward(self, x_word, word_mask, sent_mask, x_char=None, word=None, label=None):
        """

        :param x_word: [batch_size, max_sent, max_word]
        :param word: [batch_size, max_sent, max_word, max_char]
        :param word_mask: [batch_size * max_sent, max_word]
        :param sent_mask: [batch_sie, max_sent]
        :param x_char: [batch_size, max_sent, max_word, max_char]
        :return:
            [batch_size, num_class]
        """
        if self.use_type == 'char' and x_char is not None:
            x_char = x_char.view(-1, x_char.size(3))
            x_char = self.char_encode(x_char)
            x_char = x_char.view(-1, self.max_word, x_char.size(1))
            x_word = self.embedding(x_word)
            x_word = x_word.view(-1, self.max_word, self.embedding_size)
            x = torch.cat([x_word, x_char], dim = 2)
        elif self.use_type == 'elmo' and word is not None:
            # print(x.size())  # 4, 6, 35
            x = self.embedding(x_word)
            x = x.view(-1, self.max_word, self.embedding_size)

            # 使用elmo, 其参数(batch_size, timesteps, 50)
            # characters = batch_to_ids(word)
            word = word.view(-1, self.max_word, 50)
            embeddings = self.elmo(word)
            elmo_embeddings = embeddings['elmo_representations'][0]
            # elmo_embeddings = elmo_embeddings[:, :self.max_word, :]
            # elmo_mask = embeddings['mask']
            x = torch.cat([x, elmo_embeddings], dim = 2)
        else:
            x = self.embedding(x_word)
            x = x.view(-1, self.max_word, self.embedding_size)
        if label is not None:
            x, word_weights = self.word_atten(x, word_mask, label = label)
            x = self.dropout(x)
            dim = x.size(1)
            x = x.view(-1, self.max_sent, dim)
            x, sent_weights = self.sent_atten(x, sent_mask, label = label)
            x = self.dropout(x)
            x = self.linear(x)  # 150 * 2
        else:
            x, word_weights = self.word_atten(x, word_mask)
            x = self.dropout(x)
            # x = F.layer_norm(x, x.size()[1:])
            dim = x.size(1)
            x = x.view(-1, self.max_sent, dim)
            x, sent_weights = self.sent_atten(x, sent_mask)
            x = self.dropout(x)
            # x = F.layer_norm(x, x.size()[1:])
            x = self.linear(x)  # 150 * 2
            # if self.triplet:
            #     return x  # 直接返回embedding的结果
            # else:
            #     x = self.linear(x)

        return x, word_weights, sent_weights


class ClassificationNet(nn.Module):
    def __init__(self, config):
        super(ClassificationNet, self).__init__()
        if config.s_is_bidirectional:
            self.s_num_directions = 2
        else:
            self.s_num_directions = 1
        # self.in_size = config.s_hidden_size * self.s_num_directions
        self.in_size = 2
        self.num_class = config.num_class
        self.linear = nn.Linear(self.in_size, self.num_class)

    def forward(self, x):
        output = self.linear(x)
        return output


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_file", type = str, default = "./data/english/agr_en_train.csv")
#     parser.add_argument("--valid_file", type = str, default = "./data/english/agr_en_dev.csv")
#     parser.add_argument("--test_file", type = str, default = "./data/english/agr_en_fb_test.csv")
#     # parser.add_argument("--tag_format", type = str, choices = ["bio", "bmes"], default = "bio")
#
#     parser.add_argument("--save_dir", type = str, default = "./checkpoint/")
#     parser.add_argument("--log_dir", type = str, default = "./log/")
#     parser.add_argument("--config_path", type = str, default = "./checkpoint/config.pt")
#     parser.add_argument("--continue_train", type = bool, default = False, help = "continue to train model")
#     parser.add_argument("--pretrain_embedding", type = bool, default = True)
#     parser.add_argument("--embedding_file", type = str, default = "./data/glove.840B.300d.txt")
#
#     parser.add_argument("--seed", type = int, default = 123, help = "seed for random")
#     parser.add_argument("--batch_size", type = int, default = 64, help = "number of batch size")
#     parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
#     parser.add_argument("--embedding_size", type = int, default = 300)
#     parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of adam")
#     parser.add_argument("--weight_decay", type = float, default = 1e-5, help = "weight decay of adam")
#     parser.add_argument("--patience", type = int, default = 8)
#     parser.add_argument("--freeze", type = int, default = 10)
#     parser.add_argument("--num_class", type = int, default = 3)
#     parser.add_argument("--dropout_prob", type = float, default = 0.5)
#     parser.add_argument("--max_sent", type = int, default = 6)
#     parser.add_argument("--max_word", type = int, default = 35)
#
#     parser.add_argument("--w_hidden_size", type = int, default = 150)
#     parser.add_argument("--w_num_layer", type = int, default = 1, help = "number of layers")
#     parser.add_argument("--w_atten_size", type = int, default = 300)
#     parser.add_argument("--w_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--w_dropout_prob", type = float, default = 0.5)
#
#     parser.add_argument("--s_hidden_size", type = int, default = 150)
#     parser.add_argument("--s_num_layer", type = int, default = 1, help = "number of layers")
#     parser.add_argument("--s_atten_size", type = int, default = 300)
#     parser.add_argument("--s_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--s_dropout_prob", type = float, default = 0.5)
#
#     parser.add_argument("--is_use_char", type = bool, default = False)
#     parser.add_argument("--char_encode_type", type = str, default = 'lstm')
#     parser.add_argument("--c_embedding_size", type = int, default = 50)
#     parser.add_argument("--c_hidden_size", type = int, default = 20)
#     parser.add_argument("--c_num_layer", type = int, default = 1)
#     parser.add_argument("--c_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--c_dropout_prob", type = float, default = 0.5)
#     parser.add_argument("--c_kernel_size", type = list, default = [2])
#     parser.add_argument("--num_filter", type = int, default = 2)
#
#     parser.add_argument("--use_gpu", type = bool, default = True)
#     args = parser.parse_args()
#     args.vocab = None
#     # train_dataset, valid_dataset, test_dataset, vocab = data_utils.load_data(args, args.vocab)
#     # args.vocab = vocab
#     args.vocab_size = 100
#     args.n_tags = 3
#     args.alphabet_size = 36
#
#     model = Hi_Attention(args)
#     print(model)
#     for name, param in model.named_parameters():
#         print(name, param.size())
