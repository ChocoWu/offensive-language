#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

# !/user/bin/env python3
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
# from torch.nn.utils.rnn import pad_sequence
import string
import data_utils


class Vocabulary(object):
    T_PAD = '<PAD>'
    W_PAD, W_UNK = '<pad>', '<unk>'
    C_PAD, C_UNK = '<pad>', '<unk>'
    # alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    alphabet = string.printable

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.char2id = {}
        self.id2char = {}
        self.char2count = {}

        self.tag2id = {}
        self.id2tag = {}
        self.tag2count = {}

        self._init_vocab()

    def _init_vocab(self):
        for word in [self.W_PAD, self.W_UNK]:
            self.word2id[word] = len(self.word2id)
            self.id2word[self.word2id[word]] = word
            self.word2count[word] = 0

        for c in [self.C_PAD, self.C_UNK]:
            self.char2id[c] = len(self.char2id)
            self.id2char[self.char2id[c]] = c
        for c in self.alphabet:
            self.char2id[c] = len(self.char2id)
            self.id2char[self.char2id[c]] = c
            self.char2count[c] = 0

    def add_sentence(self, sentence, tags):
        for word in sentence:
            if word in self.word2id:
                self.word2count[word] += 1
            else:
                self.word2id[word] = len(self.word2id)
                self.id2word[self.word2id[word]] = word
                self.word2count[word] = 1

        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)
                self.id2tag[self.tag2id[tag]] = tag
                self.tag2count = 1
            else:
                self.tag2count += 1

    @property
    def n_words(self):
        return len(self.word2id)

    @property
    def n_tags(self):
        return len(self.tag2id)

    @property
    def n_chars(self):
        return len(self.char2id)

    def word_to_id(self, word):
        return self.word2id.get(word, self.word2id[self.W_UNK])

    def tag_to_id(self, tag):
        return self.tag2id[tag]

    def id_to_word(self, id):
        return self.id2word[id]

    def id_to_tag(self, id):
        return self.id2tag[id]

    def char_to_id(self, c):
        return self.char2id.get(c, self.char2id[self.C_UNK])

    def id_to_char(self, id):
        return self.id2char[id]


def pad_sequence(sequences, batch_first, max_sent = 0, max_word = 0, padding_value = 0):
    """
    针对的是word_sentence_level的pad
    :param sequence: [batch_size, max_sent_num, max_word_num]
    :param batch_first:
    :param pad_value: 默认为0
    :return:
    """
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


def pad_sequence_c(sequences, max_sent=None, max_word=None, max_char=None, padding_value=0):
    """
    针对char_word_sentence_level的pad
    :param sequences: 边长的数据
    :param max_sent:
    :param max_word:
    :param max_char:
    :param padding_value:
    :return:
        batch_size, max_sent, max_word, max_char
    """
    if max_sent is None and max_word is None and max_char is None:
        max_sent = max([len(doc) for doc in sequences])
        max_word = max([max([len(sent) for sent in doc]) for doc in sequences])
        max_char = max([max([max(len(word) for word in sent) for sent in doc]) for doc in sequences])

    out_dims = (len(sequences), max_sent, max_word, max_char)
    out_tensor = np.full(out_dims, padding_value)
    for i, doc in enumerate(sequences):
        doc = doc[:max_sent]
        for j, sent in enumerate(doc):
            sent = sent[:max_word]
            for k, word in enumerate(sent):
                length = len(word)
                if length > max_char:
                    out_tensor[i, j, k, :max_char, ...] = word[:max_char]
                else:
                    out_tensor[i, j, k, :length, ...] = word[:length]

    return out_tensor


class MyDataset(Dataset):
    def __init__(self, inputs, labels, c_inputs=None, is_use_char=True):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.c_inputs = c_inputs
        self.labels = labels
        self.is_use_char = is_use_char

    def __getitem__(self, item):
        if self.is_use_char:
            return torch.tensor(self.inputs[item], dtype = torch.long), \
                   torch.tensor(self.c_inputs[item], dtype = torch.long), \
                   torch.tensor(self.labels[item], dtype = torch.long)
        else:
            return torch.tensor(self.inputs[item], dtype = torch.long),\
                   torch.tensor(self.labels[item], dtype = torch.long)

    def __len__(self):
        return len(self.inputs)


def build_data(doc, labels):
    assert len(doc) == len(labels)
    tokens, labs = [], []
    for sents, label in zip(doc, labels):
        labs.append(str(label))
        temp = sents.split('|')
        res = []
        for sent in temp:
            words = sent.strip().split()
            res.append(words)
        tokens.append(res)

    return tokens, labs


def build_data_c(doc, labels):
    assert len(doc) == len(labels)
    w_tokens, c_tokens, labs = [], [], []
    for sents, label in zip(doc, labels):
        labs.append(str(label))
        s_temp = sents.split("|")
        s_res = []
        w_res = []
        for sent in s_temp:
            words = sent.strip().split()
            s_res.append(words)
            temp = []
            for word in words:
                temp.append([c for c in word])
            w_res.append(temp)
        w_tokens.append(s_res)
        c_tokens.append(w_res)
    return w_tokens, c_tokens, labs

#
# def load_data(config, vocab = None):
#     train_df = pd.read_csv(config.train_file, header = 0, names = ['face_id', 'content', 'label'])
#     valid_df = pd.read_csv(config.valid_file, header = 0, names = ['face_id', 'content', 'label'])
#     test_df = pd.read_csv(config.test_file, header = 0, names = ['face_id', 'content', 'label'])
#
#     w_train_data, c_train_data, train_label = build_data_c(train_df['content'], train_df['label'])
#     w_valid_data, c_valid_data, valid_label = build_data_c(valid_df['content'], valid_df['label'])
#     w_test_data, c_test_data, test_label = build_data_c(test_df['content'], test_df['label'])
#
#     if vocab is None:
#         vocab = Vocabulary()
#         [[vocab.add_sentence(x, y) for (x, y) in zip(data, train_label)] for data in w_train_data]
#         [[vocab.add_sentence(x, y) for (x, y) in zip(data, valid_label)] for data in w_valid_data]
#         [[vocab.add_sentence(x, y) for (x, y) in zip(data, test_label)] for data in w_test_data]
#
#     w_train_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_train_data]
#     c_train_input = [[[[vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_train_data]
#     w_train_input = pad_sequence(w_train_input, True, config.max_sent, config.max_word)
#     c_train_input = pad_sequence_c(c_train_input, config.max_sent, config.max_word, 25)
#     train_label = [vocab.tag_to_id(label) for label in train_label]
#
#     w_valid_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_valid_data]
#     c_valid_input = [[[[vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_valid_data]
#     w_valid_input = pad_sequence(w_valid_input, True, config.max_sent, config.max_word)
#     c_valid_input = pad_sequence_c(c_valid_input, config.max_sent, config.max_word, 25)
#     valid_label = [vocab.tag_to_id(label) for label in valid_label]
#
#     w_test_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_test_data]
#     c_test_input = [[[[vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_test_data]
#     w_test_input = pad_sequence(w_test_input, True, config.max_sent, config.max_word)
#     c_test_input = pad_sequence_c(c_test_input, config.max_sent, config.max_word, 25)
#     test_label = [vocab.tag_to_id(label) for label in test_label]
#
#     train_dataset = MyDataset(w_train_input, train_label, c_train_input, config.is_use_char)
#     valid_dataset = MyDataset(w_valid_input, valid_label, c_valid_input, config.is_use_char)
#     test_dataset = MyDataset(w_test_input, test_label, c_test_input, config.is_use_char)
#
#     return train_dataset, valid_dataset, test_dataset, vocab


def load_data(filename, max_sent, max_word, vocab=None, is_use_char=True):
    df = pd.read_csv(filename, header = 0, names = ['face_id', 'content', 'label'])

    if is_use_char:
        w_data, c_data, label = build_data_c(df['content'], df['label'])
        w_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_data]
        c_input = [[[[vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_data]
        w_input = pad_sequence(w_input, True, max_sent, max_word)
        c_input = pad_sequence_c(c_input, max_sent, max_word, 25)
        label = [vocab.tag_to_id(label) for label in label]
        dataset = MyDataset(w_input, label, c_input, is_use_char)
        return dataset
    else:
        w_data, label = build_data(df['content'], df['label'])
        w_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_data]
        w_input = pad_sequence(w_input, True, max_sent, max_word)
        label = [vocab.tag_to_id(label) for label in label]
        dataset = MyDataset(w_input, label, is_use_char = is_use_char)
        return dataset


def build_vocab(config):
    train_df = pd.read_csv(config.train_file, header = 0, names = ['face_id', 'content', 'label'])
    # valid_df = pd.read_csv(config.valid_file, header = 0, names = ['face_id', 'content', 'label'])
    # fb_test_df = pd.read_csv(config.fb_test_file, header = 0, names = ['face_id', 'content', 'label'])
    # tw_test_df = pd.read_csv(config.tw_test_file, header = 0, names = ['face_id', 'content', 'label'])

    train_data, train_label = build_data(train_df['content'], train_df['label'])
    # valid_data, valid_label = build_data(valid_df['content'], valid_df['label'])
    # fb_test_data, fb_test_label = build_data(fb_test_df['content'], fb_test_df['label'])
    # tw_test_data, tw_test_label = build_data(tw_test_df['content'], tw_test_df['label'])

    vocab = Vocabulary()

    [[vocab.add_sentence(x, y) for (x, y) in zip(data, train_label)] for data in train_data]
    # [[vocab.add_sentence(x, y) for (x, y) in zip(data, valid_label)] for data in valid_data]
    # [[vocab.add_sentence(x, y) for (x, y) in zip(data, fb_test_label)] for data in fb_test_data]
    # [[vocab.add_sentence(x, y) for (x, y) in zip(data, tw_test_label)] for data in tw_test_data]

    return vocab


def load_embedding(config, vocab):
    embeddings_index = {}
    with open(config.embedding_file, encoding = 'utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = coefs

    scale = np.sqrt(3.0 / config.embedding_size)
    embedding = np.random.uniform(-scale, scale, (vocab.n_words, config.embedding_size))

    for word, vector in embeddings_index.items():
        if word in vocab.word2id:
            embedding[vocab.word2id[word]] = vector

    return embedding


class Config(object):
    def __init__(self):
        self.train_file = './data/english/agr_en_train.csv'
        self.valid_file = './data/english/agr_en_dev.csv'
        self.test_file = './data/english/test_file.csv'

        self.max_sent = 6
        self.max_word = 35


# if __name__ == '__main__':
#     config = Config()
#     test_dataset, vocab = load_data(config)
    # x = [[torch.Tensor([2, 3, 4]), torch.Tensor([5, 6, 7, 8, 9])],
    #      [torch.Tensor([5, 6, 7, 8])]]
    # x = pad_sequence(x, True)
    # print(x)
    # x = 10

