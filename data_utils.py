#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Vocabulary(object):
    W_PAD, W_UNK = '<pad>', '<unk>'
    C_PAD, C_UNK = '<pad>', '<unk>'

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.tag2id = {}
        self.id2tag = {}
        self.tag2count = {}

        self._init_vocab()

    def _init_vocab(self):
        for word in [self.W_PAD, self.W_UNK]:
            self.word2id[word] = len(self.word2id)
            self.id2word[self.word2id[word]] = word
            self.word2count[word] = 0

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
                self.tag2count[tag] = 1
            else:
                self.tag2count[tag] += 1

    @property
    def n_words(self):
        return len(self.word2id)

    @property
    def n_tags(self):
        return len(self.tag2id)

    def word_to_id(self, word):
        return self.word2id.get(word, self.word2id[self.W_UNK])

    def tag_to_id(self, tag):
        return self.tag2id[tag]

    def id_to_word(self, id):
        return self.id2word[id]

    def id_to_tag(self, id):
        return self.id2tag[id]


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
    # max_word = max([max([sent.size(0) for sent in doc]) for doc in sequences])
    # max_sent = 2
    # max_word = 4
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

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item], dtype = torch.long), torch.tensor(self.labels[item], dtype = torch.long)

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


def load_data(config, vocab=None):
    train_df = pd.read_csv(config.train_file, header = 0, names = ['face_id', 'content', 'label'])
    valid_df = pd.read_csv(config.valid_file, header = 0, names = ['face_id', 'content', 'label'])
    test_df = pd.read_csv(config.test_file, header = 0, names = ['face_id', 'content', 'label'])

    train_data, train_label = build_data(train_df['content'], train_df['label'])
    valid_data, valid_label = build_data(valid_df['content'], valid_df['label'])
    test_data, test_label = build_data(test_df['content'], test_df['label'])

    if vocab is None:
        vocab = Vocabulary()
        [[vocab.add_sentence(x, y) for (x, y) in zip(data, train_label)] for data in train_data]
        [[vocab.add_sentence(x, y) for (x, y) in zip(data, valid_label)] for data in valid_data]
        [[vocab.add_sentence(x, y) for (x, y) in zip(data, test_label)] for data in test_data]

    train_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in train_data]
    train_input = pad_sequence(train_input, True, config.max_sent, config.max_word)
    train_label = [vocab.tag_to_id(label) for label in train_label]
    valid_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in valid_data]
    valid_input = pad_sequence(valid_input, True, config.max_sent, config.max_word)
    valid_label = [vocab.tag_to_id(label) for label in valid_label]
    test_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in test_data]
    test_input = pad_sequence(test_input, True, config.max_sent, config.max_word)
    test_label = [vocab.tag_to_id(label) for label in test_label]

    train_dataset = myDataset(train_input, train_label)
    valid_dataset = myDataset(valid_input, valid_label)
    test_dataset = myDataset(test_input, test_label)

    return train_dataset, valid_dataset, test_dataset, vocab


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


def count_info(content):
    """
    针对数据进行统计
    :param content:
    :return:
    """
    num_sent = []
    num_word = []

    df = pd.read_csv(content, encoding = 'utf-8', header = 0, names = ['face_id', 'content', 'label'])
    doc = df['content']
    for sents in doc:
        sent = sents.split("|")
        num_sent.append(len(sent))
        temp = []
        for words in sent:
            word = words.split('')
            temp.append(len(word))
        num_word.append(temp)


class Config(object):
    def __init__(self):
        self.train_file = '../data/english/agr_en_train.csv'
        self.valid_file = '../data/english/agr_en_dev.csv'
        self.test_file = '../data/english/agr_en_fb_test.csv'


if __name__ == '__main__':
    # config = Config()
    # load_data(config)
    x = [[torch.Tensor([2, 3, 4]), torch.Tensor([5, 6, 7, 8, 9])],
         [torch.Tensor([5, 6, 7, 8])]]
    x = pad_sequence(x, True)
    print(x)
    x = 10

