#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from torchtext.vocab import Vectors,GloVe
from torchtext import data
from torchtext import datasets
from torchtext.data import TabularDataset
import pandas as pd
import numpy as np
import os
import pickle


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    def get_pandas_df(self, filename, mode='train'):
        if mode == 'train':
            return pd.read_csv(filename, header = 0, names = ['id', 'tweet', 'label'])
        else:
            return pd.read_csv(filename, header = 0, names = ['id', 'tweet'])

    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        """
        Loads the data from fields
        sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        :param w2v_file: path to file to containing word embedding(Glove or Word2Vec)
        :param train_file: path to training file
        :param test_file: path to test file
        :param val_file: path to valid file, if valid file don't exist, split from training file
        :return:
        """
        tokenizer = lambda x: x.split()

        # Creating fields for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len,
                          batch_first=True)
        LABEL = data.Field(sequential = False, use_vocab = False)
        # LABEL = data.LabelField()
        tv_datafields = [('id', None),
                         ('tweet', TEXT),
                         ('label', LABEL)]
        test_datafields = [('id', None),
                           ('tweet', TEXT)]
        train_df = self.get_pandas_df(train_file, mode = 'train')
        train_examples = [data.Example.fromlist(i, tv_datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, tv_datafields)

        test_df = self.get_pandas_df(test_file, mode = 'test')
        test_examples = [data.Example.fromlist(i, test_datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, test_datafields)

        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, tv_datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, tv_datafields)
        else:
            train_data, val_data = train_data.split(split_ratio = 0.8)

        if os.path.exists(self.config.vocab_file) and os.path.exists(self.config.embedding_file):
            self.word_embeddings = pickle.load(open(self.config.vocab_file, 'rb'))
            self.vocab = pickle.load(open(self.config.embedding_file, 'rb'))
        else:
            TEXT.build_vocab(train_data, vectors = Vectors(w2v_file))
            self.word_embeddings = TEXT.vocab.vectors
            self.vocab = TEXT.vocab
            pickle.dump(self.vocab, open(self.config.vocab_file, 'wb'))
            pickle.dump(self.word_embeddings, open(self.config.embedding_file, 'wb'))

        print('Size of vocabulary: ', len(self.vocab))

        self.train_iterator = data.BucketIterator(train_data,
                                                  batch_size = self.config.batch_size,
                                                  sort_key = lambda x: len(x.tweet),
                                                  repeat = False,
                                                  shuffle = True)
        self.valid_iterator, self.test_iterator = data.BucketIterator.splits((val_data, test_data),
                                                                             batch_size = self.config.batch_size,
                                                                             sort_key = lambda x: len(x.tweet),
                                                                             repeat = False,
                                                                             shuffle = False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


class BatchWrapper(object):
    def __init__(self, dl, x_var, y_vars):
        self.dl = dl
        self.x_var = x_var
        self.y_vars = y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)

            if self.y_vars is not None:
                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                y = torch.cat(temp, dim = 1).float()
            else:
                y = torch.zeros((1))
            yield (x, y)

    def __len__(self):
        return len(self.dl)



