#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn

# Our simplest machine learning model is
# a linear SVM trained on word unigrams


class LinearSVM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_class):
        super(LinearSVM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_class = num_class

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        self.fc = nn.Linear(self.embedding_size, self.num_class)

    def forward(self, x):
        h = self.fc(x)
        return h

