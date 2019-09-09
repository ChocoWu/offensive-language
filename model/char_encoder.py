#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CharEncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, is_bidirectional, dropout_prob,
                 embedding=None):
        super(CharEncoderLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.is_bidirectional = is_bidirectional
        self.dropout_prob = dropout_prob

        self.lstm = nn.LSTM(input_size = self.embedding_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layer,
                            dropout = self.dropout_prob,
                            bidirectional = self.is_bidirectional,
                            batch_first = True)
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding)

        self.drop = nn.Dropout(self.dropout_prob)

    def _reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        """

        :param x: (batch_size * max_sent * max_word, max_char)
        :return:
            (batch_size * max_sent * max_word, num_layers * num_directions * hidden_size)
        """
        # mask = x.ne(0).byte()
        # mask = mask.view(-1, mask.size(3))
        # length = mask.sum(1)
        # sorted_length, idx = length.sort(0, descending = True)
        x = self.embedding(x)
        # x = x[idx]
        #
        # x = nn.utils.rnn.pack_padded_sequence(x, sorted_length, batch_first = True)
        lstm_output, (h_n, c_n) = self.lstm(x)
        h = torch.transpose(h_n, 0, 1)
        h = h.contiguous().view(-1, h.size(1)*h.size(2))
        h = self.drop(h)

        return h


class CharEncoderCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernel_size, num_filters,
                 embedding=None, mode='static', ):
        super(CharEncoderCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.mode = mode
        self.num_filters = num_filters

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        if mode == 'static':
            self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True

        conv_blocks = []
        for i, kernel_size in enumerate(self.kernel_size):
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            # maxpool_kernel_size = sentence_len - kernel_size + 1

            if i == 0:
                conv1d = nn.Conv1d(in_channels = self.embedding_size, out_channels = num_filters, padding = 1,
                                   kernel_size = kernel_size, stride = 1)
            else:
                conv1d = nn.Conv1d(in_channels = num_filters, out_channels = num_filters,
                                   kernel_size = kernel_size, stride = 1)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = kernel_size)
            )
            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)  # ModuleList is needed for registering parameters in conv_blocks

    def forward(self, x):
        """

        :param x: (batch_size * max_sent * max_word, max_char)
        :return:
            default dilation = 1, padding = 0
            L_out = (L_in + 2 * padding - dilation * (kernel_size -1) - 1)/stride + 1
            (batch_size * max_sent * max_word, C_out * L_out)
        """
        x = self.embedding(x)
        # print(x.size())
        x = torch.transpose(x, 2, 1)
        # print(x.size())
        # x = x.tranpose()
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            print(x.size())
        x = x.view(-1, x.size(1)*x.size(2))
        return x


if __name__ == "__main__":
    x = torch.tensor([[2, 3, 4, 0], [4, 5, 6, 7]], dtype = torch.long)
    model = CharEncoderCNN(10, 10, [2], 11, 'non-static')
    print(model)
    output = model(x)
    # for param in model.parameters():
    #     # print(param)
    #     print(type(param.data), param.size())
    for name, param in model.named_parameters():
        print(name, param.size())

    # for each in model.trace:
    #     print(each.shape)
    # output = model(x)
    # print(output)
