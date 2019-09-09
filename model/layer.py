#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, dropout_prob, is_bidirectional, attention_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout_prob = dropout_prob
        self.is_directonal = is_bidirectional
        self.attention_size = attention_size

        if self.is_directonal:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layer,
                            dropout = self.dropout_prob,
                            bidirectional = self.is_directonal,
                            batch_first = True)
        self.drop = nn.Dropout(self.dropout_prob)
        self.linear_1 = nn.Linear(self.hidden_size * self.num_directions, self.attention_size)
        self.linear_2 = nn.Linear(self.hidden_size * self.num_directions * self.num_layer, self.attention_size)
        self._reset_parameter()

    def _reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _masking(self, score, mask, score_mask_value=-np.inf):
        """
        def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
            score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
            score_mask_values = score_mask_value * tf.ones_like(scores)
            return tf.where(score_mask, scores, score_mask_values)
        如果没有被mask 则从score中选择，否则使用-np.inf进行填充
        :param score: alpha 参数
        :param mask: 二维结构
        :return:
        """
        score_mask_values = score_mask_value * torch.ones_like(score)
        return torch.where(mask, score, score_mask_values)

    def attention_net(self, lstm_output, final_state, mask):
        """

        :param lstm_out:  batch_size, seq_len, hidden_size * num_directions
        :param final_state: batch_size, num_layers * num_directions, hidden_size
        :return:
        """

        hidden = torch.reshape(final_state, [-1, self.num_layer * self.num_directions * self.hidden_size])
        hidden = self.linear_2(hidden)
        # print(hidden.size())
        atten_lstm = torch.tanh(self.linear_1(lstm_output))
        # print(atten_lstm.size())
        atten_weights = torch.bmm(atten_lstm, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(atten_weights, 1)
        w = soft_attn_weights * (mask.float())
        weights = w / (w.sum(1, keepdim=True) + 1e-13)

        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, x, mask):
        """

        :param x: batch_size, seq_len, embedding_size
        :param mask: batch_size, seq_len
        :return:
            batch_size, num_directions * hidden_size
        """
        # length = mask.sum(1)
        # sorted_length, idx = length.sort(0, descending=True)
        # x = x[idx]
        #
        # x = nn.utils.rnn.pack_padded_sequence(x, sorted_length, batch_first=True)
        lstm_output, (h_n, c_n) = self.lstm(x)
        # lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # print('lstm_output size', lstm_output.size())
        #
        # _, idx = idx.sort(0, descending=False)
        # lstm_output = lstm_output[idx]
        output = self.attention_net(lstm_output, h_n, mask)
        return output


# if __name__ == '__main__':
    # input = torch.LongTensor([[[2, 3, 4, 0, 0], [5, 6, 7, 9, 10]], [[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]]])
    # mask = input.ne(0).byte()
    # mask = mask.view(-1, mask.size(2))
    # print('mask', mask)
    # # print(mask)
    # # word_mask = mask.reshape(-1)
    # # print(word_mask)
    # # sent_mask = mask.sum(2)
    # # print(sent_mask)
    # # sent_mask = sent_mask.ne(0).byte()
    # # print(sent_mask)
    # model = LSTM(10, 15, 1, 0.1, True, 10)
    # embedding = nn.Embedding(20, 10)
    # x = embedding(input)
    # print(x.size())
    # print('embedding x', x.size())
    # x = x.view(-1, 5, 10)
    # output = model(x, mask)
    # print(output.size())

