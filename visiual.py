#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
可视化分析
"""
import pickle
import seaborn
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    atten_path = './checkpoint/attention.pt'
    attention = pickle.load(open(atten_path, 'rb'))
    fb_word_weights, fb_sent_weights, tw_word_weights, tw_sent_weights = attention
    """
    fb_word_weights: [916 * 6, 35]
    fb_sent_weights: [916, 6]
    """
    tweets = pd.read_csv('.', header = 0, names = ['face_id', 'content', 'label'])
    sents = []
    for tweet in tweets:
        sent = tweet.split('|')
        sents.extend(sent)

    print(fb_word_weights[:2])
    print(fb_word_weights[1])
    print(fb_sent_weights[0].shape)
    print(fb_sent_weights[1])
    # print(fb_sent_weights[:10])
    # print(tw_word_weights[:10])
    # print(tw_sent_weights[:10])
