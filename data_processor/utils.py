#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

# 针对数据进行进行处理
import emoji
from textblob import TextBlob
import re
import  nltk
from nltk.stem.porter import *
import pandas as pd
import numpy as np
from nltk.metrics import segmentation
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize = ['url', 'email', 'percent', 'money', 'phone', 'user',
                 'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate = {"hashtag", "allcaps", "elongated", "repeated",
                'emphasis', 'censored'},
    fix_html = True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter = "twitter_2018",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector = "twitter_2018",

    unpack_hashtags = True,  # perform word segmentation on hashtags
    unpack_contractions = True,  # Unpack contractions (can't -> can not)
    spell_correct_elong = False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer = SocialTokenizer(lowercase = True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts = [emoticons]
)


def remove_label(text_string):
    """
    removes some labels
    :param text_string:
    :return:
    """
    pattern = '<[/]?allcaps>|<[/]?hashtag>|<elongated>|<repeated>|<url>|<user>|<email> | ' \
              '<percent>|<money>|<phone>|<time>|<date>|<number>|<emphasis>|<censored>'
    parsed_string = re.sub(pattern, '', text_string)
    return parsed_string


def encode_label(label):
    if label == 'OFF':
        return int(1)
    else:
        return int(0)


def emoji2word(sentence):
    """
    encode emoji to word
    :param sentence:
    :return:
    """
    return emoji.demojize(sentence)


def spell_checker(sentence):
    """
    check for spelling errors
    :param sentence:
    :return:
    """
    return TextBlob(sentence).correct()


def remove_url(sentence):
    """
    Accepts a text string and replaces:
    1)urls with URLHERE
    2)lots of whitespace with one instance
    3)mentions with MENTIONSHERE

    this allows us to get standrdized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = 'URL'
    mention_regex = '@[\w\-]+'
    parsed_string = re.sub(space_pattern, ' ', sentence)
    parsed_string = re.sub(giant_url_regex, '', parsed_string)
    parsed_string = re.sub(mention_regex, '', parsed_string)
    return parsed_string


def tokenize(sentence):
    """
      removes punction & excess whitespace, sets to lowercase, and stems tweets. Returns a list of stemmed tokens
    """
    stemmer = PorterStemmer()
    tweet = " ".join(sentence.split('[a-zA-Z.,!?]*')).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return ' '.join(tokens)


def basic_tokenize(sentence):
    """
    same as tokenize but without the stemming
    """
    tweet = ' '.join(re.split('[^a-zA-Z.,!?]*', sentence)).strip()
    return tweet


def clean_str(sentence):
    """

    :param sentence:
    :return:
    """
    # pattern = r'[鈥?"#$%&!()*+,-./:;<=>?@[\\]^_`{|}~]+'
    # r"[(),!?\";:.@&|/-==>\[\]{}~#%]+"
    text = re.sub(r"[(),!?\";:.@&|/-==>\[\]{}~#%_“”‘’\-]+", " ", sentence)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def data_processor(file_name, new_file_name):
    df = pd.read_csv(file_name, header = 0, names = ['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'], sep = '\t')
    df['a_label'] = df.apply(lambda row: encode_label(row.subtask_a), axis = 1)
    df['content'] = df.apply(lambda row: remove_url(row.tweet), axis = 1)
    df['content'] = df.apply(lambda row: text_processor.pre_process_doc(row.content), axis = 1)
    df['content'] = df.apply(lambda row: remove_label(row.content), axis = 1)
    df['content'] = df.apply(lambda row: emoji2word(row.content), axis = 1)
    df['content'] = df.apply(lambda row: clean_str(row.content), axis = 1)
    a_data = {'id': df['id'], 'tweet': df['content'], 'label': df['a_label']}
    context = pd.DataFrame(a_data)
    context.to_csv(new_file_name, index = False)


if __name__ == '__main__':
    file_name = '../data/testset-levelc.tsv'
    new_file_name = '../data/testset_c.csv'
    data_processor(file_name, new_file_name)

