#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

from sklearn.metrics import classification_report
import pandas as pd
from train import Trainer
from utils import *
import c_data_utils
from torch.utils.data import DataLoader
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_data(config, vocab):
    test_df = pd.read_csv(config.test_file, header = 0, names = ['face_id', 'content', 'label'])
    w_test_data, c_test_data, test_label = c_data_utils.build_data_c(test_df['content'], test_df['label'])

    w_test_input = [[[vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_test_data]
    c_test_input = [[[[vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_test_data]
    w_test_input = c_data_utils.pad_sequence(w_test_input, True, config.max_sent, config.max_word)
    c_test_input = c_data_utils.pad_sequence_c(c_test_input, config.max_sent, config.max_word, 25)
    test_label = [vocab.tag_to_id(label) for label in test_label]

    test_dataset = c_data_utils.MyDataset(w_test_input, c_test_input, test_label)

    return test_dataset


if __name__ == "__main__":
    args = load_from_pickle('./checkpoint/config.pt')
    logger = get_logger(args.log_dir + "NER_Test_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(args)
    logger.info('load embedding......')
    embedding = c_data_utils.load_embedding(args, args.vocab)
    embedding = torch.tensor(embedding, dtype = torch.float)
    logger.info('load embedding done')

    trainer = Trainer(args, w_embedding = embedding)
    trainer.load()

    test_dataset = load_data(args, args.vocab)
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size = args.batch_size,
                             shuffle = False)

    weighted_f1, macro_f1, p, r, acc, during_time, pred_labels, gold_labels = trainer.eval(test_dataset)

    logger.info("Test Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                format(acc, p, r, weighted_f1, int(during_time)))
    logger.info(classification_report(gold_labels, pred_labels))
