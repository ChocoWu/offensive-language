#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import pickle
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
import pandas as pd
from pylab import mpl
mpl.rcParams['mathtext.fontset'] = 'cm'


def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))

    return model


def get_logger(pathname):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_to_pickle(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()

    return 1


def load_from_pickle(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


# classes = ['NAG', 'CAG', 'OAG']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def decode_label(label):
    if label == '3':
        return 'OAG'
    elif label == '2':
        return 'CAG'
    else:
        return 'NAG'


def plot_embeddings(embeddings, targets, save_name, config, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(3):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend([decode_label(config.id2tag[0]), decode_label(config.id2tag[1]), decode_label(config.id2tag[2])])
    plt.savefig(save_name)


def extract_embeddings(dataloader, model, config):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        with torch.no_grad():
            if config.use_type == 'char':
                for w_inputs, c_inputs, target in dataloader:
                    if config.use_gpu:
                        w_inputs = w_inputs.cuda()
                        c_inputs = c_inputs.cuda()
                        # labels = labels.cuda()
                        mask = w_inputs.ne(0).byte()
                        word_mask = mask.reshape(-1, mask.size(2))
                        sent_mask = mask.sum(2).ne(0).byte()
                        embeddings[k:k + len(w_inputs)] = model(x_word=w_inputs, word_mask=word_mask,
                                                                sent_mask=sent_mask, x_char=c_inputs,
                                                                word=None)[0].cpu().numpy()
                        labels[k:k + len(w_inputs)] = target.numpy()
                        k += len(w_inputs)
            elif config.use_type == 'elmo':
                for inputs, data, target in dataloader:
                    if config.use_gpu:
                        inputs = inputs.cuda()
                        data = data.cuda()
                        # labels = labels.cuda()
                    mask = inputs.ne(0).byte()
                    word_mask = mask.view(-1, mask.size(2))
                    sent_mask = mask.sum(2).ne(0).byte()
                    embeddings[k:k + len(inputs)] = model(x_word=inputs, word_mask=word_mask,
                                                          sent_mask=sent_mask, x_char=None, word=data)[0].cpu().numpy()
                    labels[k:k + len(inputs)] = target.numpy()
                    k += len(inputs)
            else:
                for inputs, target in dataloader:
                    if config.use_gpu:
                        inputs = inputs.cuda()
                        # data = data.cuda()
                        # labels = labels.cuda()
                    mask = inputs.ne(0).byte()
                    word_mask = mask.view(-1, mask.size(2))
                    sent_mask = mask.sum(2).ne(0).byte()
                    embeddings[k:k + len(inputs)] = model(x_word=inputs, word_mask=word_mask,
                                                          sent_mask=sent_mask)[0].cpu().numpy()
                    labels[k:k + len(inputs)] = target.numpy()
                    k += len(inputs)

    return embeddings, labels


# train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_baseline, train_labels_baseline)
# val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_baseline, val_labels_baseline)


def visual_loss(loss, save_name):
    plt.figure()
    x = [i for i in range(len(loss))]
    plt.plot(x, loss, label="$loss$", color='red', linewidth=2)
    plt.legend()
    plt.savefig(save_name)
    # plt.show()


def visual_f1(dev_acc, test_acc, fb_acc, tw_acc, save_name):
    plt.figure()
    assert len(dev_acc) == len(test_acc) and len(test_acc) == len(fb_acc) and len(fb_acc) == len(tw_acc)
    x = [i for i in range(len(dev_acc))]
    plt.plot(x, dev_acc, label = "dev_acc", color = colors[0], linewidth = 2)
    plt.plot(x, test_acc, label = "test_acc", color = colors[1], linewidth = 2)
    plt.plot(x, fb_acc, label = "fb_acc", color = colors[2], linewidth = 2)
    plt.plot(x, tw_acc, label = 'tw_acc', color = colors[3], linewidth = 2)

    plt.legend()
    plt.savefig(save_name)


def visual_confusion_matrix(predict, target, config, save_name):
    confusion_m = confusion_matrix(target, predict)
    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = confusion_m[i, :] / float(confusion_m[i, :].sum())
    names = [[decode_label(config.id2tag[0]), decode_label(config.id2tag[1]), decode_label(config.id2tag[2])]]
    confusion_df = pd.DataFrame(matrix_proportions, index = names, columns = names)
    plt.figure(figsize = (5, 5))
    seaborn.heatmap(confusion_df, annot = True, annot_kws = {'size': 12}, cmap = 'gist_gray_r', cbar = False,
                    square = True, fmt = '.2f')
    plt.ylabel(r'True categories', fontsize = 14)
    plt.xlabel(r'Predicted categories', fontsize = 14)
    plt.tick_params(labelsize = 12)
    plt.savefig(save_name)

