#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn
import argparse
import time
import c_data_utils
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.model import Hi_Attention
from model.model import ClassificationNet
from tqdm import tqdm
from utils import *
from losses import OnlineTripletLoss
from selector import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def eval_metric(pred_label, gold_label):
    weighted_f1 = metrics.f1_score(gold_label, pred_label, average = 'weighted')
    macro_f1 = metrics.f1_score(gold_label, pred_label, average = 'macro')
    p = metrics.precision_score(gold_label, pred_label, average = 'weighted')
    r = metrics.recall_score(gold_label, pred_label, average = 'weighted')
    acc = metrics.accuracy_score(gold_label, pred_label)

    # report = classification_report(gold_label, pred_label)
    # print(report)
    #
    # confusion_matrix = confusion_matrix(gold_label, pred_label)
    # print(confusion_matrix)
    #
    # matrix_proportions = np.zeros((3, 3))
    # for i in range(0, 3):
    #     matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
    # names = ['NAG', 'CAG', 'OAG']
    # confusion_df = pd.DataFrame(matrix_proportions, index = names, columns = names)
    # plt.figure(figsize = (5, 5))
    # seaborn.heatmap(confusion_df, annot = True, annot_kws = {'size': 12}, cmap = 'gist_gray_r', cbar = False,
    #                 square = True, fmt = '.2f')
    # plt.ylabel(r'True categories', fontsize = 14)
    # plt.xlabel(r'Predicted categories', fontsize = 14)
    # plt.tick_params(labelsize = 12)
    # plt.show()

    return weighted_f1, macro_f1, p, r, acc


def calculate_weights(inputs, classes):
    """
    calculate the weights of inputs
    根据每个类别数的倒数计算其权重
    :param inputs:
    :param classes: 类别数
    :return:
    """
    weights = []
    for i in range(classes):
        weights.append(inputs.size(0)/(inputs.size(0)-inputs.ne(i).sum().item()))
    return


class Trainer(object):
    def __init__(self, config, w_embedding, c_embedding=None, l_embedding=None):
        self.config = config
        self.l_embedding = l_embedding
        self.model = Hi_Attention(self.config, w_embedding = w_embedding, c_embedding = c_embedding)
        self.classification_net = ClassificationNet(self.config)
        self.triplet_selector = HardestNegativeTripletSelector(self.config.margin)
        self.online_triplet_loss = OnlineTripletLoss(self.config.margin, self.triplet_selector)

        ignored_params = list(map(id, self.model.elmo._scalar_mixes[0].parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        self.optim = optim.Adam(
            [{'params': base_params}, {'params': self.model.elmo._scalar_mixes[0].parameters(), "lr": 1e-2}],
            lr = self.config.lr, weight_decay = config.weight_decay)

        # self.optim = optim.Adam(self.model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        if config.use_gpu:
            self.model = self.model.cuda()
            self.classification_net = self.classification_net.cuda()
            # self.triplet_selector = self.triplet_selector.cuda()

    def __call__(self, train_dataset, valid_dataset, fb_test_dataset, tw_test_dataset):
        best_fb_test_f1 = 0.
        best_tw_test_f1 = 0.
        patience = 0
        loss_arr = []
        for epoch in range(self.config.epochs):
            if self.config.pretrain_embedding:
                if epoch < self.config.freeze:
                    self.model.embedding.weight.requires_grad = False
                else:
                    self.model.embedding.weight.requires_grad = True
            loss, during_time, loss_array, weighted_f1, macro_f1, p, r, acc = self.train(train_dataset)

            for i in self.model.elmo._scalar_mixes[0].scalar_parameters.parameters():
                logger.info(i)
            loss_arr.extend(loss_array)
            logger.info("Epoch: {} Loss: {:.4f} Time: {}".format(epoch, loss, int(during_time)))
            logger.info("Epoch: {} Train Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, acc, p, r, weighted_f1, int(during_time)))

            weighted_f1, macro_f1, p, r, acc, during_time, _, _, _, _ = self.eval(valid_dataset)
            logger.info("Epoch: {} Valid Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, acc, p, r, weighted_f1, int(during_time)))

            fb_test_f1, _, fb_test_p, fb_test_r, fb_test_acc, t_during_time, fb_pred_labels, fb_gold_labels, fb_word_weights, fb_sent_weights = self.eval(fb_test_dataset)
            logger.info("Epoch: {} Facebook Test Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, fb_test_acc, fb_test_p, fb_test_r, fb_test_f1, int(t_during_time)))

            tw_test_f1, _, tw_test_p, tw_test_r, tw_test_acc, t_during_time, tw_pred_labels, tw_gold_labels, tw_word_weights, tw_sent_weights = self.eval(tw_test_dataset)
            logger.info("Epoch: {} Twitter Test Acc: {:.4f} P: {:.4f} R: {:.4f} F1:{:.4f} Time: {}".
                        format(epoch, tw_test_acc, tw_test_p, tw_test_r, tw_test_f1, int(t_during_time)))

            if fb_test_f1 > best_fb_test_f1:
                best_fb_test_f1 = fb_test_f1
                best_tw_test_f1 = tw_test_f1
                patience = 0
                self.save()
                # save word attention and sentence attention
                save_to_pickle('attention.pt', [fb_word_weights, fb_sent_weights, tw_word_weights, tw_sent_weights])
                visual_confusion_matrix(fb_pred_labels, fb_gold_labels, self.config, './image/{}_fb_confusion.jpg'.format(time.strftime("%m-%d_%H-%M-%S")))
                visual_confusion_matrix(tw_pred_labels, tw_gold_labels, self.config, './image/{}_tw_confusion.jpg'.format(time.strftime("%m-%d_%H-%M-%S")))
                logger.info("F1: {:.4f} Model is saved!".format(best_fb_test_f1))

            else:
                patience += 1
            if patience >= self.config.patience:
                break
        logger.info("Best Valid_F1: {:.4f}, Best Facebook Test_F1: {:.4f}, Best Twitter Test_F1: {:.4f}".format(weighted_f1,
                                                                                                                best_fb_test_f1,
                                                                                                                best_tw_test_f1))
        visual_loss(loss_arr, './image/{}_loss.jpg'.format(time.strftime("%m-%d_%H-%M-%S")))

        # 查看其embedding的变化，看是否难以进行分类
        train_embeddings, train_labels = extract_embeddings(train_dataset, self.model, self.config)
        plot_embeddings(train_embeddings, train_labels, './image/{}_train_embedding.jpg'.format(time.strftime("%m-%d_%H-%M-%S")), self.config)
        valid_embedding, valid_labels = extract_embeddings(valid_dataset, self.model, self.config)
        plot_embeddings(valid_embedding, valid_labels, './image/{}_valid_embedding.jpg'.format(time.strftime("%m-%d_%H-%M-%S")), self.config)
        fb_embedding, fb_labels = extract_embeddings(fb_test_dataset, self.model, self.config)
        plot_embeddings(fb_embedding, fb_labels, './image/{}_fb_embedding.jpg'.format(time.strftime("%m-%d_%H-%M-%S")), self.config)
        tw_embedding, tw_labels = extract_embeddings(tw_test_dataset, self.model, self.config)
        plot_embeddings(tw_embedding, tw_labels, './image/{}_tw_embedding.jpg'.format(time.strftime("%m-%d_%H-%M-%S")), self.config)

    def train(self, dataset):
        start_time = time.time()
        self.model.train()

        if self.config.use_type == 'char':
            loss_array, pred_labels, gold_labels = self.train_use_char(dataset)
        elif self.config.use_type == 'elmo':
            loss_array, pred_labels, gold_labels = self.train_use_elmo(dataset)
        else:
            loss_array, pred_labels, gold_labels = self.train_use_word(dataset)

        weighted_f1, macro_f1, p, r, acc = eval_metric(pred_labels, gold_labels)
        during_time = time.time() - start_time
        return np.mean(loss_array), during_time, loss_array, weighted_f1, macro_f1, p, r, acc

    def eval(self, dataset):
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            if self.config.use_type == 'char':
                word_weights, sent_weights, pred_labels, gold_labels = self.eval_use_char(dataset)
            elif self.config.use_type == 'elmo':
                word_weights, sent_weights, pred_labels, gold_labels = self.eval_use_elmo(dataset)
                # print("word_weights:\n", word_weights)
            else:
                word_weights, sent_weights, pred_labels, gold_labels = self.eval_use_word(dataset)

            weighted_f1, macro_f1, p, r, acc = eval_metric(pred_labels, gold_labels)
            during_time = time.time() - start_time
            return weighted_f1, macro_f1, p, r, acc, during_time, pred_labels, gold_labels, word_weights, sent_weights

    def train_use_char(self, dataset):
        """
        train use character-BiLSTM
        :param dataset: [w_inputs, c_inputs, labels]
        :return:
        """
        loss_array = []
        pred_labels = []
        gold_labels = []
        for w_inputs, c_inputs, labels in tqdm(dataset):
            if self.config.use_gpu:
                w_inputs = w_inputs.cuda()
                c_inputs = c_inputs.cuda()
                labels = labels.cuda()
            mask = w_inputs.ne(0).byte()
            word_mask = mask.reshape(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weights, sent_weights = self.model(x_word=w_inputs, word_mask=word_mask,
                                                                sent_mask=sent_mask, x_char=c_inputs,
                                                                label = label_embedding)
            else:
                output, word_weights, sent_weights = self.model(w_inputs, word_mask, sent_mask, c_inputs)
            if self.config.triplet:
                triplet_loss, triplet_len = self.online_triplet_loss(output, labels)
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                # print(loss.size())
                loss = torch.mean(loss) + triplet_loss * self.config.alpha / 2
                loss = torch.mean(loss)
            else:
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                loss = torch.mean(loss)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss_array.append(loss.cpu().item())
        return loss_array, pred_labels, gold_labels

    def train_use_elmo(self, dataset):
        pred_labels = []
        gold_labels = []
        loss_array = []
        # print("use elmo")
        for input1, input2, labels in tqdm(dataset):
            if self.config.use_gpu:
                input1 = input1.cuda()
                input2 = input2.cuda()
                labels = labels.cuda()
            mask = input1.ne(0).byte()
            word_mask = mask.reshape(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                # print('label embedding')
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weights, sent_weights = self.model(x_word=input1, word_mask=word_mask,
                                                                sent_mask=sent_mask, x_char=None,
                                                                word=input2, label=label_embedding)
            else:
                # print('no label embedding')
                output, word_weights, sent_weights = self.model(x_word = input1, word_mask=word_mask,
                                                                sent_mask=sent_mask, x_char = None,
                                                                word=input2)
            if self.config.triplet:
                triplet_loss, triplet_len = self.online_triplet_loss(output, labels)
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                loss = torch.mean(loss) + triplet_loss
                loss = torch.mean(loss)
            else:
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                loss = torch.mean(loss)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss_array.append(loss.cpu().item())
        return loss_array, pred_labels, gold_labels

    def train_use_word(self, dataset):
        """
        train only use word
        :param dataset: [inputs, labels]
        :return:
        """
        pred_labels = []
        gold_labels = []
        loss_array = []
        for inputs, labels in tqdm(dataset):
            if self.config.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            mask = inputs.ne(0).byte()
            word_mask = mask.reshape(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weights, sent_weights = self.model(x_word=inputs, word_mask=word_mask,
                                                                sent_mask=sent_mask, label=label_embedding)
            else:
                output, word_weights, sent_weights = self.model(inputs, word_mask, sent_mask)
            if self.config.triplet:
                triplet_loss, triplet_len = self.online_triplet_loss(output, labels)
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                loss = torch.mean(loss) + triplet_loss
                loss = torch.mean(loss)
                pass
            else:
                output = self.classification_net(output)

                result = torch.max(output, 1)[1]
                pred_labels.extend(result.cpu().numpy().tolist())
                gold_labels.extend(labels.cpu().numpy().tolist())

                loss = self.criterion(output, labels)
                loss = torch.mean(loss)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss_array.append(loss.cpu().item())
        return loss_array, pred_labels, gold_labels

    def eval_use_char(self, dataset):
        """

        :param dataset: [inputs, c_inputs, labels]
        :return:
        """
        pred_labels = []
        gold_labels = []
        word_weights = []
        sent_weights = []
        for inputs, c_inputs, labels in tqdm(dataset):
            if self.config.use_gpu:
                inputs = inputs.cuda()
                c_inputs = c_inputs.cuda()
                labels = labels.cuda()
            mask = inputs.ne(0).byte()
            word_mask = mask.view(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weight, sent_weight = self.model(x_word=inputs, word_mask=word_mask,
                                                              sent_mask=sent_mask, x_char=c_inputs,
                                                              label = label_embedding)
            else:
                output, word_weight, sent_weight = self.model(inputs, word_mask, sent_mask, c_inputs)
            # if self.config.triplet:
            output = self.classification_net(output)
            result = torch.max(output, 1)[1]
            word_weights.extend(word_weight.cpu().numpy())
            sent_weights.extend(sent_weight.cpu().numpy())
            pred_labels.extend(result.cpu().numpy().tolist())
            gold_labels.extend(labels.cpu().numpy().tolist())
        return word_weights, sent_weights, pred_labels, gold_labels

    def eval_use_elmo(self, dataset):
        """

        :param dataset: [inputs, data, labels]
        :return:
        """
        pred_labels = []
        gold_labels = []
        word_weights = []
        sent_weights = []
        for inputs, data, labels in tqdm(dataset):
            if self.config.use_gpu:
                inputs = inputs.cuda()
                data = data.cuda()
                labels = labels.cuda()

            mask = inputs.ne(0).byte()
            word_mask = mask.view(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weight, sent_weight = self.model(x_word=inputs, word_mask=word_mask,
                                                              sent_mask=sent_mask, x_char=None,
                                                              word=data, label = label_embedding)
            else:
                output, word_weight, sent_weight = self.model(x_word=inputs, word_mask=word_mask,
                                                              sent_mask=sent_mask, x_char=None, word=data)
            # if self.config.triplet:
            output = self.classification_net(output)
            result = torch.max(output, 1)[1]
            word_weights.extend(word_weight.cpu().numpy())
            sent_weights.extend(sent_weight.cpu().numpy())
            pred_labels.extend(result.cpu().numpy().tolist())
            gold_labels.extend(labels.cpu().numpy().tolist())
        return word_weights, sent_weights, pred_labels, gold_labels

    def eval_use_word(self, dataset):
        """

        :param dataset: [inputs, labels]
        :return:
        """
        pred_labels = []
        gold_labels = []
        word_weights = []
        sent_weights = []
        for inputs, labels in tqdm(dataset):
            if self.config.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            mask = inputs.ne(0).byte()
            word_mask = mask.view(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            if self.l_embedding is not None:
                label_embedding = torch.tensor(self.l_embedding, dtype = torch.float)
                output, word_weight, sent_weight = self.model(x_word = inputs, word_mask = word_mask,
                                                              sent_mask = sent_mask, label = label_embedding)
            else:
                output, word_weight, sent_weight = self.model(inputs, word_mask, sent_mask)
            # if self.config.triplet:
            output = self.classification_net(output)

            # print(output)
            result = torch.max(output, 1)[1]
            pred_labels.extend(result.cpu().numpy().tolist())
            gold_labels.extend(labels.cpu().numpy().tolist())

            word_weights.extend(word_weight.cpu().numpy())
            sent_weights.extend(sent_weight.cpu().numpy())

        return word_weights, sent_weights, pred_labels, gold_labels

    def save(self):
        torch.save(self.model.state_dict(), self.config.save_dir + 'model.pt')
        torch.save(self.optim.state_dict(), self.config.save_dir + 'optim.pt')

    def load(self):
        self.model.load_state_dict(torch.load(self.config.save_dir + 'model.pt'))
        self.optim.load_state_dict(torch.load(self.config.save_dir + 'optim.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type = str, default = "./data/english/agr_en_train.csv")
    parser.add_argument("--valid_file", type = str, default = "./data/english/agr_en_dev.csv")
    parser.add_argument("--fb_test_file", type = str, default = "./data/english/agr_en_fb_test.csv")
    parser.add_argument("--tw_test_file", type = str, default = "./data/english/agr_en_tw_test.csv")
    # parser.add_argument("--tag_format", type = str, choices = ["bio", "bmes"], default = "bio")

    parser.add_argument("--save_dir", type = str, default = "./checkpoint/")
    parser.add_argument("--log_dir", type = str, default = "./log/")
    parser.add_argument("--config_path", type = str, default = "./checkpoint/config.pt")
    parser.add_argument("--vocab_path", type = str, default = "./checkpoint/vocab.pt")
    parser.add_argument("--embedding_path", type = str, default = "./checkpoint/embedding.pt")
    parser.add_argument("--label_embedding_path", type = str, default = './checkpoint/label_embedding.pt')
    parser.add_argument("--continue_train", type = bool, default = False, help = "continue to train model")
    parser.add_argument("--pretrain_embedding", type = bool, default = True)
    parser.add_argument("--embedding_file", type = str, default = "./data/glove.840B.300d.txt")

    parser.add_argument("--options_file", type = str,
                        default = './checkpoint/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument("--weights_file", type = str,
                        default = './checkpoint/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    parser.add_argument("--use_type", type = str, default = 'elmo')
    parser.add_argument("--seed", type = int, default = 123, help = "seed for random")
    parser.add_argument("--batch_size", type = int, default = 15, help = "number of batch size")
    parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
    parser.add_argument("--embedding_size", type = int, default = 300)
    parser.add_argument("--lr", type = float, default = 0.0003, help = "learning rate of adam")
    parser.add_argument("--weight_decay", type = float, default = 1e-5, help = "weight decay of adam")
    parser.add_argument("--patience", type = int, default = 15)
    parser.add_argument("--freeze", type = int, default = 5)
    parser.add_argument("--num_class", type = int, default = 3)
    parser.add_argument("--dropout_prob", type = float, default = 0.5)
    parser.add_argument("--max_sent", type = int, default = 6)
    parser.add_argument("--max_word", type = int, default = 35)

    parser.add_argument("--triplet", type = bool, default = False)
    parser.add_argument("--margin", type = float, default = 1.0)
    parser.add_argument("--alpha", type = float, default = 0.5)

    parser.add_argument("--w_hidden_size", type = int, default = 150)
    parser.add_argument("--w_num_layer", type = int, default = 1, help = "number of layers")
    parser.add_argument("--w_atten_size", type = int, default = 300)
    parser.add_argument("--w_is_bidirectional", type = bool, default = True)
    parser.add_argument("--w_dropout_prob", type = float, default = 0.5)

    parser.add_argument("--s_hidden_size", type = int, default = 150)
    parser.add_argument("--s_num_layer", type = int, default = 1, help = "number of layers")
    parser.add_argument("--s_atten_size", type = int, default = 300)
    parser.add_argument("--s_is_bidirectional", type = bool, default = True)
    parser.add_argument("--s_dropout_prob", type = float, default = 0.5)

    parser.add_argument("--char_encode_type", type = str, default = 'lstm')
    parser.add_argument("--c_embedding_size", type = int, default = 50)
    parser.add_argument("--c_hidden_size", type = int, default = 20)
    parser.add_argument("--c_num_layer", type = int, default = 1)
    parser.add_argument("--c_is_bidirectional", type = bool, default = True)
    parser.add_argument("--c_dropout_prob", type = float, default = 0.5)
    parser.add_argument("--c_kernel_size", type = list, default = [2])
    parser.add_argument("--num_filter", type = int, default = 2)

    parser.add_argument("--use_gpu", type = bool, default = True)
    args = parser.parse_args()

    logger = get_logger(args.log_dir + "NER_Train_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))

    logger.info(args)

    seed = args.seed
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    logger.info("Loading data...")
    if os.path.exists(args.vocab_path):
        logger.info("vocabulary exists")
        vocab = load_from_pickle(args.vocab_path)
    else:
        logger.info("vocabulary does not exit")
        vocab = c_data_utils.build_vocab(args)
        save_to_pickle(args.vocab_path, vocab)
        logger.info("vocabulary loading is done")

    logger.info("Loading data ......")
    train_dataset = c_data_utils.load_data(args.train_file, args.max_sent, args.max_word, vocab, args.use_type)
    valid_dataset = c_data_utils.load_data(args.valid_file, args.max_sent, args.max_word, vocab, args.use_type)
    fb_test_dataset = c_data_utils.load_data(args.fb_test_file, args.max_sent, args.max_word, vocab,
                                             args.use_type)
    tw_test_dataset = c_data_utils.load_data(args.tw_test_file, args.max_sent, args.max_word, vocab,
                                             args.use_type)

    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset,
                              batch_size = args.batch_size,
                              shuffle = False)
    fb_test_loader = DataLoader(dataset = fb_test_dataset,
                                batch_size = args.batch_size,
                                shuffle = False)
    tw_test_loader = DataLoader(dataset = tw_test_dataset,
                                batch_size = args.batch_size,
                                shuffle = False)
    logger.info("Data loading is done!")

    if args.pretrain_embedding:
        logger.info("Loading word embedding file...")
        if os.path.exists(args.embedding_path):
            w_embedding = load_from_pickle(args.embedding_path)
        else:
            w_embedding = c_data_utils.load_embedding(args, vocab)
            w_embedding = torch.tensor(w_embedding, dtype = torch.float)
            save_to_pickle(args.embedding_path, w_embedding)
        logger.info("Word Embedding loading is done!")

        logger.info("Loading label embedding file ... ")
        if os.path.exists(args.label_embedding_path):
            label_embedding = load_from_pickle(args.label_embedding_path)
        else:
            label_embedding = c_data_utils.load_label_embedding(args, vocab)
            label_embedding = torch.tensor(label_embedding, dtype = torch.float)
            save_to_pickle(args.label_embedding_path, label_embedding)
        logger.info("Label Embedding loading is done!")
    else:
        w_embedding = None
        label_embedding = None

    args.vocab_size = vocab.n_words
    args.n_tags = vocab.n_tags
    args.alphabet_size = vocab.n_chars
    args.id2tag = vocab.id2tag
    print(vocab.id2tag)

    trainer = Trainer(args, w_embedding, l_embedding = label_embedding)

    if args.continue_train:
        logger.info("Loading model...")
        trainer.load()

    save_to_pickle(args.config_path, args)
    logger.info("Configuration data is saved in {}".format(args.config_path))

    logger.info("Start training...")
    trainer(train_loader, valid_loader, fb_test_loader, tw_test_loader)
