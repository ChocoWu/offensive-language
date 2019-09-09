#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.train_file = ''
        self.valid_file = ''
        self.test_file = ''

        self.embedding_size = 300
        self.seed = 123
        self.batch_size = 64
        self.epoch = 100
        self.lr = 0.0001
        self.weight_decay = 1e-5
        self.patience = 8
        self.freeze = 20
        self.use_gpu = False

        # word-bilstm-attention
        self.w_num_layer = 1
        self.w_hidden_size = 200
        self.w_atten_size = 100
        self.w_is_directional = True
        self.w_drop_prob = 0.1

        # sentence-bilstm-attention
        self.s_num_layer = 1
        self.s_hidden_size = 200
        self.s_atten_size = 100
        self.s_is_directional = True
        self.s_drop_prob = 0.1

        # save file
        self.save_dir = ''
        self.log_dir = ''
        self.config_path = ''
        self.continue_train = False
        self.pretrain_embedding = False
        self.embedding_path = ''


if __name__ == '__main__':
    config = Config()
    Config.epoch = 10
    print(dir(config))
    # config.epoch = 10
    print(config.__getattribute__('epoch'))  # 10
    print(config.epoch)
    c = Config()
    print(c.epoch)

