import os
import sys
import gensim
import yaml
import argparse
import torch
import torch.nn as nn

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path,"r")))


def parser_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="where is the config file")
    args = parser.parse_args()
    config = read_config(args.config)
    args = vars(args)
    for key in args:
        if (key not in config):
            config[key] = args[key]

    if (os.path.exists(config.log_path) == 0):
        os.makedirs(config.log_path)
    return config


class LSTMEncoder(nn.Module):
    def __init__(self, dict_size, args):
        super(LSTMEncoder, self).__init__()
        # init
        self.input_size = dict_size
        self.hidden_size = args.hidden_size
        self.encoder_layer_num = args.encoder_layer_num
        self.dropout_rate = args.dropout_rate
        self.device = args.device
        self.args = args

        self.emb_size = args.emb_size
        self.embedding = nn.Embedding(self.input_size, self.emb_size)

        self.lstm = nn.LSTM(input_size = self.emb_size, \
            hidden_size = self.hidden_size, \
            num_layers = self.encoder_layer_num, \
            dropout = self.dropout_rate) \

    def forward(self, inputs, ori_sentence=None):
        """ 
            inputs: [batch_size, len]
            embedded: [batch_size, len, hidden_size]
            output: [batch_size, len, hidden_size]
        """
        batch_size = inputs.shape[0]
        embedded = self.embedding(inputs)

        h0, c0 = self.init_hidden(batch_size)
        output, (hn, cn) = self.lstm(embedded, (h0,c0))

        return output, hn


    def forward_step(self, input_word, ht, ct):
        """
            input_word: [batchsize, 1]
            embedded: [batchsize, 1, hiddensize] like: (16, 1, 512)
            embedded_trans: [1, batchsize, hiddensize]
        """
        embedded = self.embedding(input_word)
        embedded_trans = torch.transpose(embedded, 0, 1)
        output, (ht_new, ct_new) = self.lstm(embedded_trans,(ht,ct))

        return ht_new, ct_new


    def init_hidden(self, batch_size):
        """
            h0,c0: [encoder_layer_num, batch_size, hidden_len]
        """

        h0 = torch.rand(self.encoder_layer_num,
                batch_size,
                self.hidden_size,
                device = self.device)
        c0 = torch.rand(self.encoder_layer_num,
                batch_size,
                self.hidden_size,
                device = self.device)

        return h0,c0

def main():
    args = parser_init()
    print (args)
    lstm = LSTMEncoder(50003, args)
    print (lstm)

if (__name__ == "__main__"):
    main()

