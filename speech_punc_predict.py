import sys
import os
import time
import argparse
import yaml
import math

import numpy as np
import torch
import torch.nn as nn
from dataset import ClassDataset
from model import *

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, "r")))

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
    if (os.path.exists(config.model_path) == 0):
        os.makedirs(config.model_path)
    return config

def predict(encoder, decoder, test_dataset, args):
    encoder.eval()
    decoder.eval()
    if (args.device == "cuda"):
        encoder.cuda()
        decoder.cuda()

    with open(args.speech_save_path, "w") as f:
        for i,sample in enumerate(test_dataset.get_batch()):
            sentence = sample["sentence"]
            sentence = sentence[0].split()
            source = sample['source']
            target = sample['target'][0]
            sentence_len = len(source[0])
            output_sentence = ""

            print (i,sentence_len)
            print (sentence)
            h_j, c_j = encoder.init_hidden(args.batch_size)
            for j in range(sentence_len):
                source_j = source[:,j].unsqueeze(1)
                (h_j, c_j) = encoder.forward_step(source_j, h_j, c_j)
                h_final = h_j[0,:,:]
                predict_j = decoder.forward(h_final)

                pred = predict_j[0][1] > predict_j[0][0]
                tgt = target[j]

                output_sentence += sentence[j] + ' '
                print (sentence[j],end="")
                if (pred==1):
                    output_sentence += "。 "
                    print ("。",end="")
            print()
            f.write(output_sentence)
            f.write('\n')

args = parser_init()
args.batch_size = 1
print (args)

test_dataset = ClassDataset(args.speech_path, args.speech_num, args.vocab_path, args)
print ("dataset loaded")
encoder = LSTMEncoder(test_dataset.dict_size , args)
decoder = LinearDecoder(args)

encoder_model_path = os.path.join(args.model_path, "encoder_{}".format(args.load_iter))
decoder_model_path = os.path.join(args.model_path, "decoder_{}".format(args.load_iter))

print (encoder_model_path)
print (decoder_model_path)

encoder.load_state_dict(torch.load(encoder_model_path))
decoder.load_state_dict(torch.load(decoder_model_path))

print ("Model loaded")

predict(encoder, decoder, test_dataset, args)
