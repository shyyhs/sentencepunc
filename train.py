import sys
import os
import time
import math
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ClassDataset
from model import *
from tensorboardX import SummaryWriter

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
    if (os.path.exists(config.model_path) == 0):
        os.makedirs(config.model_path)
    return config

def train(encoder, decoder, train_dataset, dev_dataset, args):
    encoder.train()
    decoder.train()
    if (args.device == "cuda"):
        encoder.cuda()
        decoder.cuda()
    writer = SummaryWriter(args.log_path + args.writer_name)
    optimizer = optim.Adam(list(encoder.parameters())+ list(decoder.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # statistic variable
    step = 0
    batch_num = 0
    sentence_num = 0
    pre_sentence_num = 0
    start_time = time.time()

    # dev
    dev_dataset.shuffle()
    dev_gene = dev_dataset.get_batch()

    for i in range(args.epoch):
        train_dataset.shuffle()
        for sample in train_dataset.get_batch():
            batch_num+=1

            sentence_num += args.batch_size
            if (sentence_num - pre_sentence_num > 10000):
                print ("epoch:{} trained sentences:{} time:{}".format(i,sentence_num,time.time()-start_time))
                pre_sentence_num = sentence_num
            

            sentence = sample["sentence"]
            source = sample['source']
            target = sample['target']
            sentence_len = len(source[0])
            
            #target = np.array(Tensor.cpu(target))
            tot_loss = 0
            h_j, c_j = encoder.init_hidden(args.batch_size)
            for j in range(sentence_len):
                """
                 target_j : batch
                 source_j : batch * 1 
                 h,c : layer_num * batch_size * hidden_size
                 h_final: batch_size * hidden_size
                 predict_j: batch_size * 2
                """
                step += 1
                target_j = target[:,j]
                source_j = source[:,j].unsqueeze(1)

                (h_j, c_j) = encoder.forward_step(source_j, h_j, c_j)

                h_final = h_j[0,:,:]
                
                predict_j = decoder.forward(h_final)

                optimizer.zero_grad()
                loss = criterion(predict_j, target_j)
                tot_loss += loss

            tot_loss/=sentence_len
            tot_loss.backward()
            optimizer.step()
            writer.add_scalar("train_loss", tot_loss.item(), step)
            print (tot_loss.item())

#------------------dev part-----------------
            print (batch_num%args.dev_per_iter)
            if (batch_num % args.dev_per_iter == 0):
                encoder.eval()
                decoder.eval()
                try:
                    dev_sample = next(dev_gene)
                except:
                    dev_dataset.shuffle()
                    dev_gene = dev_dataset.get_batch()
                    dev_sample = next(dev_gene)

                sentence = dev_sample["sentence"]
                source = dev_sample['source']
                target = dev_sample['target']
                sentence_len = len(source[0])

                tot_loss = 0
                h_j, c_j = encoder.init_hidden(args.batch_size)
                for j in range(sentence_len):
                    step += 1
                    """
                     target_j : batch
                     source_j : batch * 1
                     h,c : layer_num * batch_size * hidden_size
                     h_final: batch_size * hidden_size
                     predict_j: batch_size * 2
                    """
                    target_j = target[:,j]
                    source_j = source[:,j].unsqueeze(1)

                    (h_j, c_j) = encoder.forward_step(source_j, h_j, c_j)

                    h_final = h_j[0,:,:]

                    predict_j = decoder.forward(h_final)

                    optimizer.zero_grad()
                    loss = criterion(predict_j, target_j)
                    tot_loss+=loss

                tot_loss/=sentence_len
                print("dev_loss",tot_loss.item())
                writer.add_scalar("dev_loss", tot_loss.item(), step)
                del tot_loss
                encoder.train()
                decoder.train()
#------------------dev part finish-----------------

        torch.save(encoder.state_dict(),os.path.join(args.model_path,"encoder_{}".format(i)))
        torch.save(decoder.state_dict(),os.path.join(args.model_path,"decoder_{}".format(i)))


            #print (source)

            #optimizer.zero_grad()
            #loss = criterion(predict, target)
            #loss.backward()
            #optimizer.step()
            #writer.add_scalar("loss",loss.item(), step)

        #if (i % args.save_iter == 0):
        #    torch.save(model.state_dict(), args.log_path + "{}_{}.pt".format(args.writer_name,i))

args = parser_init()
print (args)
train_dataset = ClassDataset(args.train_path, args.train_num, args.vocab_path, args)
dev_dataset = ClassDataset(args.dev_path, args.dev_num, args.vocab_path, args)
encoder = LSTMEncoder(train_dataset.dict_size , args)
decoder = LinearDecoder(args)

print (train_dataset)
print (encoder)
print (decoder)

train(encoder, decoder, train_dataset, dev_dataset, args)
