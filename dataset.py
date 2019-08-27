import os
import sys
import time
import random
import codecs

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle # shuffle source and target together

class ClassDataset(Dataset):
    """ dataset for classification """
    def __init__(self, srcfile, tgtfile, dictfile, args=None):
        """
        Args:
            srcfile: the path of training data, src (train.txt)
            tgtfile: the path of training data, tgt
            dictfile: the path of dict (vocab.txt)
        """
        # 0. init
        self.srcfile = srcfile 
        self.tgtfile = tgtfile 
        self.dictfile = dictfile

        self.sentence = [] # original sentence
        self.source = [] # list to save the sentences
        self.target = [] # list to save the numbers(0 1 for every word)
        self.dict_size = 0 # how many words in the dataset
        self.data_size = 0 # how many lines in the datafile

        self.w2i = {} # word to index
        self.i2w = {} # index to word

        self.batch_size = 1
        if (hasattr(args, "batch_size") == True):
            self.batch_size = args.batch_size
        self.batch_pos = 0

        self.max_len = 0 # if the length of some sentence > max_len, trunc these sentences to sentence[-max_len:]
        if (hasattr(args, "max_len") == True):
            self.max_len = args.max_len
        self.trunced_line = 0

        # device = gpu(cuda) or cpu
        if (hasattr(args, "device") == False):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if (torch.cuda.is_available()):
                self.device = torch.device(args.device)
            else:
                self.device = torch.device("cpu")

        if (os.path.exists(self.srcfile)==0):
            print ("data file not found!")
            raise ValueError

        if (os.path.exists(self.tgtfile)==0):
            print ("data file not found!")
            raise ValueError

        if (os.path.exists(self.dictfile)==0):
            print ("dict file not found!")
            raise ValueError

        # 1. read the dictionary
        with open(self.dictfile, "r") as f:
            lines = f.readlines()
            self.dict_size = len(lines)
            for i, line in enumerate(lines):
                word = line.strip()

                self.w2i[word] = i # these two lines are important ..
                self.i2w[i] = word

        with open(self.srcfile, "r") as f1:
            with open(self.tgtfile, "r") as f2:
                src_lines = f1.readlines()
                tgt_lines = f2.readlines()
                self.data_size = len(src_lines)
                for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
                    src_line_idx = self._sentence2idx(src_line)
                    tgt_line = [int(num) for num in tgt_line.strip().split(' ')]
                    if (self.max_len > 0 and len(src_line_idx) > self.max_len):
                        src_line_idx = src_line_idx[-self.max_len:]
                        tgt_line = tgt_line[-self.max_len:]

                    self.target.append(tgt_line)
                    self.sentence.append(src_line)
                    self.source.append(src_line_idx)

    def _check(self):
        for src,tgt in zip(self.source, self.target):
            print(src)
            print(tgt)
            if ((len(src)!=len(tgt))):
                return 0
        return 1

    def _tokenize(self, line):
        return line.strip('\n').strip().split()

    def _sentence2idx(self, sentence): # sentence -> list of index
        idx = []
        sentence = self._tokenize(sentence)
        for word in sentence:
            word_id = self.w2i.get(word)
            if (word_id == None): word_id = self.w2i.get('<unk>') # when meets unk
            idx.append(word_id)
        return idx

    def __len__(self):
        return self.data_size

    def __getitem__(self, i): # not in use
        sample = {'source': self.source[i],
                'target': self.target[i]}
        return sample

    def shuffle(self): # see sklearn.utils.shuffle
        self.batch_pos = 0
        self.source, self.target = shuffle(self.source, self.target)

    def get_batch(self):
        while (self.batch_pos + self.batch_size < self.data_size):
            batch_sentence = self.sentence[self.batch_pos: self.batch_pos + self.batch_size]
            batch_source = self.source[self.batch_pos: self.batch_pos + self.batch_size]
            batch_target = self.target[self.batch_pos: self.batch_pos + self.batch_size]
            self.batch_pos += self.batch_size

            s_maxlen = max([len(s) for s in batch_source])
            sentence_maxlen = max([len(s) for s in batch_sentence])

            for i,(s,t) in enumerate(zip(batch_source,batch_target)):
                s = s + [self.w2i['<pad>']] * (s_maxlen - len(s))
                batch_source[i] = s
                t = t + [0] * (s_maxlen - len(t))
                batch_target[i] = t

            batch_source = torch.tensor(batch_source, \
                                        dtype = torch.long, \
                                        device = self.device)
            batch_target = torch.tensor(batch_target, \
                                        dtype = torch.long, \
                                        device = self.device)

            sample = {"source": batch_source,  \
                    "target": batch_target, \
                    "sentence": batch_sentence}

            yield sample # learn the usage of yield


def main():
    srcfile = '/home/song/git/sentencepunc/data/final_data/as.valid'
    tgtfile = '/home/song/git/sentencepunc/data/final_data/as.valid.num'
    dictfile = '/home/song/git/sentencepunc/data/final_data/as.vocab'
    test = ClassDataset(srcfile, tgtfile, dictfile)
    print (test._check())
    print (test.__len__())
    #for i in range(1):
    #    test.shuffle()
    #    print (i)
    #    for sample in test.get_batch():
    #        print (sample["sentence"])
    #        print(sample["source"])
    #        print(sample["target"])


if (__name__ == "__main__"):
    main()
