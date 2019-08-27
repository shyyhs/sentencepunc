import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, help="extract dict from which file")
parser.add_argument("--dict_path",type=str, help="path to save the dict")
parser.add_argument("--vocab_size",type=int, default=32000, help="the size of the vocab")
args = parser.parse_args()

# make dictionary
UNK = '<unk>' # unknown word
PAD = '<pad>' # padding
SOS = '<sos>' # start of the sentence
EOS = '<eos>' # end of the sentence

def generate_vocab(data_path, dict_path, vocab_size):
    d = {}
    with open(data_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip('\n').strip().split()
            for word in line:
                d[word] = d.get(word,0) + 1

    dictlist = [[k,v] for k,v in d.items()]
    dictlist.sort(key = lambda x: -x[1]) # sort from the most frequent to least frequent
    dictlist = [x[0] for x in dictlist] # discard frequency(x[1]), use word only
    dictlist = [PAD, UNK, SOS, EOS] + dictlist[0:vocab_size]

    with open(dict_path, "w") as f:
        for key in dictlist:
            f.write(key + '\n')

generate_vocab(args.data_path, args.dict_path, args.vocab_size)
    


