# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import jieba

# 这个类是用来记录词表的，每次对一句话进行分词，并加入词表中
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:'UNK', 3:'SBV', 4:'VOB', 5:'POB', 6:'ADV', 7:'CMP', 8:'ATT', 
                           9:'F', 10:'COO', 11:'DBL', 12:'DOB', 13:'VV', 14:'IC', 15:'MT', 16:'HED'}
        self.n_words = 16  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in jieba.cut(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def readfile(file,):
    
    #read data file
    lines = open(file,encoding = 'utf-8').read().strip().split('\n')
    
    pairs = [[l.split('\t')[0],l.split('\t')[0]] for l in lines]
    
    
    #实例化Lang类
    input_lang = Lang('input')
    output_lang = Lang('output')
    
    return input_lang, output_lang, pairs
    


MAX_LENGTH = 200

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    # 分词后，太长的去掉，这里设置的长度是200
    return len(jieba.lcut(p[0])) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(input_lang,output_lang,pairs):
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


#input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#print(random.choice(pairs))

import os
import torch
import gensim
import numpy as np
from gensim.models import word2vec

cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
# dep_embedding = os.path.join(cur, 'emb/dep_vec_10.bin')
# token_embedding = os.path.join(cur, 'emb/token_vec_300.bin')
embedding_path = os.path.join(cur, 'emb/sgns.wiki.bigram-char.bz2')

def get_vector_fix(word, model, unk_token='UNK'):
    try:
        return model.get_vector(word)
    except KeyError:
        return model.get_vector(unk_token)

def get_weight(path, embed_size, lang):
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False, encoding='utf-8')
    vocab_size = lang.n_words
    weight = torch.zeros(vocab_size, embed_size)

    for i in range(len(wvmodel.index_to_key)):
        try:
            index = lang.word2index[wvmodel.index_to_key[i]]
        except:
            continue
        word = lang.index2word[lang.word2index[wvmodel.index_to_key[i]]]

        weight[index, :] = torch.from_numpy(get_vector_fix(word, wvmodel,'UNK'))
    return weight

if __name__ =='__main__':
    input_lang, output_lang, pairs = readfile(r'./测试文本/orig.txt')
    input_lang, output_lang, pairs = prepareData(input_lang, output_lang, pairs)
    # token_weight = get_weight(token_embedding, 300, input_lang)
    # dep_weight = get_weight(dep_embedding, 10, input_lang)
    weight = get_weight(embedding_path, 300, input_lang)
    print(f"we get the weight, the shape is {weight.shape}")

