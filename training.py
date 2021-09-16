#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:37:24 2021

@author: bring
"""
from model import EncoderRNN,DecoderRNN
import DataPreprocessing
import random

import torch
from torch import nn,optim

import time
import math
import numpy as np

import jieba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 3

MAX_LENGTH = 200

#将sentence转成index
def indexesFromSentence(lang, sentence):

    text = jieba.cut(sentence)
    
    result_index = []
    
    word_index = lang.word2index.keys()
    
    for word in text:
        if word in word_index:
            result_index.append(lang.word2index[word])
        else:
            result_index.append(UNK_token)
            
    return result_index

#将sentence的index转成tensor
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#以pair的形式出现
def tensorsFromPair(input_lang,output_lang,pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder.to(device)
    decoder.to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(1, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].to(device), encoder_hidden.to(device))
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output.to(device), target_tensor[di].to(device))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output.to(device), target_tensor[di].to(device))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder,pairs, n_iters, input_lang,output_lang,print_every=1000, plot_every=100, learning_rate=0.01):
    print('train on '+str(device))
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang,output_lang,random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def text2vec(text,lang,model,device,max_length):
    
    model.eval()
    
    text_idx_vec = tensorFromSentence(lang,text)
    
    encoder_hidden = model.initHidden()
    input_length = text_idx_vec.size(0)
    
    encoder_output = torch.zeros(1, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            text_idx_vec[ei].to(device), encoder_hidden.to(device))
    
    return encoder_output

def similarity(text1,text2,lang,model,device,max_length):

    text_vec1 = text2vec(text1,lang,model,device,max_length).detach().cpu().numpy()
    text_vec2 = text2vec(text2,lang,model,device,max_length).detach().cpu().numpy()
    
    numerator = np.multiply(text_vec1,text_vec2).sum()
    dominator = np.multiply(np.sqrt(np.square(text_vec1).sum()),np.sqrt(np.square(text_vec2).sum()))

    return numerator/dominator

def read_text(text_path):
    f = open(text_path,encoding = 'utf-8')
    return f.read().replace('\n','，')


if __name__ == "__main__":
    input_lang, output_lang, pairs=DataPreprocessing.readfile(r'./测试文本/orig.txt')
    input_lang, output_lang, pairs = DataPreprocessing.prepareData(input_lang, output_lang, pairs)
    
    hidden_size = 256
    
    encoder = EncoderRNN(input_lang.n_words,hidden_size)
    decoder = DecoderRNN(hidden_size,input_lang.n_words)

    trainIters(encoder, decoder,pairs, 75000,input_lang,output_lang, print_every=5000)
    
    text1 = read_text(r'./测试文本/orig.txt')
    text2 = read_text(r'./测试文本/orig_0.8_dis_10.txt')
    
    result = similarity(text1,text2,input_lang,encoder,device,MAX_LENGTH)







