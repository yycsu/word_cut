# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 15 16:37:24 2021

# @author: bring
# """
# from model import EncoderRNN,DecoderRNN
# import DataPreprocessing
# import random

# import torch
# from torch import nn,optim

# import time
# import math
# import numpy as np

# import jieba


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOS_token = 0
# EOS_token = 1
# UNK_token = 3

# MAX_LENGTH = 200

# #将sentence转成index
# def indexesFromSentence(lang, sentence):

#     text = jieba.cut(sentence)
    
#     result_index = []
    
#     word_index = lang.word2index.keys()
    
#     for word in text:
#         if word in word_index:
#             result_index.append(lang.word2index[word])
#         else:
#             result_index.append(UNK_token)
            
#     return result_index

# #将sentence的index转成tensor
# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# #以pair的形式出现
# def tensorsFromPair(input_lang,output_lang,pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)

# teacher_forcing_ratio = 0.5

# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)


# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
#     encoder_hidden = encoder.initHidden()

#     encoder.to(device)
#     decoder.to(device)

#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()

#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)

#     encoder_outputs = torch.zeros(1, encoder.hidden_size, device=device)

#     loss = 0

#     for ei in range(input_length):
#         encoder_output, _ = encoder(
#             input_tensor[ei].to(device))
#         #encoder_outputs[ei] = encoder_output[0, 0]

#     decoder_input = torch.tensor([[SOS_token]], device=device)

#     # decoder_hidden = encoder_hidden

#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, _= decoder(
#                 decoder_input)
#             loss += criterion(decoder_output.to(device), target_tensor[di].to(device))
#             decoder_input = target_tensor[di]  # Teacher forcing

#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, _= decoder(
#                 decoder_input)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input

#             loss += criterion(decoder_output.to(device), target_tensor[di].to(device))
#             if decoder_input.item() == EOS_token:
#                 break

#     loss.backward()

#     encoder_optimizer.step()
#     decoder_optimizer.step()

#     return loss.item() / target_length

# def trainIters(encoder, decoder,pairs, n_iters, input_lang,output_lang,print_every=1000, plot_every=100, learning_rate=0.01):
#     print('train on '+str(device))
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every

#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(input_lang,output_lang,random.choice(pairs))
#                       for i in range(n_iters)]
#     criterion = nn.NLLLoss()

#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]

#         loss = train(input_tensor, target_tensor, encoder,
#                      decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss

#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                          iter, iter / n_iters * 100, print_loss_avg))

#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0


# def text2vec(text,lang,model,device,max_length):
    
#     model.eval()
    
#     text_idx_vec = tensorFromSentence(lang,text)
    
#     input_length = text_idx_vec.size(0)
    
#     encoder_output = torch.zeros(1, encoder.hidden_size, device=device)
#     # 这一段是通过lstm前传的方式，计算出一整段文本的embedding
#     for ei in range(input_length-1):
#         encoder_output, _ = encoder(
#             text_idx_vec[ei].to(device))
    
#     return encoder_output

# def similarity(text1,text2,lang,model,device,max_length):

#     text_vec1 = text2vec(text1,lang,model,device,max_length).detach().cpu().numpy()
#     text_vec2 = text2vec(text2,lang,model,device,max_length).detach().cpu().numpy()
    
#     numerator = np.multiply(text_vec1,text_vec2).sum()
#     dominator = np.multiply(np.sqrt(np.square(text_vec1).sum()),np.sqrt(np.square(text_vec2).sum()))

#     return numerator/dominator

# def read_text(text_path):
#     f = open(text_path,encoding = 'utf-8')
#     return f.read().replace('\n','，')


# if __name__ == "__main__":
#     input_lang, output_lang, pairs=DataPreprocessing.readfile(r'./测试文本/orig.txt')
#     input_lang, output_lang, pairs = DataPreprocessing.prepareData(input_lang, output_lang, pairs)
    
#     hidden_size = 256
    
#     encoder = EncoderRNN(input_lang.n_words,hidden_size)
#     decoder = DecoderRNN(hidden_size,input_lang.n_words)

#     # trainIters(encoder, decoder,pairs, 75000,input_lang,output_lang, print_every=5000)

#     trainIters(encoder, decoder, pairs, 750,input_lang,output_lang, print_every=100, plot_every=50)

#     # text1 = read_text(r'./测试文本/orig.txt')
#     # text2 = read_text(r'./测试文本/orig_0.8_dis_10.txt')

#     text1 = read_text(r'./测试文本/test1.txt')
#     text2 = read_text(r'./测试文本/test2.txt')
    
#     result = similarity(text1,text2,input_lang,encoder,device,MAX_LENGTH)




import os
import time
import torch
import gensim
import torch.nn as nn
import numpy as np
from cut_words.utils import read_data, get_dataloader, pre_processing, compute_f1, get_weight
from model import LSTM_CRF, AdversarialModel
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors


n_classes = 5
batch_size = 128
embedding_size = 300
hidden_size = 20
epochs = 50


# def train(model, vocab_size, tag2idx, embedding_size, hidden_size, train_dataloader, test_dataloader, max_length, vectors=None):
#     model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     start_time = time.time()
#     loss_history = []
#     print("dataloader length: ", len(train_dataloader))
#     model.train()
#     f1_history = []
#     idx2tag = {value: key for key, value in tag2idx.items()}    # 得到索引对应标签的字典
#     for epoch in range(epochs):
#         total_loss = 0.
#         f1 = 0
#         for idx, (inputs, targets, cls_targests, length_list) in enumerate(train_dataloader):
#             model.zero_grad()
#             loss = (-1) * model(inputs, length_list, targets)   # 前传，计算模型损失函数
#             total_loss += loss.item()       # 加和到总损失中
#             pred = model.predict(inputs, length_list)       # 对输入进行预测，这里训练完成一个batch就直接预测，这里是为了方便打印中间结果
#             f1 += compute_f1(pred, targets, length_list)    # 计算一个batch的f1_score值，
#             loss.backward()     # 损失回传
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)     # 梯度裁剪，梯度的范数超过0.5会被裁剪成0.5
#             optimizer.step()        # 更新模型参数
#             if (idx + 1) % 20 == 0 and idx:     # 训练iterator是10的倍数时，打印一次log，当然idx为0，即第一次的时候不打印
#                 cur_loss = total_loss
#                 loss_history.append(cur_loss / (idx+1))
#                 f1_history.append(f1 / (idx+1))
#                 total_loss = 0
#                 # print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, idx*batch_size,
#                 #                                                            cur_loss / (idx * batch_size), f1 / (idx+1)))
#                 print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, idx*batch_size,
#                                                             cur_loss / (20 * batch_size), f1 / (idx+1)))
                


# 重新定义训练函数，使用 AdversarialModel
def train(adversarial_model, adversarial_optimizer, train_dataloader, test_dataloader, max_length):
    # model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # start_time = time.time()
    # loss_history = []
    # print("dataloader length: ", len(train_dataloader))
    # model.train()
    # f1_history = []
    # idx2tag = {value: key for key, value in tag2idx.items()}    # 得到索引对应标签的字典
    for epoch in range(epochs):
        total_loss = 0.
        for idx, (inputs, dep_targets, crf_targets, cls_targets, length_list) in enumerate(train_dataloader):
            # 前向传播
            ner_loss, domain_loss = adversarial_model(inputs, length_list, dep_targets, crf_targets, cls_targets)

            # 计算加和损失
            merge_loss = ner_loss + domain_loss

            # 计算两者的加和损失
            total_loss += merge_loss

            # 修改成一次性反传更新参数
            adversarial_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(adversarial_model.parameters(), 0.5)
            merge_loss.backward()
            adversarial_optimizer.step()

            # # 计算并更新NER模型的梯度
            # ner_optimizer.zero_grad()
            # ner_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(ner_model.parameters(), 0.5)     # 梯度裁剪，梯度的范数超过0.5会被裁剪成0.5
            # ner_optimizer.step()

            # # 计算并更新领域判别器的梯度
            # domain_discriminator_optimizer.zero_grad()
            # # domain_labels = torch.zeros_like(domain_output)  # 这里也要根据实际情况设置领域标签
            # domain_labels = cls_targets.unsqueeze(1).float()
            # domain_loss = nn.BCELoss()(domain_output, domain_labels)
            # domain_loss.backward()
            # torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), 0.5)     # 梯度裁剪，梯度的范数超过0.5会被裁剪成0.5
            # domain_discriminator_optimizer.step()

            if (idx + 1) % 20 == 0 and idx:     # 训练iterator是10的倍数时，打印一次log，当然idx为0，即第一次的时候不打印
                cur_loss = total_loss
                total_loss = 0
                # print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, idx*batch_size,
                #                                                            cur_loss / (idx * batch_size), f1 / (idx+1)))
                print("epochs : {}, batch : {}, loss : {}".format(epoch+1, idx*batch_size,
                                                            cur_loss / (20 * batch_size)))


if __name__ == '__main__':
    # 加载预训练数据集
    # vectors = Vectors('/Users/yaoyi/vscode_project/lstm-crf/glove.6B.100d.txt', '/Users/yaoyi/vscode_project//embedding/')
    
    # 加载中文wiki百科预训练向量

    # 获取当前路径
    curdir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

    # 添加训练列表
    train_input = "cut_words/train_small.txt"
    train_output_path = os.path.join(curdir, train_input)
    test_input = "cut_words/test_small.txt"
    test_output_path = os.path.join(curdir, test_input)

    # 读取对应的数据
    x_train, y_dep_train, y_crf_train, y_cls_train = read_data(train_output_path, 2560)
    x_test, y_dep_test, y_crf_test, y_cls_test = read_data(test_output_path, 256)
    word2idx, idx2word, tag2idx, idx2tag, vocab_size = pre_processing(x_train, y_crf_train, y_dep_train, x_test, y_crf_test, y_dep_test)

    # 构造数据集
    train_dataloader, train_max_length = get_dataloader(word2idx, tag2idx, x_train, y_crf_train, y_dep_train, y_cls_train, batch_size)   # 开始构建训练data_loader，并确定训练集的最大长度
    test_dataloader, test_max_length = get_dataloader(word2idx, tag2idx, x_test, y_crf_test, y_dep_test, y_cls_test, 32)       # 开始构建测试data_loader，并确定测试集的最大长度

    cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    embedding_path = os.path.join(cur, 'emb/sgns.wiki.bigram-char.bz2')

    # 将KeyVectors保存成二进制的方式，方便下次加载
    wvmodel_path = os.path.join(cur, 'emb/wvmodle.kv')
    if (os.path.exists(wvmodel_path)):
        wvmodel = gensim.models.KeyedVectors.load(wvmodel_path)
    else:
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False, encoding='utf-8')
        wvmodel.save(wvmodel_path)

    # 从这里获取预训练向量，没问题的，这里的pretrain_vectors是输入词表的embedding，输出标签不包含在内
    pretrain_vectors = get_weight(wvmodel, embedding_size, word2idx, idx2word, vocab_size)
    
    # 创建lstm_crf模型
    ner_model = LSTM_CRF(vocab_size, tag2idx, embedding_size, hidden_size, train_max_length, vectors=pretrain_vectors)

    # 创建领域分类模型
    domain_discriminator = nn.Sequential(
        nn.MaxPool1d(kernel_size=256),
        nn.Flatten(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )

    adversarial_model = AdversarialModel(ner_model, domain_discriminator)

    adversarial_optimizer = torch.optim.Adam(adversarial_model.parameters(), lr=0.001)

    # ner_optimizer = torch.optim.Adam(ner_model.parameters(), lr=0.001)

    # domain_discriminator_optimizer = torch.optim.Adam(domain_discriminator.parameters(), lr=0.001)

    # 开启debug情况
    torch.autograd.set_detect_anomaly(True)

    # 加载训练集中的字向量
    train(adversarial_model, adversarial_optimizer, train_dataloader, test_dataloader, max_length=train_max_length)    # 开始训练模型

