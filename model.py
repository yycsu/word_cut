# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 15 12:38:37 2021

# @author: bring
# """

# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(input_size, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=2, bidirectional=True, batch_first=True)

#     def forward(self, input, h0, c0):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         # output, hidden = self.gru(output, hidden)
#         output, (h0, c0) = self.lstm(output, (h0, c0))
#         return output, h0, c0

#     # def initHidden(self):
#     #     return torch.zeros(1, 1, self.hidden_size, device=device)

#     def initHidden(self):
#         return torch.zeros(4, 1, self.hidden_size//2, device=device)

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=2, bidirectional=True, batch_first=True)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, h0, c0):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         # output, hidden = self.gru(output, hidden)
#         output, (h0, c0) = self.lstm(output, (h0, c0))
#         output = self.softmax(self.out(output[0]))
#         return output, h0, c0

#     # def initHidden(self):
#     #     return torch.zeros(1, 1, self.hidden_size, device=device)

#     def initHidden(self):
#         return torch.zeros(4, 1, self.hidden_size//2, device=device)


# # 以下是pytorch版本
# import torch
# import torch.nn as nn
# from torchcrf import CRF  # 需要安装torchcrf库
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class MyCustomModel(nn.Module):
#     def __init__(self, num_tags, char_vector_dim, dependency_vector_dim):
#         super(MyCustomModel, self).__init__()
#         # 定义层
#         self.char_embedding = ...  # 使用预训练权重进行初始化
#         self.dependency_embedding = ...  # 使用预训练权重进行初始化

#         self.encoder = nn.LSTM(char_vector_dim, 256, batch_first=True, bidirectional=True)
#         self.shared_encoder = nn.LSTM(char_vector_dim + dependency_vector_dim, 256, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(256 * 2, 256, batch_first=True, bidirectional=True)

#         self.reconstructed_output_layer = nn.Linear(256 * 2, char_vector_dim)
#         self.max_pooling = nn.AdaptiveMaxPool1d(1)
#         self.domain_output_layer = nn.Linear(256 * 2, 2)

#         self.crf = CRF(num_tags)  # 假设CRF实现已经提供

#     def forward(self, input_chars, input_dependencies, mask):
#         # 输入应该是两个元素的列表: [input_chars, input_dependencies]
#         char_embedding_output = self.char_embedding(input_chars)
#         dependency_embedding_output = self.dependency_embedding(input_dependencies)

#         merged_embeddings = torch.cat([char_embedding_output, dependency_embedding_output], dim=-1)

#         encoder_output, _ = self.encoder(char_embedding_output)
#         shared_encoder_output, _ = self.shared_encoder(merged_embeddings)

#         decoder_output, _ = self.decoder(encoder_output)
#         decoder_with_shared_output = torch.cat([decoder_output, shared_encoder_output], dim=-1)

#         reconstructed_output = self.reconstructed_output_layer(decoder_with_shared_output)

#         # 对shared_encoder_output进行max pooling
#         pooled_output, _ = torch.max(shared_encoder_output, dim=1)

#         domain_output = self.domain_output_layer(pooled_output)

#         crf_output = self.crf.decode(decoder_with_shared_output, mask=mask)

#         return crf_output, domain_output, reconstructed_output

#     def loss_function(self, crf_output, domain_output, reconstructed_output, tags, domain_labels, mask):
#         crf_loss = -self.crf(crf_output, tags, mask=mask)
#         domain_loss = nn.CrossEntropyLoss()(domain_output, domain_labels)
#         reconstruction_loss = nn.MSELoss()(reconstructed_output, input_chars)
#         return crf_loss, domain_loss, reconstruction_loss

# # 使用
# num_tags = ...  # 定义CRF层的标签数量
# char_vector_dim = ...  # 定义字符向量维度
# dependency_vector_dim = ...  # 定义依赖向量维度

# # 初始化模型
# my_model = MyCustomModel(num_tags, char_vector_dim, dependency_vector_dim)



# # 以下是pytorch版本
# import torch
# import torch.nn as nn
# from torchcrf import CRF  # 需要安装torchcrf库
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class MyCustomModel(nn.Module):
#     def __init__(self, num_tags, char_vector_dim, dependency_vector_dim):
#         super(MyCustomModel, self).__init__()
#         # 定义层
#         self.char_embedding = ...  # 使用预训练权重进行初始化
#         self.dependency_embedding = ...  # 使用预训练权重进行初始化

#         self.encoder = nn.LSTM(char_vector_dim, 256, batch_first=True, bidirectional=True)
#         self.shared_encoder = nn.LSTM(char_vector_dim + dependency_vector_dim, 256, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(256 * 2, 256, batch_first=True, bidirectional=True)

#         self.reconstructed_output_layer = nn.Linear(256 * 2, char_vector_dim)
#         self.max_pooling = nn.AdaptiveMaxPool1d(1)
#         self.domain_output_layer = nn.Linear(256 * 2, 2)

#         self.crf = CRF(num_tags)  # 假设CRF实现已经提供

#     def forward(self, input_chars, input_dependencies, mask):
#         # 输入应该是两个元素的列表: [input_chars, input_dependencies]
#         char_embedding_output = self.char_embedding(input_chars)
#         dependency_embedding_output = self.dependency_embedding(input_dependencies)

#         merged_embeddings = torch.cat([char_embedding_output, dependency_embedding_output], dim=-1)

#         encoder_output, _ = self.encoder(char_embedding_output)
#         shared_encoder_output, _ = self.shared_encoder(merged_embeddings)

#         decoder_output, _ = self.decoder(encoder_output)
#         decoder_with_shared_output = torch.cat([decoder_output, shared_encoder_output], dim=-1)

#         reconstructed_output = self.reconstructed_output_layer(decoder_with_shared_output)

#         # 对shared_encoder_output进行max pooling
#         pooled_output, _ = torch.max(shared_encoder_output, dim=1)

#         domain_output = self.domain_output_layer(pooled_output)

#         crf_output = self.crf.decode(decoder_with_shared_output, mask=mask)

#         return crf_output, domain_output, reconstructed_output

#     def loss_function(self, crf_output, domain_output, reconstructed_output, tags, domain_labels, mask):
#         crf_loss = -self.crf(crf_output, tags, mask=mask)
#         domain_loss = nn.CrossEntropyLoss()(domain_output, domain_labels)
#         reconstruction_loss = nn.MSELoss()(reconstructed_output, input_chars)
#         return crf_loss, domain_loss, reconstruction_loss

# # 使用
# num_tags = ...  # 定义CRF层的标签数量
# char_vector_dim = ...  # 定义字符向量维度
# dependency_vector_dim = ...  # 定义依赖向量维度

# # 初始化模型
# my_model = MyCustomModel(num_tags, char_vector_dim, dependency_vector_dim



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchcrf import CRF

class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_index, embedding_size, hidden_size, max_length, vectors=None):
        super(LSTM_CRF, self).__init__()    # 继承nn.Module搭建的模型
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.target_size = len(tag_to_index)
        if vectors is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)   # 没有传入向量，那么直接初始化
        else:
            self.embedding = nn.Embedding.from_pretrained(vectors)  # 否则，使用传入的向量进行初始化
        self.lstm = nn.LSTM(embedding_size * 2, hidden_size // 2, bidirectional=True)   # 由于双向，所以中间的hidden_size需要处以2
        self.hidden_to_tag = nn.Linear(hidden_size, self.target_size)   # 用隐层的输出，作为输入，输出的长度是各类标签的长度
        self.crf = CRF(self.target_size, batch_first=True)  # nn.torchcrf层输入一个，参数1是命名实体识别的标签长度，另外一个是是否使用batch_first
        self.max_length = max_length    # 还有一个max_length

    def get_mask(self, length_list):        # 得到每一个padding后句子的mask，其中有实际标签的地方是1，没有的地方是0
        mask = []
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_length - length)])
        return torch.tensor(mask, dtype=torch.bool)

    def LSTM_Layer(self, sentences, dep_targets, length_list):

        sentence_embeds = self.embedding(sentences)      # 输入的是转换完成的一个batch，250个句子，每个句子背转换成了固定长度124，每个元素是一个index，而self.embedding会将每个字符转换成一个向量， 转换完成后的shape变成了(128, 256, 300)
        dep_embeds = self.embedding(dep_targets)    # 输入的是句法依存关系的向量batch，大小也是(128, 256, 300)

        embeds = torch.cat([sentence_embeds, dep_embeds], dim=2)

        packed_embeds = pack_padded_sequence(embeds, lengths=length_list, batch_first=True, enforce_sorted=False)    # pack_padded_sequence可以将一个填充过的序列(padded sequence) 转换为一个紧密打包的序列(packed seqeunce)， 以提高模型的计算效率
        
        embeds_lstm_out, _ = self.lstm(packed_embeds)       # 将embedding通过lstm层，得到对应的输出
        
        result, _ = pad_packed_sequence(embeds_lstm_out, batch_first=True, total_length=self.max_length)

        feature = self.hidden_to_tag(result)        # 将结果通过一个线性层，最后返回结果shape是250 * 124 * 12

        return feature

    def CRF_layer(self, input, targets, length_list):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        return self.crf(input, targets, self.get_mask(length_list))     # 输入是发射矩阵向量input，标签targets，

    def forward(self, sentences, length_list, targets):
        x = self.LSTM_Layer(sentences, length_list)     # 输入x是每个句子转换成数字之后的向量，250 * 124(batch_size * padding_length), 输出是250 * 124 * 12
        x = self.CRF_layer(x, targets, length_list)     # 输入x是250 * 124 * 12, 输入的targets是250 * 124，是最后的label

        return x        # 计算出来是最后的一个值，例如 tensor([-10671.6328, grad_fn=<SumBackward0>])

    def predict(self, sentences, length_list):
        out = self.LSTM_Layer(sentences, length_list)       # 输入sentences，然后通过self.LSTM_Layer层
        mask = self.get_mask(length_list)       # 计算mask

        return self.crf.decode(out, mask)       # 最后得到预测结果


# 定义领域判别器
class DomainDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# 定义对抗训练网络
class AdversarialModel(nn.Module):
    def __init__(self, ner_model, domain_discriminator):
        super(AdversarialModel, self).__init__()
        self.ner_model = ner_model
        self.domain_discriminator = domain_discriminator

    def forward(self, sentences, length_list, dep_targets, crf_targets):
        ner_output = self.ner_model.LSTM_Layer(sentences, dep_targets, length_list)
        ner_loss = (-1) * self.ner_model.CRF_layer(ner_output, crf_targets, length_list)

        domain_output = self.domain_discriminator(ner_output.permute(0, 2, 1))
        
        # # 复制一份 ner_output，避免 inplace 操作
        # ner_output_copy = ner_output.clone().detach()
        
        # # 添加领域判别器
        # domain_output = self.domain_discriminator(ner_output_copy.permute(0, 2, 1))

        return ner_loss, domain_output