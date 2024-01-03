# from tqdm import tqdm

# def write_to_csv(filename,csv_name):
#     with open(filename, "r", encoding="utf-8") as f:
#         # data = json.load(f)
#         data = f.readlines()

#     list_data=[]
#     for idx, item in tqdm(enumerate(data)):
#         tokens, labels = item.strip().split('\t')
#         if(len(tokens)==0):
#             # print(tokens)
#             continue
#         list_data.append([' '.join(tokens),' '.join(labels)])

#     df=pd.DataFrame(list_data,columns=['text','labels'])
#     df.to_csv(csv_name,sep='\t',index=False)

# if __name__ == "__main__":
#     filename="/data/yvanyao/yvanyao/yvanyao/Text-similarity-based-on-GRU-autoencoder/cut_programe/output.txt"
#     csv_name='/data/yvanyao/yvanyao/yvanyao/Text-similarity-based-on-GRU-autoencoder/cut_programe/output.tsv'
#     write_to_csv(filename,csv_name)
#     filename="./data/test.json"
#     csv_name='data/test.tsv'
#     write_to_csv(filename,csv_name)


# import os
# import torch
# from torch.utils.data import Dataset, DataLoader

# import torch
# from torch.utils.data import Dataset, DataLoader

# class CRFDataset(Dataset):
#     def __init__(self, file_path):
#         self.data = self.load_data(file_path)

#     def load_data(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
        
#         data = []
#         for line in lines:
#             text, labels = line.strip().split('\t')
#             text_tokens = text.split(' ')
#             label_tokens = labels.split(' ')
#             data.append((text_tokens, label_tokens))
        
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text_tokens, label_tokens = self.data[idx]
#         return {
#             'text_tokens': text_tokens,
#             'label_tokens': label_tokens
#         }

# # Padding function to ensure equal size within a batch
# def collate_fn(batch):
#     text_tokens_batch = [sample['text_tokens'] for sample in batch]
#     label_tokens_batch = [sample['label_tokens'] for sample in batch]

#     # Find the maximum lengths in the batch
#     max_text_len = max(len(tokens) for tokens in text_tokens_batch)
#     max_label_len = max(len(tokens) for tokens in label_tokens_batch)

#     # Pad sequences to the maximum lengths
#     padded_text_tokens_batch = [tokens + ['<PAD>'] * (max_text_len - len(tokens)) for tokens in text_tokens_batch]
#     padded_label_tokens_batch = [tokens + ['<PAD>'] * (max_label_len - len(tokens)) for tokens in label_tokens_batch]

#     return {
#         'text_tokens': padded_text_tokens_batch,
#         'label_tokens': padded_label_tokens_batch
#     }

# # Example usage:
# file_path = 'your_dataset.txt'
# crf_dataset = CRFDataset(file_path)

# # Create a DataLoader with collate_fn
# batch_size = 64
# dataloader = DataLoader(crf_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# # Iterate through batches
# for batch in dataloader:
#     text_tokens_batch = batch['text_tokens']
#     label_tokens_batch = batch['label_tokens']

#     # Now, text_tokens_batch and label_tokens_batch have consistent sizes within a batch
#     # Convert tokens to numerical representations if needed
#     # Feed the data to your CRF model for training or inference
#     # ...


# def data2label(words):
#     chars = []
#     tags = []
#     for w in words:
#         chars.extend(list(w))
#         if len(w) == 1:
#             tags.append('S')
#         else:
#             tags.extend(['B'] + ['I'] * (len(w) - 2) + ['E'])


# 创建于2024年1月1日，搭建模型
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset


def read_data(path, length):
    sentences_list = []         # 每一个元素是一整个句子
    sentences_crf_list_labels = []  # 句子crf标签列表
    sentences_dep_list_lables = [] # 句子的句法依存关系列表
    sentences_class_list_labels = [] # 句子文档标签（该句子属于什么文档类型）
    with open(path, 'r', encoding='UTF-8') as f:
        sentence_crf_labels = []    # 每个元素是这个句子的每个单词的标签
        sentence_dep_lables = []    # 每个元素是这个句子的每个单词的句法依存标签
        sentence = []           # 每个元素是这个句子的每个单词

        for line in f:
            line = line.strip()
            if not line:        # 如果遇到了空白行
                if sentence:    # 防止空白行连续多个，导致出现空白的句子
                    sentences_list.append(' '.join(sentence))
                    sentences_crf_list_labels.append(' '.join(sentence_crf_labels))
                    sentences_dep_list_lables.append(' '.join(sentence_dep_lables))
                    sentences_class_list_labels.append(sentence_class_label)
                    # 创建新的句子的list，准备读入下一个句子
                    sentence = []
                    sentence_dep_lables = []
                    sentence_crf_labels = []
                    sentence_class_label = None
            else:
                res = line.split()
                assert len(res) == 3
                # 句子末尾，添加句子类别
                if res[0] == '-DOCEND-':
                    sentence_class_label = int(res[2])
                else:
                    sentence.append(res[0])
                    sentence_dep_lables.append(res[1])
                    sentence_crf_labels.append(res[2])

        if sentence:            # 防止最后一行没有空白行，导致最后一句话录入不到
            sentences_list.append(sentence)
            sentences_crf_list_labels.append(sentence_crf_labels)
            sentences_dep_list_lables.append(sentence_dep_lables)
            sentences_class_list_labels.append(sentence_class_label)
    return sentences_list[:length], sentences_crf_list_labels[:length], sentences_dep_list_lables[:length], sentences_class_list_labels[:length]

def build_vocab(sentences_list):
    ret = []
    for sentences in sentences_list:
        ret += [word for word in sentences.split()]
    return list(set(ret))

class mydataset(Dataset):
    def __init__(self, x : torch.Tensor, y_dep: torch.Tensor, y_crf : torch.Tensor, y_cls: torch.Tensor, length_list):
        self.x = x
        self.y_dep = y_dep
        self.y_crf = y_crf
        self.y_cls = y_cls
        self.length_list = length_list
    def __getitem__(self, index):
        data = self.x[index]
        dep_labels = self.y_dep[index]
        crf_labels = self.y_crf[index]
        length = self.length_list[index]
        cls_labels = self.y_cls[index]
        return data, dep_labels, crf_labels, cls_labels, length
    def __len__(self):
        return len(self.x)

def get_idx(word, d):
    if d[word] is not None:
        return d[word]
    else:
        return d['<unknown>']

def sentence2vector(sentence, d):
    return [get_idx(word, d) for word in sentence.split()]

def padding(x, max_length, d):
    length = 0
    for i in range(max_length - len(x)):    # 也很简单，就是在所有的不够长的文本后面加上padding
        x.append(d['<pad>'])
    return x

def get_dataloader(word2idx, tag2idx, x, y_crf, y_dep, y_cls, batch_size, device):
    # word2idx, tag2idx, vocab_size = pre_processing()
    inputs = [sentence2vector(s, word2idx) for s in x]  # 每个句子通过字典从char映射成数字
    dep_inputs = [sentence2vector(s, word2idx) for s in y_dep] # 每个句子依存关系从char映射成数字
    crf_targets = [sentence2vector(s, tag2idx) for s in y_crf]  # 每个标签都通过字典从char映射成数字
    cls_targets = y_cls

    length_list = [256 if len(sentence) > 256 else len(sentence) for sentence in inputs]    # 获取所有句子，每一个行的长度列表，太长截断

    max_length = 0
    # max_length = max(max(length_list), max_length)  # 获得句子中长度最长的那一个的长度
    max_length = 256    # 手动设定了一个124/256
    print(f"the max_length is {max_length}")

    inputs = [sample[:max_length] for sample in inputs]
    dep_inputs = [sample[:max_length] for sample in dep_inputs]
    crf_targets = [sample[:max_length] for sample in crf_targets]

    # 进行文本填充
    inputs = torch.tensor([padding(sentence, max_length, word2idx) for sentence in inputs])     # 将所有句子padding到同一个长度
    dep_inputs = torch.tensor([padding(sentence, max_length, tag2idx) for sentence in dep_inputs], dtype=torch.long)      # 将所有标签padding到同一个长度
    crf_targets = torch.tensor([padding(sentence, max_length, tag2idx) for sentence in crf_targets], dtype=torch.long)      # 将所有标签padding到同一个长度

    dataset = mydataset(inputs, dep_inputs, crf_targets, cls_targets, length_list)   # 创建一个dataset的类，类中需要实现__init__、__get_item__、__len__三个函数， 其中get_item回返回x，y以及句子长度
    
    # 如果device是gpu，则打开pin_memory
    pin_memory = True if device == 'gpu' else False
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=pin_memory)  # 标准的torch的dataloader类的用法

    return dataloader, max_length   # 最后返回了dataloader和ma_length

def pre_processing(x_train, y_train_crf, y_train_dep, x_test, y_test_crf, y_test_dep):
    # x_train, y_train = read_data("data/conll2003/train.txt", 200)     # 先将文本读取出来，每行没拆分成单字，空格分割开
    # x_test, y_test = read_data("data/conll2003/test.txt", 100)     # 每一行字符的标签，也是通过空格分割开
    d_x = build_vocab(x_train + x_test + y_train_dep + y_test_dep)       # 通过遍历所有行，得到所有的词列表，这里将句法关系也放入词汇表中共享向量
    d_y = build_vocab(y_train_crf+y_test_crf)       # 通过便利所有行，得到所有的标签列表
    word2idx = {d_x[i]: i for i in range(len(d_x))}     # 创建word2idx，一个字典，key和value分别是词和索引
    tag2idx = {d_y[i]: i for i in range(len(d_y))}      # 创建tag2idx，一个字典，key和value分别是标签和索引
    # tag2idx["<START>"] = 9      # 在标签字典中添加开始和结束标记符
    # tag2idx["<STOP>"] = 10
    pad_idx = len(word2idx)
    word2idx['<pad>'] = pad_idx     # 在词字典中添加pad标记符
    tag2idx['<pad>'] = len(tag2idx) # 在标签字典中添加pad标记符
    vocab_size = len(word2idx)      # 获取词表的长度
    idx2tag = {value: key for key, value in tag2idx.items()}    # 获取索引到标签的映射
    idx2word = {value: key for key, value in word2idx.items()}
    print(tag2idx)
    return word2idx, idx2word, tag2idx, idx2tag, vocab_size    # 这里最后返回的是，词到索引映射、标签到索引映射、词表的长度

def compute_f1(pred, targets, length_list):
    tp, fn, fp = [], [], []
    for i in range(15):     # 这里的tp需要长度为15，是因为我们一共有15种标签，每个类别可能有一个tp,fn和fp
        tp.append(0)
        fn.append(0)
        fp.append(0)
    for i, length in enumerate(length_list):
        for j in range(length):
            a, b = pred[i][j], targets[i][j]
            if (a == b):
                tp[a] += 1
            else:
                fp[a] += 1
                fn[b] += 1
    tps = 0
    fps = 0
    fns = 0
    for i in range(9):      # 我们无需统计后6种标签的预测准确度，只需保证前9个的准确即可
        tps += tp[i]
        fps += fp[i]
        fns += fn[i]
    p = tps / (tps + fps)   # 得到precision值
    r = tps / (tps + fns)   # 得到recall值
    return 2 * p * r / (p + r)      # 得到f1score

# weight = torch.zeros(vocab_size, embed_size)
def get_vector_fix(word, model):
    try:
        return model.get_vector(word)
    except KeyError:
        return nn.init.normal_(torch.empty(model.embedding_dim), mean=0, std=0.1)
        

def get_weight(wvmodel, embed_size, word2index, index2word, vocab_size):
    # 我觉得这里可以进行uniform初始化
    # weight = nn.Parameter(torch.empty(vocab_size, embed_size))
    # nn.init.uniform_(weight, 0, 0.1)

    # weight = nn.Embedding(vocab_size, embed_size)   # 没有传入向量，那么直接初始化

    weight = torch.empty(vocab_size, embed_size)
    init.uniform_(weight, 0, 0.1)

    # 开始给部分向量赋值
    for i in range(len(wvmodel.index_to_key)):
        try:
            index = word2index[wvmodel.index_to_key[i]]
        except:
            continue
        word = index2word[word2index[wvmodel.index_to_key[i]]]

        weight[index, :] = torch.from_numpy(get_vector_fix(word, wvmodel))

    return weight


# input_lang, output_lang, pairs = readfile(r'./测试文本/orig.txt')
# input_lang, output_lang, pairs = prepareData(input_lang, output_lang, pairs)
# # token_weight = get_weight(token_embedding, 300, input_lang)
# # dep_weight = get_weight(dep_embedding, 10, input_lang)
# weight = get_weight(embedding_path, 300, input_lang)