# import codecs
# import sys
# import os

# def character_tagging(input_file, output_file):
#     input_data = codecs.open(input_file, 'r', 'utf-8')
#     output_data = codecs.open(output_file, 'w', 'utf-8')
#     for line in input_data.readlines():
#         # 单行数据处理
#         word_list = line.strip().split()
#         # 如果文本为空，直接跳过
#         if len(word_list) == 0:
#             continue
#         char_list = []
#         label_list = []
#         for word in word_list:
#             if len(word) == 1:
#                 # output_data.write(word + "\tS\n")
#                 char_list.append(word)
#                 label_list.append('S')
#             else:
#                 char_list.append(word[0])
#                 label_list.append('B')

#                 for w in word[1:len(word)-1]:
#                     char_list.append(w)
#                     label_list.append('M')
                
#                 char_list.append(word[len(word)-1])
#                 label_list.append('E')

#                 # output_data.write(word[0] + "\tB\n")
#                 # for w in word[1:len(word)-1]:
#                 #     output_data.write(w + "\tM\n")
#                 # output_data.write(word[len(word)-1] + "\tE\n")
#         output_data.write(' '.join(char_list) + '\t' + ' '.join(label_list))
#         output_data.write("\n")
#     input_data.close()
#     output_data.close()

# if __name__ == '__main__':
#     # if len(sys.argv) != 3:
#     #     print("pls use: python make_crf_train_data.py input output")
#     #     sys.exit()
#     # input_file = sys.argv[1]
#     # output_file = sys.argv[2]
#     curdir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
#     print(curdir)
#     train_input_file = "cityu_training.utf8"
#     train_output_file = "train.txt"
#     train_input_path = os.path.join(curdir, train_input_file)
#     train_output_path = os.path.join(curdir, train_output_file)

#     test_input_file = "cityu_test_gold.utf8"
#     test_output_file = "test.txt"
#     test_input_path = os.path.join(curdir, test_input_file)
#     test_output_path = os.path.join(curdir, test_output_file)

#     character_tagging(train_input_path, train_output_path)
#     character_tagging(test_input_path, test_output_path)




import codecs
import sys
import os
from ddparser import DDParser

ddp = DDParser()

def character_tagging(input_file_list, output_file):
    # 如果输出文件存在，先删除该文件，再重新创建并写入
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"the output file exist, remove it first...")
    for idx, input_file in enumerate(input_file_list):
        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'a', 'utf-8')
        for line in input_data.readlines():
            # 单行数据处理
            word_list = line.strip().split()
            # 解析句法列表
            deperl_list = []
            # 这里可以增加一列作为句法依存分析
            sentence = ''.join(word_list)
            # 使用ddparser进行分析
            result = ddp.parse(sentence)
            # 解析出对应的字符串
            parse_word_list = result[0]['word']
            parse_prel_list = result[0]['deprel']

            for words, prel in zip(parse_word_list, parse_prel_list):
                for word in words:
                    deperl_list.append(prel)

            # 如果文本为空，直接跳过
            if len(word_list) == 0:
                continue
            char_list = []
            label_list = []
            for word in word_list:
                if len(word) == 1:
                    # output_data.write(word + "\tS\n")
                    char_list.append(word)
                    label_list.append('S')
                else:
                    char_list.append(word[0])
                    label_list.append('B')

                    for w in word[1:len(word)-1]:
                        char_list.append(w)
                        label_list.append('M')
                    
                    char_list.append(word[len(word)-1])
                    label_list.append('E')

                    # output_data.write(word[0] + "\tB\n")
                    # for w in word[1:len(word)-1]:
                    #     output_data.write(w + "\tM\n")
                    # output_data.write(word[len(word)-1] + "\tE\n")
            # output_data.write(' '.join(char_list) + '\t' + ' '.join(label_list))
            # output_data.write("\n")

            # 确保生成的几个列表长度一样
            assert len(char_list) == len(label_list) == len(deperl_list)

            # 将单行的数据按照单行多列的方式展开
            for char, label, prel in zip(char_list, label_list, deperl_list):
                text = ' '.join([char, label, prel]) + '\n'
                output_data.write(text)

            # 在每个句子的末尾加上结束符，并写入该句子的类型信息，方便后续的计算损失函数
            output_data.write(f'-DOCEND- -X- {idx}' + 2 * '\n')

    input_data.close()
    output_data.close()

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("pls use: python make_crf_train_data.py input output")
    #     sys.exit()
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]


    curdir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    print(curdir)

    # 添加训练列表
    # train_input_list = ["cityu_training_small.utf8", "pku_training_small.utf8"]
    # train_input_list = ["cityu_training_small.utf8", "as_training_small.utf8"]
    train_input_list = ["pku_training_small.utf8", "msr_training_small.utf8"]
    train_output_file = "train_small.txt"
    train_input_path = [os.path.join(curdir, train_input_file) for train_input_file in train_input_list]
    train_output_path = os.path.join(curdir, train_output_file)

    # test_input_list = ["cityu_test_gold_small.utf8", "pku_test_gold_small.utf8"]
    # test_input_list = ["cityu_test_gold_small.utf8", "as_test_gold_small.utf8"]
    test_input_list = ["pku_test_gold_small.utf8", "msr_test_gold_small.utf8"]
    test_output_file = "test_small.txt"
    test_input_path = [os.path.join(curdir, test_input_file) for test_input_file in test_input_list]
    test_output_path = os.path.join(curdir, test_output_file)

    character_tagging(train_input_path, train_output_path)
    character_tagging(test_input_path, test_output_path)