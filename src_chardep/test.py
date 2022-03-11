import numpy as np
import torch
from sklearn import preprocessing
import pyximport

char_sent = [('n', '弹'), ('NN-n', '性'), ('d', '十'), ('VA-a', '足'), ('DEC', '的'), ('n', '贡'), ('NN-n', '丸'), ('CC', '和'), ('a', '胖'), ('f', '嘟'), ('VA-f', '嘟'), ('PU', '、'), ('n', '皮'), ('n', 'Q'), ('VA-n', 'Q'), ('DEC', '的'), ('n', '肉'), ('NN-n', '圆'), ('PU', '，'), ('AD', '都'), ('VC', '是'), ('n', '新'), ('NR-n', '竹'), ('DEG', '的'), ('n', '特'), ('NN-n', '产'), ('PU', '，'), ('VV', '吃'), ('DER', '得'), ('VV', '出'), ('n', '业'), ('NN-n', '者'), ('a', '扎'), ('VA-a', '实'), ('DEC', '的'), ('n', '手'), ('NN-n', '工'), ('PU', '。')]
char_heads = [2, 4, 4, 7, 4, 7, 18, 18, 11, 11, 18, 11, 14, 15, 11, 11, 18, 21, 21, 21, 0, 23, 26, 23, 26, 21, 21, 21, 28, 28, 32, 37, 34, 37, 34, 37, 28, 21]
word_sent = ['弹性', '十足', '的', '贡丸', '和', '胖嘟嘟', '、', '皮QQ', '的', '肉圆', '，', '都', '是', '新竹', '的', '特产', '，', '吃', '得', '出', '业者', '扎实', '的', '手工', '。']
char_tpyes = [2, 4, 4, 7, 4, 7, 18, 18, 11, 11, 18, 11, 14, 15, 11, 11, 18, 21, 21, 21, 0, 23, 26, 23, 26, 21, 21, 21, 28, 28, 32, 37, 34, 37, 34, 37, 28, 21]

char_idx = 0
char2word_id = [0 for _ in char_sent] + [0]
word_heads = []
word_types = []

for word_id, word in enumerate(word_sent):
    word_len = 0
    char_word = ""
    word_head = 0
    word_type = 'root'
    for i in range(char_idx, len(char_sent)):
        char_word += char_sent[i][1]
        word_len += 1
        if char_word == word:
            break

    for i in range(char_idx, char_idx + word_len):
        if char_heads[i] < char_idx + 1 or char_heads[i] > char_idx + word_len:
            word_head = char_heads[i]
            word_type = char_tpyes[i]
        char2word_id[i + 1] = word_id + 1

    char_idx += word_len
    word_heads.append(word_head)
    word_types.append(word_type)

new_word_heads = [0 for _ in word_heads]
for i in range(len(word_sent)):
    new_word_heads[i] = char2word_id[word_heads[i]]
print(new_word_heads)

