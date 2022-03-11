import re
import numpy as np
import codecs
import os
import sys
import ctrees
def print_sentence_to_const(treelist, fout):
    """Print a labeled sentence into CoNLL format.
    """

    for i , tree in enumerate(treelist):
        fout.write("{}\n".format(tree.linearize()))
    fout.write("\n")

idx = 0
pl = 0
sst = ""
def dfs(head, str, words,pos):
    global idx
    global pl
    global sst
    child = []
    # print(idx)
    # print(str[idx])
    idx += 1
    label = str[idx]
    idx += 1
    if str[idx] != '(':
        words.append(str[idx])
        pos.append(label)
        pl += 1
        idx += 1
        return pl - 1

    while idx < len(str) and str[idx] !=')':
        child.append(dfs(head, str, words, pos))
        idx += 1


    assert len(child) == 2
    if label[-1] == 'x' or label[-1] == 'z':
        if(child[1]) >= len(head):
            print(sst)
        head[child[1]] = child[0]
        return child[0]
    else:
        head[child[0]] = child[1]
        return child[1]

with open("../chardata/zms_char.txt") as infile:
    treebank = infile.readlines()

write_dep = open("../chardata/zms_char_dep.txt", "w")
#need_list = ["竞拍","发愁","停车场","超暴力","有利可图","从长远来看","反其道而行之"]
head = [0 for _ in range(1000)]
cun = 0
for tree in treebank:
    sst = tree
    idx = 0
    pl = 1
    words = []
    pos = []
    tokens = tree.replace("(", " ( ").replace(")", " ) ").split()
    # print(tokens)
    root = dfs(head, tokens, words, pos)
    head[root] = 0
    # print(head[1:pl], words, pos)
    # print(pl)
    ww = ""
    for w in words:
        ww += w
    write_dep.write(str(cun) + "\t" + ww + "\t" + sst)
    cun += 1
    for j in range(pl - 1):
        write_dep.write(str(j) + "\t"+ words[j] + "\t" + str(pos[j]) + "\t" + str(head[j+1]) + "\t" + "inword" + "\n")
    write_dep.write("\n")






