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

class DependencyInstance(object):
    def __init__(self, words, postags, heads, types, lines = None):
        self.words = words
        self.postags = postags
        self.heads = heads
        self.types = types
        self.lines =lines

    def length(self):
        return len(self.words)


class CoNLLXReader(object):
    def __init__(self, file_path, type_vocab = None):
        self.__source_file = open(file_path, 'r')
        self.type_vocab = type_vocab

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            #line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        postags = []
        types = []
        heads = []

        # words.append(Zparser.ROOT)
        # postags.append(Zparser.ROOT)
        # types.append(Zparser.ROOT)
        # heads.append(0)

        for tokens in lines:

            word = tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)

            postags.append(pos)

            types.append(type)

            heads.append(head)

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        return DependencyInstance(words, postags, heads, types)


class CharReader(object):
    def __init__(self, file_path, type_vocab = None):
        self.__source_file = open(file_path, 'r', encoding='GBK')
        self.type_vocab = type_vocab

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            # line = line.decode('GBK')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        postags = []
        types = []
        heads = []

        # words.append(Zparser.ROOT)
        # postags.append(Zparser.ROOT)
        # types.append(Zparser.ROOT)
        # heads.append(0)
        for tokens in lines[1:]:
            word = tokens[1]
            pos = tokens[2]
            if tokens[3] == 'o':
                tokens[3] = 0
                print(lines[0][0], tokens)
            head = int(tokens[3])
            type = tokens[4]

            words.append(word)

            postags.append(pos)

            types.append(type)

            heads.append(head)

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        return DependencyInstance(words, postags, heads, types, lines)

char_reader = CharReader("../chardata/chardep.txt")
inst = char_reader.getNext()

char_dict = {}
counter = 0
while inst is not None:

    inst_size = inst.length()
    counter += 1
    if counter % 10000 == 0:
        print("reading data: %d" % counter)
    word = "".join(inst.words)
    char_dict[word] = inst
    inst = char_reader.getNext()

print(len(char_dict))
char_const_dict = {}
datalist = ['train', 'dev', 'test']
# for name in datalist:
#     #fout = open("../chardata/" + name + "_charconst.txt", "w")
#     fout = open("../chardata/" + name + "_sdctconst.txt", "w")
#     treebank = ctrees.load_trees("../data/" + name + "_ctbc.txt", char_dict, char_const_dict)
#
#     print_sentence_to_const(treebank, fout)

flag = []

def rmove_c(cur, childs, mod_list):
    child = childs[cur]
    flag[cur] = 1

    for ch in child:
        if flag[ch]==1:
            mod_list.append((cur, ch))
        else:
            rmove_c(ch, childs, mod_list)

    return


def dfs(cur, fa, label, childs, pos_list, words):
    child = childs[cur]
    flag[cur] = 1
    if len(child)== 0:
        return ctrees.LeafTreebankNode(pos_list[cur - 1], words[cur-1])
    tree_child = []
    f = 0
    for ch in child:
        if flag[ch] == 1:
            continue
        if ch > cur and f == 0:
            f = 1
            tree_child.append(ctrees.LeafTreebankNode(pos_list[cur - 1], words[cur-1]))
        tree_child.append(dfs(ch, cur, label, childs, pos_list, words))
    if f == 0:
        tree_child.append(ctrees.LeafTreebankNode(pos_list[cur - 1], words[cur - 1]))

    if fa == 0:
        return ctrees.InternalTreebankNode(label, tree_child)
    return ctrees.InternalTreebankNode('iw', tree_child)
#
fout = open("../chardata/SDCT_charconst.txt", "w")
need_list = ["竞拍","发愁","停车场","超暴力","有利可图","从长远来看","反其道而行之"]
# need_list = ["鲟鱼","天安门","普天同庆","50余","拉美裔","空对地","好人好事","旅游业"]
for word in char_dict:
    dep_children = [[] for _ in char_dict[word].words]
    dep_children.append([])  # start from 1
    root = 0
    pos_list = []
    if word == "从长远来看":
        char_dict[word].heads[4] = 0
        char_dict[word].heads[3] = 5
    if word == "做东":
        char_dict[word].heads[0] = 2
    for idx, (head, charpos) in enumerate(zip(char_dict[word].heads, char_dict[word].postags)):
        if head == 0 and root == 0:
            root_pos = char_dict[word].postags[idx]
            pos_list.append(root_pos)
            root = idx + 1
        else:
            pos_list.append(charpos)
            if head == 0 or head > len(char_dict[word].words):
                if root == 0:
                    dep_children[len(char_dict[word].words)].append(idx + 1)
                else:
                    dep_children[root].append(idx + 1)
            else:
                dep_children[head].append(idx + 1)

    flag = [0 for _ in dep_children]
    mod_list = []
    rmove_c(root, dep_children, mod_list)
    for i in range(1, len(char_dict[word].words) + 1):
        if flag[i] == 0:
            rmove_c(i, dep_children, mod_list)
    for (fa, son) in mod_list:
        dep_children[root].append(son)

    dep_children[root].sort()
    flag = [0 for _ in dep_children]
    word_node = dfs(root, 0, "R", dep_children, pos_list, word)
    for i in range(1, len(char_dict[word].words) + 1):
        if flag[i] == 0:
            print(root)
            print(char_dict[word].heads)
            print(char_dict[word].types)
            print(dep_children)
        assert flag[i] == 1

    if word in need_list:
        print(word)
        print(char_dict[word].postags)
        print(char_dict[word].heads)
        print(char_dict[word].types)
        print(word_node.linearize())
    fout.write(word_node.linearize())
    fout.write("\n")
fout.write("\n")



