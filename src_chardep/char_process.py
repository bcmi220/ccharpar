import re
import numpy as np
import codecs
import os
import sys

def print_sentence_to_conll(sentences, dep_wcpos, wordposs, charposs, heads, tpyes, unif_types, fout):
    """Print a labeled sentence into CoNLL format.
    """

    for i , (word, wcpos, wordpos, charpos, head, type, unif_type) in enumerate(zip(sentences, dep_wcpos, wordposs, charposs, heads, tpyes, unif_types)):
        fout.write(str(i+1)+'\t'+word+'\t'+wcpos+'\t'+wordpos+'\t'+charpos+'\t'+"_"+'\t'+str(head)+'\t'+type+'\t'+unif_type+'\t'+"_")  # .rjust(15)

        fout.write("\n")
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
        self.__source_file = open(file_path, 'r')#,encoding='GBK')#GBK')
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

char_reader = CharReader("../chardata/zms_char_dep.txt")
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
    #print(word)
    inst = char_reader.getNext()

print(len(char_dict))

datalist = ['train', 'dev', 'test']
for name in datalist:

    dep_reader = CoNLLXReader("../data/" + name + "_ctbc.conll")

    counter = 0
    inst = dep_reader.getNext()
    no_char_tree = {}

    fout = open("../chardata/" + name + "_zmsdep.txt", "w")

    while inst is not None:

        inst_size = inst.length()

        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        new_words = []
        new_wordpos = []
        new_charpos = []
        new_wcpos = []
        new_head = []
        new_type = []
        unif_type = []
        old2newidx = {}
        is_subroot = []
        sent = inst.words
        # print(inst.words)
        # print(inst.postags)
        # print(inst.heads)
        # print(inst.types)
        for old_idx, (word, pos, head, type) in enumerate(zip(inst.words, inst.postags, inst.heads, inst.types)):
            if len(word) > 1 :#and (pos != 'PU' or word.isalnum()):
                if word in char_dict:
                    char_inst = char_dict[word]
                    # print(word)
                    # print(char_inst.words)
                    # print(char_inst.postags)
                    # print(char_inst.heads)
                    # print(char_inst.types)
                    begin_id = len(new_words)
                    for char_word, char_pos, char_head, char_type in zip(char_inst.words, char_inst.postags, char_inst.heads, char_inst.types):
                        new_words.append(char_word)
                        if char_head != 0 :
                            new_head.append(char_head + begin_id)
                            new_type.append(char_type)
                            unif_type.append("inword")
                            new_wordpos.append(pos)
                            new_charpos.append(char_pos)
                            new_wcpos.append(char_pos)
                            is_subroot.append(-1)
                        else:
                            new_head.append(head)
                            new_type.append(char_type + "-" + type)
                            unif_type.append(type)
                            is_subroot.append(1)
                            new_wordpos.append(pos)
                            new_charpos.append(char_pos)
                            new_wcpos.append(pos + "-" + char_pos)
                            #head id start from 1, thus add 1 to old idx
                            old2newidx[old_idx + 1] = len(new_words)
                else:
                    #no char tree
                    for token in word[:-1]:
                        new_words.append(token)
                        new_wordpos.append(pos)
                        new_charpos.append('n')
                        new_wcpos.append('n')
                        new_head.append(len(new_words) + 1)
                        new_type.append('nn')
                        unif_type.append("inword")
                        is_subroot.append(-1)

                    new_words.append(word[-1])
                    new_wordpos.append(pos)
                    new_charpos.append('n')
                    new_wcpos.append(pos + "-" + 'n')
                    new_head.append(head)
                    new_type.append('root-n' + "-" + type)
                    unif_type.append(type)
                    is_subroot.append(1)
                    old2newidx[old_idx + 1] = len(new_words)
            else:
                new_words.append(word)
                new_wordpos.append(pos)
                new_charpos.append(pos)
                new_wcpos.append(pos)
                new_head.append(head)
                new_type.append(type)
                unif_type.append(type)
                is_subroot.append(1)
                old2newidx[old_idx + 1] = len(new_words)
        # print(new_words)
        # print(new_wordpos)
        # print(new_head)
        # print(new_type)
        cov_heads = []
        for idx, head in enumerate(new_head):

            if is_subroot[idx] == 1 and head!= 0:
                cov_heads.append(old2newidx[head])
            else:
                cov_heads.append(head)
        # print(cov_heads)
        if counter < 0:
            break

        # dep_wordpos.append(new_wordpos)
        # dep_charpos.append(new_charpos)
        # dep_wcpos.append(new_wcpos)
        # # dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
        # dep_sentences.append(new_words)
        # dep_heads.append(cov_heads)
        # dep_types.append(new_type)

        print_sentence_to_conll(new_words, new_wcpos, new_wordpos, new_charpos, cov_heads, new_type, unif_type, fout)
        inst = dep_reader.getNext()
        if inst is None:
            print(sent)

    dep_reader.close()
    print("Total number of data: %d" % counter)




