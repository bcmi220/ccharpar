
import argparse
import itertools
import os.path
import time
import uuid

import torch
import torch.optim.lr_scheduler

import numpy as np
import math
import json
from Datareader import syndep_reader
import trees
import vocabulary
import makehp
import Zparser
import utils

tokens = Zparser

def count_wh(str, data):
    cun_w = 0
    total = 0
    for i, c_tree in enumerate(data):
        nodes = [c_tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                cun_w += node.cun_w
                # if node.label[0] =="<H>" and len([leaf.father for leaf in c_tree.leaves()]) < 20:
                #     print(node.cun_w)
                #     print(c_tree.convert().linearize())
                #     print([leaf.father for leaf in c_tree.leaves()])
                #     print(node.convert().linearize())
                #     heads = [leaf.head for leaf in node.leaves()]

                    # print([leaf.father - min(heads)+1 if leaf.head != node.head else 0 for leaf in node.leaves()])
                nodes.extend(reversed(node.children))
            else:
                total += 1

    print("total wrong head of :", str, "is", cun_w, "total head: ", total, "percent: ", 100*cun_w/total)

class Dataset(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.tag_vocab = vocabulary.Vocabulary()
        self.word_vocab = vocabulary.Vocabulary()
        self.label_vocab = vocabulary.Vocabulary()
        self.type_vocab = vocabulary.Vocabulary()
        self.char_vocab = vocabulary.Vocabulary()
        self.dataset = {}
        self.tpye = 'char'
        self.max_line = 0
        self.dev_num = 0
        self.test_num = 0

    def process(self, args):
        # srl dev set which uses 24 section of ptb is different from syn

        synconst_word_train_path = args.synconst_word_train_path
        synconst_word_dev_path = args.synconst_word_dev_path
        synconst_word_test_path = args.synconst_word_test_path

        syndep_word_train_path = args.syndep_word_train_path
        syndep_word_dev_path = args.syndep_word_dev_path
        syndep_word_test_path = args.syndep_word_test_path

        if args.dataset == 'zms':
            synconst_char_train_path = args.synconst_zms_train_path
            synconst_char_dev_path = args.synconst_zms_dev_path
            synconst_char_test_path = args.synconst_zms_test_path

            syndep_char_train_path = args.syndep_zms_train_path
            syndep_char_dev_path = args.syndep_zms_dev_path
            syndep_char_test_path = args.syndep_zms_test_path
        else:
            synconst_char_train_path = args.synconst_char_train_path
            synconst_char_dev_path = args.synconst_char_dev_path
            synconst_char_test_path = args.synconst_char_test_path

            syndep_char_train_path = args.syndep_char_train_path
            syndep_char_dev_path = args.syndep_char_dev_path
            syndep_char_test_path = args.syndep_char_test_path

        # word data ===================================================

        self.dataset['train_word_syndep_sent'], self.dataset['train_word_syndep_pos'], self.dataset['train_word_syndep_head'], self.dataset['train_word_syndep_type'] = \
            syndep_reader.read_syndep( syndep_word_train_path, self.hparams.max_len_train)

        self.dataset['dev_word_syndep_sent'], self.dataset['dev_word_syndep_pos'], self.dataset['dev_word_syndep_head'], self.dataset['dev_word_syndep_type'] = \
            syndep_reader.read_syndep( syndep_word_dev_path, self.hparams.max_len_dev)

        self.dataset['test_word_syndep_sent'], self.dataset['test_word_syndep_pos'], self.dataset['test_word_syndep_head'], self.dataset['test_word_syndep_type'] = \
            syndep_reader.read_syndep(syndep_word_test_path)

        print("Loading training trees from {}...".format(synconst_word_train_path))
        with open(synconst_word_train_path) as infile:
            treebank = infile.read()
        train_word_treebank = trees.load_trees(treebank, self.dataset['train_word_syndep_head'], self.dataset['train_word_syndep_type'], self.dataset['train_word_syndep_pos'])
        if self.hparams.max_len_train > 0:
            train_word_treebank = [tree for tree in train_word_treebank if len(list(tree.leaves())) <= self.hparams.max_len_train]
        print("Loaded {:,} training examples.".format(len(train_word_treebank)))
        self.dataset['train_word_synconst_tree'] = train_word_treebank

        print("Loading development trees from {}...".format(synconst_word_dev_path))
        with open(synconst_word_dev_path) as infile:
            treebank = infile.read()
        dev_word_treebank = trees.load_trees(treebank, self.dataset['dev_word_syndep_head'], self.dataset['dev_word_syndep_type'], self.dataset['dev_word_syndep_pos'])
        # different dev, srl is empty
        if self.hparams.max_len_dev > 0:
            dev_word_treebank = [tree for tree in dev_word_treebank if len(list(tree.leaves())) <= self.hparams.max_len_dev]
        print("Loaded {:,} development examples.".format(len(dev_word_treebank)))
        self.dataset['dev_word_synconst_tree'] = dev_word_treebank

        print("Loading test trees from {}...".format(synconst_word_test_path))
        with open(synconst_word_test_path) as infile:
            treebank = infile.read()
        test_word_treebank = trees.load_trees(treebank, self.dataset['test_word_syndep_head'], self.dataset['test_word_syndep_type'], self.dataset['test_word_syndep_pos'])
        print("Loaded {:,} test examples.".format(len(test_word_treebank)))
        self.dataset['test_word_synconst_tree'] = test_word_treebank

        print("Processing trees for training...")
        self.dataset['train_word_synconst_parse'] = [tree.convert() for tree in train_word_treebank]
        dev_word_parse = [tree.convert() for tree in dev_word_treebank]
        test_word_parse = [tree.convert() for tree in test_word_treebank]

        count_wh("train word data:", self.dataset['train_word_synconst_parse'])
        count_wh("dev word data:", dev_word_parse)
        count_wh("test word data:", test_word_parse)

        #char data ===================================================

        self.dataset['train_char_syndep_sent'], self.dataset['train_char_syndep_pos'], self.dataset[
            'train_char_syndep_head'], self.dataset['train_char_syndep_type'] = \
            syndep_reader.read_syndep(syndep_char_train_path, self.hparams.max_len_train)

        self.dataset['dev_char_syndep_sent'], self.dataset['dev_char_syndep_pos'], self.dataset['dev_char_syndep_head'], \
        self.dataset['dev_char_syndep_type'] = \
            syndep_reader.read_syndep(syndep_char_dev_path, self.hparams.max_len_dev)

        self.dataset['test_char_syndep_sent'], self.dataset['test_char_syndep_pos'], self.dataset[
            'test_char_syndep_head'], self.dataset['test_char_syndep_type'] = \
            syndep_reader.read_syndep(syndep_char_test_path)

        print("Loading training trees from {}...".format(synconst_char_train_path))
        with open(synconst_char_train_path) as infile:
            treebank = infile.read()
        train_char_treebank = trees.load_trees(treebank, self.dataset['train_char_syndep_head'],
                                          self.dataset['train_char_syndep_type'], self.dataset['train_char_syndep_pos'])
        if self.hparams.max_len_train > 0:
            train_char_treebank = [tree for tree in train_char_treebank if len(list(tree.leaves())) <= self.hparams.max_len_train]
        print("Loaded {:,} training examples.".format(len(train_char_treebank)))
        self.dataset['train_char_synconst_tree'] = train_char_treebank

        print("Loading development trees from {}...".format(synconst_char_dev_path))
        with open(synconst_char_dev_path) as infile:
            treebank = infile.read()
        dev_char_treebank = trees.load_trees(treebank, self.dataset['dev_char_syndep_head'], self.dataset['dev_char_syndep_type'],
                                        self.dataset['dev_char_syndep_pos'])
        # different dev, srl is empty
        if self.hparams.max_len_dev > 0:
            dev_char_treebank = [tree for tree in dev_char_treebank if len(list(tree.leaves())) <= self.hparams.max_len_dev]
        print("Loaded {:,} development examples.".format(len(dev_char_treebank)))
        self.dataset['dev_char_synconst_tree'] = dev_char_treebank

        print("Loading test trees from {}...".format(synconst_char_test_path))
        with open(synconst_char_test_path) as infile:
            treebank = infile.read()
        test_char_treebank = trees.load_trees(treebank, self.dataset['test_char_syndep_head'], self.dataset['test_char_syndep_type'],
                                         self.dataset['test_char_syndep_pos'])
        print("Loaded {:,} test examples.".format(len(test_char_treebank)))
        self.dataset['test_char_synconst_tree'] = test_char_treebank

        print("Processing trees for training...")
        self.dataset['train_char_synconst_parse'] = [tree.convert() for tree in train_char_treebank]
        dev_char_parse = [tree.convert() for tree in dev_char_treebank]
        test_char_parse = [tree.convert() for tree in test_char_treebank]

        count_wh("train char data:", self.dataset['train_char_synconst_parse'])
        count_wh("dev char data:", dev_char_parse)
        count_wh("test char data:", test_char_parse)

        self.tpye = args.dataset
        if args.dataset != 'word':
            self.dataset['train_synconst_parse'] = self.dataset['train_char_synconst_parse']
            self.dataset['train_synconst_tree'] = self.dataset['train_char_synconst_tree']
            self.dataset['train_syndep_sent'] = self.dataset['train_char_syndep_sent']
            self.dataset['dev_treebank'] = dev_char_treebank
            self.dataset['dev_syndep_pos'] = self.dataset['dev_char_syndep_pos']
            self.dataset['test_treebank'] = test_char_treebank
            self.dataset['test_syndep_pos'] = self.dataset['test_char_syndep_pos']
        else:
            self.dataset['train_synconst_parse'] = self.dataset['train_word_synconst_parse']
            self.dataset['train_synconst_tree'] = self.dataset['train_word_synconst_tree']
            self.dataset['train_syndep_sent'] = self.dataset['train_word_syndep_sent']
            self.dataset['dev_treebank'] = dev_word_treebank
            self.dataset['dev_syndep_pos'] = self.dataset['dev_word_syndep_pos']
            self.dataset['test_treebank'] = test_word_treebank
            self.dataset['test_syndep_pos'] = self.dataset['test_word_syndep_pos']

        print("Constructing vocabularies...")

        self.tag_vocab.index(Zparser.START)
        self.tag_vocab.index(Zparser.STOP)
        self.tag_vocab.index(Zparser.TAG_UNK)

        self.word_vocab.index(Zparser.START)
        self.word_vocab.index(Zparser.STOP)
        self.word_vocab.index(Zparser.UNK)

        self.label_vocab.index(())
        sublabels = [Zparser.Sub_Head]
        self.label_vocab.index(tuple(sublabels))

        self.type_vocab = vocabulary.Vocabulary()

        char_set = set()

        for i, tree in enumerate(self.dataset['train_synconst_parse']):

            const_sentences = [leaf.word for leaf in tree.leaves()]
            if len(const_sentences) != len(self.dataset['train_syndep_sent'][i]):
                print(const_sentences)
                print(self.dataset['train_syndep_sent'][i])
            assert len(const_sentences) == len(self.dataset['train_syndep_sent'][i])
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalParseNode):
                    self.label_vocab.index(node.label)
                    nodes.extend(reversed(node.children))
                else:
                    self.tag_vocab.index(node.tag)
                    self.tag_vocab.index(node.goldtag)
                    self.word_vocab.index(node.word)
                    self.type_vocab.index(node.type)
                    char_set |= set(node.word)

        # char_vocab.index(tokens.CHAR_PAD)

        # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
        highest_codepoint = max(ord(char) for char in char_set)
        if highest_codepoint < 512:
            if highest_codepoint < 256:
                highest_codepoint = 256
            else:
                highest_codepoint = 512

            # This also takes care of constants like tokens.CHAR_PAD
            for codepoint in range(highest_codepoint):
                char_index = self.char_vocab.index(chr(codepoint))
                assert char_index == codepoint
        else:
            self.char_vocab.index(tokens.CHAR_UNK)
            self.char_vocab.index(tokens.CHAR_START_SENTENCE)
            self.char_vocab.index(tokens.CHAR_START_WORD)
            self.char_vocab.index(tokens.CHAR_STOP_WORD)
            self.char_vocab.index(tokens.CHAR_STOP_SENTENCE)
            for char in sorted(char_set):
                self.char_vocab.index(char)

        self.tag_vocab.freeze()
        self.word_vocab.freeze()
        self.label_vocab.freeze()
        self.char_vocab.freeze()
        self.type_vocab.freeze()

        def print_vocabulary(name, vocab):
            special = {tokens.START, tokens.STOP, tokens.UNK}
            print("{} ({:,}): {}".format(
                name, vocab.size,
                sorted(value for value in vocab.values if value in special) +
                sorted(value for value in vocab.values if value not in special)))

        if args.print_vocabs:
            print_vocabulary("Tag", self.tag_vocab)
            print_vocabulary("Word", self.word_vocab)
            print_vocabulary("Label", self.label_vocab)
            print_vocabulary("Char", self.char_vocab)
            print_vocabulary("Type", self.type_vocab)

        # need=["建筑业"]
        # for i, tree in enumerate(self.dataset['train_synconst_parse']):
        #
        #     const_sentences = [leaf.word for leaf in tree.leaves()]
        #     sent = "".join(const_sentences)
        #     for ns in need:
        #         if ns in sent:
        #             print(tree.convert().linearize())
        # for i, tree in enumerate(dev_char_parse):
        #
        #     const_sentences = [leaf.word for leaf in tree.leaves()]
        #     sent = "".join(const_sentences)
        #     for ns in need:
        #         if ns in sent:
        #             print(tree.convert().linearize())
        #
        # for i, tree in enumerate(test_char_parse):
        #
        #     const_sentences = [leaf.word for leaf in tree.leaves()]
        #     sent = "".join(const_sentences)
        #     for ns in need:
        #         if ns in sent:
        #             print(tree.convert().linearize())

        self.max_line = len(self.dataset['train_synconst_parse'])

        print("max_length:", max([len([leaf.word for leaf in tree.leaves()]) for tree in self.dataset['train_synconst_parse']]))
