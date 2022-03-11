from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pretrained_bert.tokenization import BertTokenizer
from pretrained_bert.modeling import BertForPreTraining
from pretrained_bert.optimization import BertAdam


from torch.utils.data import Dataset
import random
from Evaluator import evaluate
from Evaluator import dep_eval
from Evaluator import pos_eval
from Evaluator import seg_dep_eval
import copy
import trees
import makehp
import json
from stanfordcorenlp import StanfordCoreNLP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall, self.precision, self.fscore)

SegDandC = 'seg(d&c)'
SegDorC = 'seg(d|c)'
SegD = 'seg(d)'
SegC = 'seg(c)'
SegT = 'seg(tag)'
seg_types = [SegT, SegC, SegDandC]

class EvalManyTask(object):
    def __init__(self, hparams, dataset, eval_batch_size, evalb_dir, model_path_base, log_path):


        self.hparams = hparams
        self.dataset = dataset.dataset
        self.dataset_type = dataset.tpye
        self.evalb_dir = evalb_dir
        self.model_path_base = model_path_base
        self.test_model_path = None
        self.log_path = log_path
        self.summary_dict = {}
        self.eval_batch_size = eval_batch_size
        if self.dataset_type == 'word':
            self.stanford_nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27', lang = 'zh')

        self.best_dev_score = -np.inf
        self.best_model_path = None
        self.summary_dict['dev_char_synconst'] = 0
        self.summary_dict['dev_char_syndep_uas'] = 0
        self.summary_dict['dev_char_syndep_las'] = 0
        self.summary_dict['dev_char_pos'] = 0
        self.summary_dict['dev_word_synconst'] = 0
        self.summary_dict['dev_word_syndep_uas'] = 0
        self.summary_dict['dev_word_syndep_las'] = 0
        self.summary_dict['dev_word_pos'] = 0
        self.summary_dict['dev_predinput_synconst'] = 0
        self.summary_dict['dev_predinput_syndep_uas'] = 0
        self.summary_dict['dev_predinput_syndep_las'] = 0
        self.summary_dict['dev_predinput_pos'] = 0

        self.summary_dict['test_char_synconst'] = 0
        self.summary_dict['test_char_syndep_uas'] = 0
        self.summary_dict['test_char_syndep_las'] = 0
        self.summary_dict['test_char_pos'] = 0
        self.summary_dict['test_word_synconst'] = 0
        self.summary_dict['test_word_syndep_uas'] = 0
        self.summary_dict['test_word_syndep_las'] = 0
        self.summary_dict['test_word_pos'] = 0
        self.summary_dict['test_predinput_synconst'] = 0
        self.summary_dict['test_predinput_syndep_uas'] = 0
        self.summary_dict['test_predinput_syndep_las'] = 0
        self.summary_dict['test_predinput_pos'] = 0

        for seg_type in seg_types:
            for data_type in ['dev', 'test']:
                for eval_type in ['word', 'head', 'type', 'pos']:
                    self.summary_dict[data_type + '_' + seg_type + '_' + eval_type] = 0

    def eval_multitask(self, model, start_time, epoch_num):

        logger.info("***** Running dev *****")
        logger.info("  Batch size = %d", self.eval_batch_size)

        dev_start_time = time.time()


        print("Start Dev Eval:")

        print("===============================================")
        print("Start syntax dev eval:")
        self.syn_dev(model)


        print(
            "dev-elapsed {} "
            "total-elapsed {}".format(
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )
        print(
            '============================================================================================================================')

        print("Start Test Eval:")
        test_start_time = time.time()

        print("===============================================")
        print("Start syntax test eval:")
        self.syn_test(model)

        print(
            "test-elapsed {} "
            "total-elapsed {}".format(
                format_elapsed(test_start_time),
                format_elapsed(start_time),
            )
        )

        if self.dataset_type != 'word':
            self.summary_dict['total dev score'] = self.summary_dict['dev_char_synconst'].fscore + self.summary_dict['dev_char_syndep_las']

        else:
            self.summary_dict['total dev score'] = self.summary_dict['dev_word_synconst'].fscore + self.summary_dict['dev_word_syndep_las']

        is_save_model = False

        if self.summary_dict['total dev score'] > self.best_dev_score:
            if self.best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = self.best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            self.best_dev_score = self.summary_dict['total dev score']

            if self.dataset_type != 'word':
                self.best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}_devpos={:.2f}".format(
                    self.model_path_base, self.summary_dict['dev_char_synconst'].fscore, self.summary_dict['dev_char_syndep_uas'], self.summary_dict['dev_char_syndep_las'],
                    self.summary_dict['dev_char_pos'])
            else:
                self.best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}_devpos={:.2f}".format(
                    self.model_path_base, self.summary_dict['dev_word_synconst'].fscore, self.summary_dict['dev_word_syndep_uas'],
                    self.summary_dict['dev_word_syndep_las'],
                    self.summary_dict['dev_word_pos'])
            is_save_model = True

        char_log_data = "{} epoch, char: dev-fscore {:},test-fscore {:}, dev-uas {:.2f}, dev-las {:.2f}, " \
                   "test-uas {:.2f}, test-las {:.2f}, " \
                   "dev-pos {:}, test-pos {:}, " \
                   "dev_score {:.2f}, best_dev_score {:.2f}" \
            .format(epoch_num, self.summary_dict['dev_char_synconst'], self.summary_dict['test_char_synconst'],
                    self.summary_dict['dev_char_syndep_uas'], self.summary_dict['dev_char_syndep_las'],
                    self.summary_dict['test_char_syndep_uas'], self.summary_dict['test_char_syndep_las'],
                    self.summary_dict['dev_char_pos'], self.summary_dict['test_char_pos'],
                    self.summary_dict['total dev score'], self.best_dev_score)

        word_log_data = "gold seg word(pred for const): (pred)dev-fscore {:},(pred)test-fscore {:}, dev-uas {:.2f}, dev-las {:.2f}, " \
                        "test-uas {:.2f}, test-las {:.2f}, " \
            .format(self.summary_dict['dev_word_synconst'], self.summary_dict['test_word_synconst'],
                    self.summary_dict['dev_word_syndep_uas'], self.summary_dict['dev_word_syndep_las'],
                    self.summary_dict['test_word_syndep_uas'], self.summary_dict['test_word_syndep_las'])

        log_data = char_log_data + '\n' + word_log_data + '\n'
        if self.dataset_type =='word':
            predinput_log_data = "predinput segandpos: dev-fscore {:}, test-fscore {:}, dev-uas {:.2f}, dev-las {:.2f}, " \
                                 "test-uas {:.2f}, test-las {:.2f}, " \
                .format(self.summary_dict['dev_predinput_synconst'], self.summary_dict['test_predinput_synconst'],
                        self.summary_dict['dev_predinput_syndep_uas'], self.summary_dict['dev_predinput_syndep_las'],
                        self.summary_dict['test_predinput_syndep_uas'], self.summary_dict['test_predinput_syndep_las'])

            log_data = log_data + predinput_log_data + '\n' + '\n'

        else:
            for seg_type in seg_types:
                seg_word_log_data = "{} word : dev-word {:.2f}, dev-head {:.2f}, dev-type {:.2f}，dev-pos {:.2f}," \
                            "test-word {:.2f}, test-head {:.2f}, test-type {:.2f} test-pos {:.2f}" \
                .format(seg_type, self.summary_dict['dev_'+seg_type+'_word'], self.summary_dict['dev_'+seg_type+'_head'],
                        self.summary_dict['dev_'+seg_type+'_type'], self.summary_dict['dev_'+seg_type+'_pos'],
                        self.summary_dict['test_'+seg_type+'_word'], self.summary_dict['test_'+seg_type+'_head'],
                        self.summary_dict['test_'+seg_type+'_type'], self.summary_dict['test_'+seg_type+'_pos'])
                log_data = log_data + seg_word_log_data + '\n' + '\n'


        if self.log_path is not None:
            if not os.path.exists(self.log_path):
                flog = open(self.log_path, 'w')
            flog = open(self.log_path, 'r+')
            content = flog.read()
            flog.seek(0, 0)
            flog.write(log_data + content)

        return self.best_model_path, is_save_model

    def givesent_char2word(self, char_heads, char_tpyes, char_sent, word_sent):

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
                    word_type = ""
                    for tc in char_tpyes[i][::-1]:
                        if tc == '-':
                            break
                        word_type += tc
                char2word_id[i + 1] = word_id + 1
            char_idx += word_len
            word_heads.append(word_head)
            word_types.append(word_type[::-1])

        new_word_heads = [0 for _ in word_heads]
        for i in range(len(word_sent)):
            new_word_heads[i] = char2word_id[word_heads[i]]

        return new_word_heads, word_types

    def tree_make_sent(self, char_tree, use_seg_type):
        word_sent = []
        word_pos = []
        nodes = [char_tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalTreebankNode):
                is_seg_root = False
                if use_seg_type == SegC and ('-#' in node.label or '#t' in node.label):
                    is_seg_root = True
                if use_seg_type == SegDandC and (('-#' in node.label and 'root-' in node.type) or ('#t' in node.label and '-' in node.type)):
                    is_seg_root = True
                if is_seg_root:
                    word = ""
                    for leaf in node.leaves():
                        word += leaf.word
                    word_sent.append(word)
                    wordtag = ""
                    for ct in node.label:
                        if ct != '-' and ct != '#':
                            wordtag += ct
                        else:
                            break
                    word_pos.append(wordtag)
                else:
                    nodes.extend(reversed(node.children))
            else:
                word_sent.append(node.word)
                wordtag = ""
                for ct in node.goldtag:
                    if ct != '-' and ct != '#':
                        wordtag += ct
                    else:
                        break
                word_pos.append(wordtag)

        return word_sent, word_pos

    def char2word(self, char_heads, char_tpyes, char_sent, char_tree, use_seg_type):

        word_sent = []
        word_pos = []
        if use_seg_type == SegT:
            words_idx = [-1 for _ in char_sent]
            pos_seg = [tag for tag, char in char_sent]
            for char_id, ((tag, char), type) in enumerate(zip(char_sent, char_tpyes)):
                f_zms = 0
                if '-' in tag:  # head of word
                    if '#b' in tag:
                        f_zms = 1
                    if '#i' in tag:
                        f_zms = 2
                    wordtag = ""
                    for ct in tag:
                        if ct != '-':
                            wordtag += ct
                        else:
                            break
                    head_list = [char_id + 1]
                    words_idx[char_id] = char_id
                    pos_seg[char_id] = wordtag
                    for i in range(char_id - 1, -1, -1):
                        if char_heads[i] in head_list and (char_sent[i][0].islower() or (f_zms == 2 and '#b' not in char_sent[i][0])):
                            words_idx[i] = char_id
                            head_list.append(i + 1)
                            pos_seg[i] = wordtag
                        else:
                            break
                    for i in range(char_id + 1, len(char_sent)):
                        if char_heads[i] in head_list and (char_sent[i][0].islower() or (f_zms == 1 and '#i' in char_sent[i][0])):
                            words_idx[i] = char_id
                            head_list.append(i + 1)
                            pos_seg[i] = wordtag
                        else:
                            break
            word = ""
            pre_pos = ""
            for i in range(len(words_idx)):
                if i == 0 or words_idx[i] == -1 or words_idx[i] != words_idx[i - 1]:

                    if i > 0:
                        word_sent.append(word)
                        word_pos.append(pre_pos)
                    word = char_sent[i][1]
                    pre_pos = pos_seg[i]
                else:
                    word += char_sent[i][1]
                    pre_pos = pos_seg[i]
            word_sent.append(word)
            word_pos.append(pre_pos)
        else:
            word_sent, word_pos = self.tree_make_sent(char_tree, use_seg_type)

        word_heads, word_types =self.givesent_char2word(char_heads, char_tpyes, char_sent, word_sent)

        return word_heads, word_types, word_sent, word_pos

    # def const_char2word(self):
    #
    #     def process_NONE(tree):
    #
    #         if isinstance(tree, LeafTreebankNode):
    #             label = tree.tag
    #             if label == '-NONE-':
    #                 return None
    #             else:
    #                 return tree
    #
    #         tr = []
    #         label = tree.label
    #         if label == '-NONE-':
    #             return None
    #         for node in tree.children:
    #             new_node = process_NONE(node)
    #             if new_node is not None:
    #                 tr.append(new_node)
    #         if tr == []:
    #             return None
    #         else:
    #             return InternalTreebankNode(label, tr)
    #
    #     new_trees = []
    #     for i, tree in enumerate(trees):
    #         new_tree = process_NONE(tree)
    #         new_trees.append(new_tree)

    def const_wordtree(self, tree):

        temp_tree = copy.deepcopy(tree)
        nodes = [temp_tree]
        while nodes:
            node = nodes.pop()
            new_children = []
            if isinstance(node, trees.InternalTreebankNode):
                for child in node.children:
                    flag = 1
                    if isinstance(child, trees.InternalTreebankNode):
                        if '-#' in child.label or '#t' in child.label:
                            flag = 0
                            child_leaves = [leaf for leaf in child.leaves()]
                            new_children += child_leaves

                    if flag == 1:
                        new_children.append(child)

                node.children = new_children
                nodes.extend(reversed(node.children))


        return temp_tree

    def predinput_word2char(self,pred_char, gold_char_sent):
        idx = 0
        temp_tree = copy.deepcopy(pred_char)

        def dfs(node):
            nonlocal  idx
            nonlocal gold_char_sent
            new_children = []
            if isinstance(node, trees.InternalTreebankNode):
                for child in node.children:
                    if isinstance(child, trees.LeafTreebankNode):
                        word = child.word
                        char = ""
                        for char_i in word:
                            char += char_i
                            if char == gold_char_sent[idx][1]:
                                new_children.append(
                                    trees.LeafTreebankNode(gold_char_sent[idx][0], gold_char_sent[idx][1],
                                                           child.head, child.father, child.type,
                                                           gold_char_sent[idx][0]))
                                char = ""
                                idx += 1
                    else:
                        new_children.append(dfs(child))

                node.children = new_children
                return node

        dfs(temp_tree)
        return temp_tree

    def syn_dev(self, model):
        syntree_pred = []
        syntree_predinput = []
        pos_pred = []
        dev_treebank = self.dataset['dev_treebank']

        for dev_start_index in range(0, len(dev_treebank), self.eval_batch_size):
            subbatch_trees = dev_treebank[dev_start_index:dev_start_index + self.eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

            syntree, _ = model.parse_batch(subbatch_sentences)
            syntree_pred.extend(syntree)
            pos_pred.extend([[leaf.goldtag for leaf in tree.leaves()] for tree in syntree])
            if self.dataset_type == 'word':
                subbatch_pred_sentences = []
                for tree in subbatch_trees:
                    pred_sent = ""
                    for leaf in tree.leaves():
                        pred_sent += leaf.word
                    pred_sent = self.stanford_nlp.pos_tag(pred_sent)#[[word,tag],....]
                    subbatch_pred_sentences.append([(token[1],token[0]) for token in pred_sent])

                #print("gold:", subbatch_sentences[0])
                #print("pred:", subbatch_pred_sentences[0])
                #print("=======================================================")
                syntree, _ = model.parse_batch(subbatch_pred_sentences)
                syntree_predinput.extend(syntree)

        # const parsing:

        word_dev_pos = [[leaf.tag for leaf in tree.leaves()] for tree in self.dataset['dev_word_synconst_tree']]

        if self.dataset_type != 'word':

            print("char constituent eval:")
            # for i in range(5):
            #     print("gold:", self.dataset['dev_char_synconst_tree'][i].linearize())
            #     print("pred:", syntree_pred[i].linearize())
            self.summary_dict['dev_char_synconst'] = evaluate.evalb(self.evalb_dir, self.dataset['dev_char_synconst_tree'], syntree_pred)
            # const_word_tree_pred = []
            # const_word_tree_gold = []
            # for pred_char, gold_char in zip(syntree_pred,self.dataset['dev_char_synconst_tree']):
            #     const_word_tree_pred.append(self.const_wordtree(pred_char))
            #     const_word_tree_gold.append(self.const_wordtree(gold_char))
            #
            # print("pred word constituent eval:")
            # self.summary_dict['dev_word_synconst'] = evaluate.evalb(self.evalb_dir, const_word_tree_gold, const_word_tree_pred)

            dev_pred_heads = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            dev_pred_types = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            syndep_dev_pos = [[leaf.tag for leaf in tree.leaves()] for tree in self.dataset['dev_char_synconst_tree']]
            syndep_dev_sents = [[(leaf.goldtag, leaf.word) for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(dev_pred_heads) == len(dev_pred_types)
            assert len(dev_pred_types) == len(self.dataset['dev_char_syndep_type'])
            print("Char dep :")
            self.summary_dict['dev_char_syndep_uas'], self.summary_dict['dev_char_syndep_las'] = \
                dep_eval.eval(len(dev_pred_heads), self.dataset['dev_char_syndep_sent'], syndep_dev_pos,
                              dev_pred_heads, dev_pred_types, self.dataset['dev_char_syndep_head'],
                              self.dataset['dev_char_syndep_type'],
                              punct_set=self.hparams.punctuation, symbolic_root=False)
            word_sents = [[leaf.word for leaf in tree.leaves()] for tree in self.dataset['dev_word_synconst_tree']]
            word_pred_heads = []
            word_pred_types = []

            pred_seg_words = {}
            pred_seg_postags = {}
            pred_seg_heads = {}
            pred_seg_types = {}
            for seg_t in seg_types:
                pred_seg_words[seg_t] = []
                pred_seg_postags[seg_t] = []
                pred_seg_heads[seg_t] = []
                pred_seg_types[seg_t] = []

            for dev_pred_head, dev_pred_type, syndep_dev_sent, word_sent, char_tree in zip(dev_pred_heads, dev_pred_types, syndep_dev_sents, word_sents, syntree_pred):
                word_pred_head, word_pred_type= self.givesent_char2word(dev_pred_head, dev_pred_type, syndep_dev_sent, word_sent)
                word_pred_heads.append(word_pred_head)
                word_pred_types.append(word_pred_type)
                #if self.dataset_type != 'zms':
                for seg_t in seg_types:
                    seg_head, seg_type, seg_word, seg_pos = self.char2word(dev_pred_head, dev_pred_type, syndep_dev_sent, char_tree, seg_t)
                    pred_seg_heads[seg_t].append(seg_head)
                    pred_seg_types[seg_t].append(seg_type)
                    pred_seg_words[seg_t].append(seg_word)
                    pred_seg_postags[seg_t].append(seg_pos)
            #if self.dataset_type != 'zms':
            for seg_t in seg_types:
                print("begin dev " + seg_t+ " eval: ")
                self.summary_dict['dev_' + seg_t + '_word'], self.summary_dict['dev_' + seg_t + '_head'], self.summary_dict['dev_' + seg_t + '_type'] = \
                    seg_dep_eval.eval(len(pred_seg_heads[seg_t]), self.dataset['dev_word_syndep_sent'], word_dev_pos,
                                      self.dataset['dev_word_syndep_head'], self.dataset['dev_word_syndep_type'],
                                      pred_seg_words[seg_t], pred_seg_postags[seg_t], pred_seg_heads[seg_t], pred_seg_types[seg_t],
                                      punct_set=self.hparams.punctuation, symbolic_root=False)

            print("Start Pos dev eval:")
            self.summary_dict['dev_char_pos'] = pos_eval.eval(self.dataset['dev_char_syndep_sent'], self.dataset['dev_char_syndep_sent'], self.dataset['dev_char_syndep_pos'], pos_pred)
            #if self.dataset_type != 'zms':
            for seg_t in seg_types:
                self.summary_dict['dev_' + seg_t + '_pos'] = pos_eval.eval(self.dataset['dev_word_syndep_sent'],pred_seg_words[seg_t],
                                                                  self.dataset['dev_word_syndep_pos'], pred_seg_postags[seg_t])



        else:
            self.summary_dict['dev_word_synconst'] = evaluate.evalb(self.evalb_dir,
                                                                     self.dataset['dev_word_synconst_tree'],
                                                                     syntree_pred)
            word_pred_heads = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            word_pred_types = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(word_pred_heads) == len(word_pred_types)
            assert len(word_pred_types) == len(self.dataset['dev_word_syndep_type'])
            print("Start Pos dev eval:")
            self.summary_dict['dev_word_pos'] = pos_eval.eval(self.dataset['dev_word_syndep_sent'], self.dataset['dev_word_syndep_sent'], self.dataset['dev_word_syndep_pos'], pos_pred)

            const_word_tree_pred = []
            const_word_tree_gold = []
            for pred_char, gold_char in zip(syntree_predinput, self.dataset['dev_char_synconst_tree']):
                gold_char_tree = self.const_wordtree(gold_char)
                const_word_tree_gold.append(gold_char_tree)
                gold_char_sent = [(leaf.tag,leaf.word) for leaf in gold_char_tree.leaves()]
                pred_char_tree = self.predinput_word2char(pred_char, gold_char_sent)
                gold_leaves = list(gold_char_tree.leaves())
                predicted_leaves = list(pred_char_tree.leaves())
                #print(gold_char.linearize())
                #print(pred_char.linearize())
                # print([leaf.word for leaf in pred_char_tree.leaves()])
                assert all(
                    gold_leaf.word == predicted_leaf.word
                    for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))
                const_word_tree_pred.append(self.const_wordtree(pred_char_tree))

            print("predinput word constituent eval:")
            self.summary_dict['dev_predinput_synconst'] = evaluate.evalb(self.evalb_dir, const_word_tree_gold,
                                                                    const_word_tree_pred)

            dev_predinput_words = [[leaf.word for leaf in tree.leaves()] for tree in syntree_predinput]
            dev_predinput_pos = [[leaf.tag for leaf in tree.leaves()] for tree in syntree_predinput]
            dev_predinput_heads = [[leaf.father for leaf in tree.leaves()] for tree in syntree_predinput]
            dev_predinput_types = [[leaf.type for leaf in tree.leaves()] for tree in syntree_predinput]
            print("predinput word dep eval:")
            self.summary_dict['dev_predinput_word'], self.summary_dict['dev_predinput_head'], self.summary_dict[
                'dev_predinput_type'] = \
                seg_dep_eval.eval(len(dev_predinput_heads), self.dataset['dev_word_syndep_sent'], word_dev_pos,
                                  self.dataset['dev_word_syndep_head'], self.dataset['dev_word_syndep_type'],
                                  dev_predinput_words, dev_predinput_pos, dev_predinput_heads,
                                  dev_predinput_types,
                                  punct_set=self.hparams.punctuation, symbolic_root=False)


        print("Word with gold segment dep :")
        self.summary_dict['dev_word_syndep_uas'], self.summary_dict['dev_word_syndep_las'] = \
            dep_eval.eval(len(word_pred_heads), self.dataset['dev_word_syndep_sent'], word_dev_pos,
                          word_pred_heads, word_pred_types, self.dataset['dev_word_syndep_head'],
                          self.dataset['dev_word_syndep_type'],
                          punct_set=self.hparams.punctuation, symbolic_root=False)

        print("===============================================")
        # print("Start Pos dev eval:")
        #
        # self.summary_dict['dev_word_pos'] = pos_eval.eval(self.dataset['dev_word_syndep_pos'], pos_pred)


    def syn_test(self, model):
        syntree_pred = []
        syntree_predinput = []
        pos_pred = []
        test_treebank = self.dataset['test_treebank']
        for dev_start_index in range(0, len(test_treebank), self.eval_batch_size):
            subbatch_trees = test_treebank[dev_start_index:dev_start_index + self.eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

            syntree, _ = model.parse_batch(subbatch_sentences)

            syntree_pred.extend(syntree)
            pos_pred.extend([[leaf.goldtag for leaf in tree.leaves()] for tree in syntree])
            if self.dataset_type == 'word':
                subbatch_pred_sentences = []
                for tree in subbatch_trees:
                    pred_sent = ""
                    for leaf in tree.leaves():
                        pred_sent += leaf.word
                    pred_sent = self.stanford_nlp.pos_tag(pred_sent)#[[word,tag],....]
                    subbatch_pred_sentences.append([(token[1],token[0]) for token in pred_sent])

                syntree, _ = model.parse_batch(subbatch_pred_sentences)
                syntree_predinput.extend(syntree)


        word_test_pos = [[leaf.tag for leaf in tree.leaves()] for tree in self.dataset['test_word_synconst_tree']]

        if self.dataset_type != 'word':
            print("char constituent eval:")
            # maxlen = 0
            # for gold, pred  in zip(self.dataset['test_char_synconst_tree'], syntree_pred):
            #     maxlen = max(len(gold.linearize()),max(maxlen,len(pred.linearize())))
            # print("maxlen: ", maxlen)
            self.summary_dict['test_char_synconst'] = evaluate.evalb(self.evalb_dir,
                                                                    self.dataset['test_char_synconst_tree'],
                                                                    syntree_pred)
            # const_word_tree_pred = []
            # const_word_tree_gold = []
            # for pred_char, gold_char in zip(syntree_pred, self.dataset['test_char_synconst_tree']):
            #     const_word_tree_pred.append(self.const_wordtree(pred_char))
            #     const_word_tree_gold.append(self.const_wordtree(gold_char))
            # print("pred word constituent eval:")
            # self.summary_dict['test_word_synconst'] = evaluate.evalb(self.evalb_dir, const_word_tree_gold,
            #                                                         const_word_tree_pred)

            char_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            char_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            char_pos = [[leaf.tag for leaf in tree.leaves()] for tree in syntree_pred]
            char_sents = [[(leaf.goldtag, leaf.word) for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(char_pred_head) == len(char_pred_type)
            assert len(char_pred_type) == len(self.dataset['test_char_syndep_type'])
            print("Char dep :")
            self.summary_dict['test_char_syndep_uas'], self.summary_dict['test_char_syndep_las'] = \
                dep_eval.eval(len(char_pred_head), self.dataset['test_char_syndep_sent'], char_pos,
                              char_pred_head, char_pred_type, self.dataset['test_char_syndep_head'],
                              self.dataset['test_char_syndep_type'],
                              punct_set=self.hparams.punctuation, symbolic_root=False)
            word_sents = [[leaf.word for leaf in tree.leaves()] for tree in self.dataset['test_word_synconst_tree']]
            word_pred_heads = []
            word_pred_types = []
            pred_seg_words = {}
            pred_seg_postags = {}
            pred_seg_heads = {}
            pred_seg_types = {}
            for seg_t in seg_types:
                pred_seg_words[seg_t] = []
                pred_seg_postags[seg_t] = []
                pred_seg_heads[seg_t] = []
                pred_seg_types[seg_t] = []
            for pred_head, pred_type, char_sent, word_sent, char_tree in zip(char_pred_head, char_pred_type, char_sents, word_sents, syntree_pred):
                word_pred_head, word_pred_type = self.givesent_char2word(pred_head, pred_type, char_sent, word_sent)
                word_pred_heads.append(word_pred_head)
                word_pred_types.append(word_pred_type)
                #if self.dataset_type != 'zms':
                for seg_t in seg_types:
                    seg_head, seg_type, seg_word, seg_pos = self.char2word(pred_head, pred_type, char_sent, char_tree, seg_t)
                    pred_seg_heads[seg_t].append(seg_head)
                    pred_seg_types[seg_t].append(seg_type)
                    pred_seg_words[seg_t].append(seg_word)
                    pred_seg_postags[seg_t].append(seg_pos)
            #if self.dataset_type != 'zms':
            for seg_t in seg_types:
                print("begin test " + seg_t + " eval: ")
                self.summary_dict['test_'+seg_t+'_word'], self.summary_dict['test_'+seg_t+'_head'], self.summary_dict['test_'+seg_t+'_type'] = \
                    seg_dep_eval.eval(len(pred_seg_heads[seg_t]), self.dataset['test_word_syndep_sent'], word_test_pos,
                                      self.dataset['test_word_syndep_head'], self.dataset['test_word_syndep_type'],
                                      pred_seg_words[seg_t], pred_seg_postags[seg_t], pred_seg_heads[seg_t], pred_seg_types[seg_t],
                                      punct_set=self.hparams.punctuation, symbolic_root=False)


            print("Start Pos dev eval:")
            self.summary_dict['test_char_pos'] = pos_eval.eval(self.dataset['test_char_syndep_sent'], self.dataset['test_char_syndep_sent'], self.dataset['test_char_syndep_pos'], pos_pred)
            #if self.dataset_type != 'zms':
            for seg_t in seg_types:
                self.summary_dict['test_'+seg_t+'_pos'] = pos_eval.eval(self.dataset['test_word_syndep_sent'], pred_seg_words[seg_t],
                                                                           self.dataset['test_word_syndep_pos'], pred_seg_postags[seg_t])
        else:

            self.summary_dict['test_word_synconst'] = evaluate.evalb(self.evalb_dir,
                                                                     self.dataset['test_word_synconst_tree'],
                                                                     syntree_pred)
            word_pred_heads = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            word_pred_types = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(word_pred_heads) == len(word_pred_types)
            assert len(word_pred_types) == len(self.dataset['test_word_syndep_type'])

            print("Start Pos dev eval:")
            self.summary_dict['test_word_pos'] = pos_eval.eval(self.dataset['test_word_syndep_sent'],self.dataset['test_word_syndep_sent'],self.dataset['test_word_syndep_pos'], pos_pred)

            const_word_tree_pred = []
            const_word_tree_gold = []
            cun_error = 0
            for pred_char, gold_char in zip(syntree_predinput, self.dataset['test_char_synconst_tree']):
                gold_char_tree = self.const_wordtree(gold_char)
                gold_char_sent = [(leaf.tag, leaf.word) for leaf in gold_char_tree.leaves()]
                pred_char_tree = self.predinput_word2char(pred_char, gold_char_sent)
                gold_leaves = list(gold_char_tree.leaves())
                predicted_leaves = list(pred_char_tree.leaves())
                if len(gold_leaves)!=len(predicted_leaves):
                    print("gold:", gold_char_sent)
                    print(gold_char.linearize())
                    print("pred:", [(leaf.tag, leaf.word) for leaf in pred_char_tree.leaves()])
                    print(pred_char.linearize())
                    cun_error += 1
                    continue
                assert all(
                    gold_leaf.word == predicted_leaf.word
                    for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))
                const_word_tree_pred.append(self.const_wordtree(pred_char_tree))
                const_word_tree_gold.append(gold_char_tree)
            print("error:", cun_error)

            print("predinput word constituent eval:")
            self.summary_dict['test_predinput_synconst'] = evaluate.evalb(self.evalb_dir, const_word_tree_gold,
                                                                         const_word_tree_pred)

            predinput_words = [[leaf.word for leaf in tree.leaves()] for tree in syntree_predinput]
            predinput_pos = [[leaf.tag for leaf in tree.leaves()] for tree in syntree_predinput]
            predinput_heads = [[leaf.father for leaf in tree.leaves()] for tree in syntree_predinput]
            predinput_types = [[leaf.type for leaf in tree.leaves()] for tree in syntree_predinput]
            print("predinput word dep eval:")
            self.summary_dict['test_predinput_word'], self.summary_dict['test_predinput_head'], self.summary_dict[
                'test_predinput_type'] = \
                seg_dep_eval.eval(len(predinput_heads), self.dataset['test_word_syndep_sent'], word_test_pos,
                                  self.dataset['test_word_syndep_head'], self.dataset['test_word_syndep_type'],
                                  predinput_words, predinput_pos, predinput_heads,
                                  predinput_types,
                                  punct_set=self.hparams.punctuation, symbolic_root=False)

        print("Word with gold segment dep :")
        self.summary_dict['test_word_syndep_uas'], self.summary_dict['test_word_syndep_las'] = \
            dep_eval.eval(len(word_pred_heads), self.dataset['test_word_syndep_sent'], word_test_pos,
                          word_pred_heads, word_pred_types, self.dataset['test_word_syndep_head'],
                          self.dataset['test_word_syndep_type'],
                          punct_set=self.hparams.punctuation, symbolic_root=False)


        print("===============================================")
        # print("Start Pos dev eval:")
        #
        # self.summary_dict['test_pos'] = pos_eval.eval(self.dataset['test_syndep_pos'], pos_pred)




