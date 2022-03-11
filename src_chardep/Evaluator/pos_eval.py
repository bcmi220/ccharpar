__author__ = 'max'

import re
import numpy as np


def _print_f1(total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("{}: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(message, precision, recall, f1))
    return precision, recall, f1

def eval(gold_sents, pred_sents, gold_pos, pred_pos):

    corr = 0.
    total = 0.
    g_words_total = 0
    c_words= 0
    c_pos = 0
    p_words_total = 0
    assert len(gold_pos) == len(pred_pos)

    for g_words, p_words, golds, predictions in zip(gold_sents, pred_sents, gold_pos, pred_pos):
        #assert len(golds) == len(predictions)
        g_len = 0
        for p_word, p_pos in zip(p_words, predictions):
            if p_pos != 'PU':
                p_words_total += 1
        for g_word, g_pos in zip(g_words, golds):
            p_len = 0
            if g_pos != 'PU':
                g_words_total += 1
            cun = 0
            for p_word, p_pos in zip(p_words, predictions):
                if g_word == p_word and p_len == g_len:
                    c_words += 1
                    if g_pos == p_pos and g_pos != 'PU':
                        c_pos += 1
                    cun +=1

                p_len += len(p_word)
            g_len += len(g_word)

        # for gold, prediction in zip(golds, predictions):
        #     if gold == prediction:
        #         corr += 1
        #     total += 1

    seg_pos_prec, seg_pos_recall, seg_pos_f1 = _print_f1(g_words_total, p_words_total, c_pos,
                                                            "W. Punct: words segment f1")
    print('POS: pred total: %d, gold total: %d, acc: %.4f%%' % (
        p_words_total, g_words_total, seg_pos_f1))

    return seg_pos_f1

