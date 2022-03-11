__author__ = 'max'

import re
import numpy as np

def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None


def is_punctuation(word, pos, punct_set=None):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set or pos == 'PU' # for chinese


def _print_f1(total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("{}: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(message, precision, recall, f1))
    return precision, recall, f1

def eval(batch_size, g_words, g_postags, g_heads, g_types, p_words, p_postags, p_heads, p_types,
         punct_set=None, symbolic_root=False):

    g_words_total = 0
    c_words = 0
    c_heads = 0
    c_types = 0
    p_words_total = 0

    g_nopunc_words_total = 0
    c_nopunc_words = 0
    c_nopunc_heads = 0
    c_nopunc_types = 0
    p_nopunc_words_total = 0



    # ucorr = 0.
    # lcorr = 0.
    # total = 0.
    # ucomplete_match = 0.
    # lcomplete_match = 0.
    #
    # ucorr_nopunc = 0.
    # lcorr_nopunc = 0.
    # total_nopunc = 0.
    # ucomplete_match_nopunc = 0.
    # lcomplete_match_nopunc = 0.

    # corr_root = 0.
    # total_root = 0.
    start = 1 if symbolic_root else 0
    for i in range(batch_size):
        g_len = 0
        g_total_len = []
        p_total_len = []
        p_len = 0
        for p_word, p_pos in zip(p_words[i], p_postags[i]):
            p_words_total += 1
            if not is_punctuation(p_word, p_pos, punct_set):
                p_nopunc_words_total += 1
            p_len += len(p_word)
            p_total_len.append(p_len)

        for g_word in g_words[i]:
            g_len += len(g_word)
            g_total_len.append(g_len)

        g_len = 0
        for g_word, g_pos, g_head, g_type in zip(g_words[i], g_postags[i], g_heads[i], g_types[i]):
            p_len = 0
            g_words_total += 1
            cun = 0
            for p_word, p_head, p_type in zip(p_words[i], p_heads[i], p_types[i]):
                if g_word == p_word and p_len == g_len:
                    c_words += 1
                    if (p_head == 0 and g_head == 0) or \
                            (p_head > 0 and g_head > 0 and p_total_len[p_head-1] == g_total_len[g_head-1] and g_words[i][g_head-1] == p_words[i][p_head-1]):
                        c_heads += 1
                        if p_type == g_type:
                            c_types += 1
                    cun +=1

                p_len += len(p_word)

            if not is_punctuation(g_word, g_pos, punct_set):
                p_len = 0
                g_nopunc_words_total += 1
                for p_word, p_pos, p_head, p_type in zip(p_words[i], p_postags[i], p_heads[i], p_types[i]):
                    if g_word == p_word and p_len == g_len and not is_punctuation(p_word, p_pos, punct_set):
                        c_nopunc_words += 1
                        if (p_head == 0 and g_head == 0) or \
                            (p_head > 0 and g_head > 0 and p_total_len[p_head - 1] == g_total_len[g_head - 1] and g_words[i][g_head - 1] == p_words[i][p_head - 1]):
                            c_nopunc_heads += 1
                            if p_type == g_type:
                                c_nopunc_types += 1
                    p_len += len(p_word)

            g_len += len(g_word)

    seg_word_prec, seg_word_recall, seg_word_f1 = _print_f1(g_words_total, p_words_total, c_words, "W. Punct: words segment f1")
    seg_head_prec, seg_head_recall, seg_head_f1 = _print_f1(g_words_total, p_words_total, c_heads, "W. Punct: heads segment f1")
    seg_type_prec, seg_type_recall, seg_type_f1 = _print_f1(g_words_total, p_words_total, c_types, "W. Punct: types segment f1")

    seg_nopunc_word_prec, seg_nopunc_word_recall, seg_nopunc_word_f1 = \
        _print_f1(g_nopunc_words_total, p_nopunc_words_total, c_nopunc_words, "Wo Punct: words segment f1")
    seg_nopunc_head_prec, seg_nopunc_head_recall, seg_nopunc_head_f1 = \
        _print_f1(g_nopunc_words_total, p_nopunc_words_total, c_nopunc_heads, "Wo Punct: heads segment f1")
    seg_nopunc_type_prec, seg_nopunc_type_recall, seg_nopunc_type_f1 = \
        _print_f1(g_nopunc_words_total, p_nopunc_words_total, c_nopunc_types, "Wo Punct: types segment f1")


    return seg_nopunc_word_f1, seg_nopunc_head_f1, seg_nopunc_type_f1, #seg_word_f1, seg_head_f1, seg_type_f1
