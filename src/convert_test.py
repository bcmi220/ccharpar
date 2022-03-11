from dep_reader_test import CoNLLXReader
import dep_eval
import numpy as np

dep_test_reader = CoNLLXReader("constcovert/gold_dev_3.3.0.sd")
#dep_test_reader = CoNLLXReader("constcovert/goldtest.sd")
#dep_test_reader = CoNLLXReader("data/test_pro.conll")
#dep_pred_reader = CoNLLXReader("constcovert/2layers_bert_gw.sd")
dep_pred_reader = CoNLLXReader("constcovert/12layers_cwt_dev_predict.sd")
#dep_pred_reader = CoNLLXReader("constcovert/baseline_12cwt.sd")
#dep_pred_reader = CoNLLXReader("constcovert/12layers_3.3.0.sd")

dep_test_data = []
test_inst = dep_test_reader.getNext()
dep_test_headid = np.zeros([3000, 300], dtype=int)
dep_test_type = []
dep_test_word = []
dep_test_pos = []
dep_test_lengs = np.zeros(3000, dtype=int)
cun = 0
while test_inst is not None:
    inst_size = test_inst.length()
    dep_test_lengs[cun] = inst_size
    sent = test_inst.sentence
    dep_test_data.append((sent.words, test_inst.postags, test_inst.heads, test_inst.types))
    for i in range(inst_size):
        dep_test_headid[cun][i] = test_inst.heads[i]
    dep_test_type.append(test_inst.types)
    dep_test_word.append(sent.words)
    dep_test_pos.append(sent.postags)
    # dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
    test_inst = dep_test_reader.getNext()
    cun = cun + 1

dep_test_reader.close()

dep_pred_data = []
pred_inst = dep_pred_reader.getNext()
dep_pred_headid = np.zeros([3000, 300], dtype=int)
dep_pred_type = []
dep_pred_word = []
dep_pred_pos = []
dep_pred_lengs = np.zeros(3000, dtype=int)
cun = 0
while pred_inst is not None:
    inst_size = pred_inst.length()
    dep_pred_lengs[cun] = inst_size
    sent = pred_inst.sentence
    dep_pred_data.append((sent.words, pred_inst.postags, pred_inst.heads, pred_inst.types))
    for i in range(inst_size):
        dep_pred_headid[cun][i] = pred_inst.heads[i]
    dep_pred_type.append(pred_inst.types)
    dep_pred_word.append(sent.words)
    dep_pred_pos.append(sent.postags)
    # dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
    pred_inst = dep_pred_reader.getNext()
    cun = cun + 1

dep_pred_reader.close()
punct_set = '.' '``' "''" ':' ','
assert len(dep_test_headid) == len(dep_pred_headid)
stats, stats_nopunc, stats_root, test_total_inst = dep_eval.eval(len(dep_test_headid), dep_test_word, dep_test_pos, dep_pred_headid,
                                                                 dep_pred_type, dep_test_headid, dep_test_type,
                                                                 dep_test_lengs, punct_set=punct_set,
                                                          symbolic_root=False)

test_ucorrect, test_lcorrect, test_total, test_ucomlpete_match, test_lcomplete_match = stats
test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc, test_ucomlpete_match_nopunc, test_lcomplete_match_nopunc = stats_nopunc
test_root_correct, test_total_root = stats_root
best_epoch = 0
print(
    'best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
        test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total,
        test_lcorrect * 100 / test_total,
        test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst,
        best_epoch))
print(
    'best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
        test_ucorrect_nopunc * 100 / test_total_nopunc,
        test_lcorrect_nopunc * 100 / test_total_nopunc,
        test_ucomlpete_match_nopunc * 100 / test_total_inst,
        test_lcomplete_match_nopunc * 100 / test_total_inst,
        best_epoch))
print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
    test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
print(
    '============================================================================================================================')
