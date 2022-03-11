import functools

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
#import src_dep_const_test.chart_helper as chart_helper
import Decoder.hpsg_decoder as hpsg_helper
import Decoder.synconst_scorer as synconst_scorer
import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))
#
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
#
class Synconst_score(nn.Module):
    def __init__(self, hparams, synconst_vocab):
        super(Synconst_score, self).__init__()

        self.hparams = hparams
        input_dim = hparams.d_model

        self.f_label = nn.Sequential(
            nn.Linear(input_dim, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, synconst_vocab.size - 1),
        )

    def label_score(self, span_rep):
        return self.f_label(span_rep)

    def forward(self, fencepost_annotations_start, fencepost_annotations_end):

        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                             - torch.unsqueeze(fencepost_annotations_start, 1))

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
            ], 2)

        return label_scores_chart

class BiLinear(nn.Module):
    '''
    Bi-linear layer
    '''
    def __init__(self, left_features, right_features, out_features, bias=True):
        '''

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        '''

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        '''
        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(-1, self.left_features)
        input_right = input_right.view(-1, self.right_features)

        # output [batch, out_features]
        output = nn.functional.bilinear(input_left, input_right, self.U, self.bias)
        output = output + nn.functional.linear(input_left, self.W_l, None) + nn.functional.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output
#
class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, hparams):
        super(BiAAttention, self).__init__()
        self.hparams = hparams

        self.dep_weight = nn.Parameter(torch_t.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
        nn.init.xavier_uniform_(self.dep_weight)

    def forward(self, input_d, input_e, input_s = None):

        score = torch.matmul(torch.cat(
            [input_d, torch_t.FloatTensor(input_d.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), self.dep_weight)
        score1 = torch.matmul(score, torch.transpose(torch.cat(
            [input_e, torch_t.FloatTensor(input_e.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), 0, 1))

        return score1

class Dep_score(nn.Module):
    def __init__(self, hparams, num_labels):
        super(Dep_score, self).__init__()

        self.dropout_out = nn.Dropout2d(p=0.33)
        self.hparams = hparams
        out_dim = hparams.d_biaffine#d_biaffine
        self.arc_h = nn.Linear(hparams.d_model, hparams.d_biaffine)
        self.arc_c = nn.Linear(hparams.d_model, hparams.d_biaffine)

        self.attention = BiAAttention(hparams)

        self.type_h = nn.Linear(hparams.d_model, hparams.d_label_hidden)
        self.type_c = nn.Linear(hparams.d_model, hparams.d_label_hidden)
        self.bilinear = BiLinear(hparams.d_label_hidden, hparams.d_label_hidden, num_labels)

    def forward(self, outputs, outpute):
        # output from rnn [batch, length, hidden_size]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
        outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)

        # output size [batch, length, arc_space]
        arc_h = nn.functional.relu(self.arc_h(outputs))
        arc_c = nn.functional.relu(self.arc_c(outpute))

        # output size [batch, length, type_space]
        type_h = nn.functional.relu(self.type_h(outputs))
        type_c = nn.functional.relu(self.type_c(outpute))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=0)
        type = torch.cat([type_h, type_c], dim=0)

        arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
        arc_h, arc_c = arc.chunk(2, 0)

        type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
        type_h, type_c = type.chunk(2, 0)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        out_arc = self.attention(arc_h, arc_c)
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type

class Uniform_Decoder(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            hparams,
    ):
        super(Uniform_Decoder, self).__init__()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab

        self.hparams = hparams
        if self.hparams.joint_syn_const:
            self.synconst_f = Synconst_score(hparams, label_vocab)
        if self.hparams.joint_syn_dep:
            self.dep_score = Dep_score(hparams, type_vocab.size)
            self.loss_func = torch.nn.CrossEntropyLoss(size_average=False)
            self.loss_funt = torch.nn.CrossEntropyLoss(size_average=False)

        if self.hparams.joint_pos and not self.hparams.use_pos_layer:
            self.f_pos = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_label_hidden),
                LayerNormalization(hparams.d_label_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_label_hidden, tag_vocab.size),
            )
            self.loss_pos = torch.nn.CrossEntropyLoss(size_average=False)

    def cal_loss(self, annotations, fencepost_annotations_start, fencepost_annotations_end, batch_idxs, sentences, gold_trees):

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        syndep_loss = 0
        synconst_loss = 0
        pos_loss = 0
        loss = 0
        if self.hparams.joint_syn_const:
            with torch.no_grad():
                for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                    label_scores_chart = self.synconst_f(fencepost_annotations_start[start:end,:],
                                                                            fencepost_annotations_end[start:end, :])
                    label_scores_chart_np = label_scores_chart.cpu().data.numpy()
                    decoder_args = dict(
                        sentence_len=len(sentences[i]),
                        label_scores_chart=label_scores_chart_np,
                        gold=gold_trees[i],
                        label_vocab=self.label_vocab,
                        is_train=True)
                    p_score, p_i, p_j, p_label, p_augment = synconst_scorer.decode(False, **decoder_args)
                    g_score, g_i, g_j, g_label, g_augment = synconst_scorer.decode(True, **decoder_args)
                    paugment_total += p_augment
                    num_p += p_i.shape[0]
                    pis.append(p_i + start)
                    pjs.append(p_j + start)
                    gis.append(g_i + start)
                    gjs.append(g_j + start)
                    plabels.append(p_label)
                    glabels.append(g_label)

            cells_i = from_numpy(np.concatenate(pis + gis))
            cells_j = from_numpy(np.concatenate(pjs + gjs))
            cells_label = from_numpy(np.concatenate(plabels + glabels))

            cells_label_scores = self.synconst_f.label_score(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
            cells_label_scores = torch.cat([
                cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
                cells_label_scores
            ], 1)
            cells_label_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
            loss = cells_label_scores[:num_p].sum() - cells_label_scores[num_p:].sum() + paugment_total
            synconst_loss = loss

        if self.hparams.joint_syn_dep:
            # syndep loss
            cun = 0
            for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                #[start,....,end-1]->[<root>,1, 2,...,n]
                leng = end - start
                arc_score, type_score = self.dep_score(annotations[start:end,:], annotations[start:end,:])
                # arc_score, type_score = self.dep_score(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:])
                arc_gather = [leaf.father for leaf in gold_trees[snum].leaves()]
                type_gather = [self.type_vocab.index(leaf.type) for leaf in gold_trees[snum].leaves()]
                cun += 1
                assert len(arc_gather) == leng - 1
                arc_score = torch.transpose(arc_score,0, 1)
                dep_loss = self.loss_func(arc_score[1:, :], from_numpy(np.array(arc_gather)).requires_grad_(False)) \
                       +  self.loss_funt(type_score[1:, :],from_numpy(np.array(type_gather)).requires_grad_(False))
                loss = loss +  dep_loss
                syndep_loss = syndep_loss + dep_loss

        if self.hparams.joint_pos and not self.hparams.use_pos_layer:
            for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                #[start,....,end-1]->[<root>,1, 2,...,n]
                leng = end - start
                pos_score = self.f_pos(annotations[start:end,:])
                pos_gather = [self.tag_vocab.index(leaf.goldtag) for leaf in gold_trees[snum].leaves()]
                assert len(pos_gather) == leng - 1
                pos_loss = pos_loss + self.loss_pos(pos_score[1:, :], from_numpy(np.array(pos_gather)).requires_grad_(False))

            loss = loss + pos_loss

        return loss, synconst_loss, syndep_loss, pos_loss

    def decode(self, annotations, fencepost_annotations_start, fencepost_annotations_end, batch_idxs, sentences, pred_tags = None):

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        syntree_pred = []
        score_list = []
        for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
            pos_pred = None
            if pred_tags is not None:
                pos_pred = pred_tags[i]
            if self.hparams.joint_pos and not self.hparams.use_pos_layer:
                pos_score = self.f_pos(annotations[start:end, :])
                pos_score_np = pos_score.cpu().data.numpy()
                pos_score_np = pos_score_np[1:, :]  # remove root
                pos_pred = pos_score_np.argmax(axis=1)
                pos_pred = [self.tag_vocab.value(pos_pred_index) for pos_pred_index in pos_pred]

            syn_tree, score = self.hpsg_decoder(fencepost_annotations_start[start:end, :],
                                                  fencepost_annotations_end[start:end, :], annotations[start:end, :], sentences[i], pos_pred)

            syntree_pred.append(syn_tree)
            score_list.append(score)

        return syntree_pred, score_list

    def hpsg_decoder(self, fencepost_annotations_start, fencepost_annotations_end, annotation, sentence, pred_pos = None):

        leng = annotation.size(0)

        if self.hparams.joint_syn_const:
            label_scores_chart = self.synconst_f(fencepost_annotations_start, fencepost_annotations_end)
            label_scores_chart_np = label_scores_chart.cpu().data.numpy()
        else:
            label_scores_chart_np = np.zeros((leng, leng, self.label_vocab.size), dtype= np.float32)

        if self.hparams.joint_syn_dep:
            arc_score, type_score = self.dep_score(annotation, annotation)
            arc_score_dc = torch.transpose(arc_score, 0, 1)
            arc_dc_np = arc_score_dc.cpu().data.numpy()

            type_np = type_score.cpu().data.numpy()
            type_np = type_np[1:, :]  # remove root
            type = type_np.argmax(axis=1)

        else:
            arc_dc_np = np.zeros((leng, leng), dtype= np.float32)
            type = np.zeros(leng, dtype= int)

        score, p_i, p_j, p_label, p_father = hpsg_helper.decode(sentence_len=len(sentence),
            label_scores_chart=label_scores_chart_np * self.hparams.const_lada,
            type_scores_chart=arc_dc_np * (1.0 - self.hparams.const_lada))
        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.

        #make arrange table, sort the verb id
        idx = -1

        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab.value(label_idx)
            if (i + 1) >= j:
                tag, word = sentence[i]
                pred_tag = None
                if pred_pos is not None:
                    pred_tag = pred_pos[i]
                else:
                    pred_tag = tag
                tree = trees.LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(type[i]), pred_tag)
                if label:
                    assert label[0] != Sub_Head
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label and label[0] != Sub_Head:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree_list = make_tree()
        assert len(tree_list) == 1
        tree = tree_list[0]
        return tree.convert(), score