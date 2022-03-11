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
from dataset import Dataset
from Eval import EvalManyTask
import trees
import vocabulary
import makehp
import Zparser
import utils

tokens = Zparser

uid = uuid.uuid4().hex[:6]

def torch_load(load_path):
    if Zparser.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return makehp.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=450,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,

        partitioned=True,
        use_cat=False,
        use_lstm = False,
        joint_syn_dep = False,
        joint_syn_const = False,
        joint_pos = False,
        pos_layers = 4,
        use_pos_layer = False,

        const_lada = 0.5,

        num_layers=12,
        d_model=1024,
        num_heads=8,#12
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_biaffine = 1024,

        attention_dropout=0.2,
        embedding_dropout=0.2,
        relu_dropout=0.2,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_elmo = False,
        use_bert=False,
        use_xlnet = False,
        use_chars_lstm=False,

        dataset = 'char',

        model_name = "dep+const",
        embedding_type = 'random',
        #['glove','sskip','random']
        embedding_path = "/data/csskip.gz",
        punctuation='.' '``' "''" ':' ',',

        d_char_emb = 64, # A larger value may be better for use_chars_lstm

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-base-chinese",
        bert_do_lower_case=False,
        bert_transliterate="",
        xlnet_model="xlnet-chinese-mid",
        xlnet_do_lower_case=False,
        pad_left=False,
        )

def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    CTB_dataset = Dataset(hparams)
    CTB_dataset.process(args)

    print("Initializing model...")

    load_path = None
    if load_path is not None:
        print(f"Loading parameters from {load_path}")
        info = torch_load(load_path)
        parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])
    else:
        parser = Zparser.ChartParser(
            CTB_dataset.tag_vocab,
            CTB_dataset.word_vocab,
            CTB_dataset.label_vocab,
            CTB_dataset.char_vocab,
            CTB_dataset.type_vocab,
            hparams,
        )

    print("Initializing optimizer...")
    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    if load_path is not None:
        trainer.load_state_dict(info['optimizer'])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(CTB_dataset.dataset['train_synconst_parse']) / args.checks_per_epoch

    model_path = None
    model_name = hparams.model_name

    print("This is ", model_name)
    start_time = time.time()

    def save_args(hparams):
        arg_path = "models_log/" + model_name + '.arg.json'
        kwargs = hparams.to_dict()
        json.dump({'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    save_args(hparams)

    best_epoch = 0

    evaluator = EvalManyTask(hparams=hparams, dataset=CTB_dataset,
                             eval_batch_size=args.eval_batch_size,
                             evalb_dir=args.evalb_dir, model_path_base=args.model_path_base,
                             log_path="{}_log".format("models_log/" + hparams.model_name))


    train_data = CTB_dataset.dataset['train_synconst_parse']

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break
        #save_path, is_save_model = evaluator.eval_multitask(parser, start_time, epoch)

        np.random.shuffle(train_data)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_data), args.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // args.batch_size)

            parser.train()

            batch_loss_value = 0.0
            batch_loss_syndep = 0.0
            batch_loss_synconst =0.0
            batch_tag_loss = 0
            batch_trees = train_data[start_index:start_index + args.batch_size]
            batch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in batch_trees]

            for subbatch_sentences, subbatch_trees in parser.split_batch(batch_sentences, batch_trees, args.subbatch_max_tokens):
                loss, synconst_loss, syndep_loss, tag_loss = parser.parse_batch(subbatch_sentences, subbatch_trees)

                loss = loss / len(batch_trees)
                syndep_total_loss = syndep_loss / len(batch_trees)
                synconst_loss = synconst_loss / len(batch_trees)
                tag_loss = tag_loss / len(batch_trees)
                if loss > 0:
                    loss_value = float(loss.data.cpu().numpy())
                    batch_loss_value += loss_value

                if synconst_loss > 0:
                    batch_loss_synconst += float(synconst_loss.data.cpu().numpy())
                if syndep_total_loss > 0:
                    batch_loss_syndep += float(syndep_total_loss.data.cpu().numpy())
                if tag_loss > 0:
                    batch_tag_loss += float(tag_loss.data.cpu().numpy())

                if loss > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "synconst-loss {:.4f} "
                "syndep-loss {:.4f} "
                "pos-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_data) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    batch_loss_synconst,
                    batch_loss_syndep,
                    batch_tag_loss,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every

                save_path, is_save_model = evaluator.eval_multitask(parser, start_time, epoch)
                if is_save_model:
                    torch.save({
                        'spec': parser.spec,
                        'state_dict': parser.state_dict(),
                        'optimizer': trainer.state_dict(),
                    }, save_path + ".pt")

        # adjust learning rate at the end of an epoch
        if hparams.step_decay:
            if (total_processed // args.batch_size + 1) > hparams.learning_rate_warmup_steps:
                scheduler.step(evaluator.best_dev_score)

        if epoch % 10 == 0: #save each 10 epoch
            if model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            model_path = "{}".format(
                args.model_path_base)
            print("Saving new best model to {}...".format(model_path))
            torch.save({
                'spec': parser.spec,
                'state_dict': parser.state_dict(),
                'optimizer': trainer.state_dict(),
            }, model_path + ".pt")

def run_test(args):


    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])
    CTB_dataset = Dataset(parser.hparams)
    CTB_dataset.process(args)
    evaluator = EvalManyTask(hparams=parser.hparams, dataset=CTB_dataset,
                                 eval_batch_size=args.eval_batch_size,
                                 evalb_dir=args.evalb_dir, model_path_base=args.model_path_base,
                                 log_path=None)

    start_time = time.time()
    step = (args.lambda_r + 0.1 - args.lambda_l) / 0.1
    lambda_h = args.lambda_l
    for i in range(int(step)):
        parser.hparams.const_lada = lambda_h
        print("lambda_h: ",lambda_h)
        save_path, is_save_model = evaluator.eval_multitask(parser, start_time, 0)
        lambda_h+=0.1

def run_parse(args):
    # if args.output_path != '-' and os.path.exists(args.output_path):
    #     print("Error: output file already exists:", args.output_path)
    #     return

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])
    parser.eval()
    print("Parsing sentences...")
    with open(args.input_path) as input_file:
        sentences = input_file.readlines()
    sentences = [sentence.split() for sentence in sentences]

    # Parser does not do tagging, so use a dummy tag when parsing from raw text
    if 'UNK' in parser.tag_vocab.indices:
        dummy_tag = 'UNK'
    else:
        dummy_tag = parser.tag_vocab.value(0)

    start_time = time.time()

    def save_data(syntree_pred, srlspan_pred, srldep_pred, cun):
        pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
        pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
        appent_string = "_" + str(cun) + ".txt"
        if args.output_path_synconst != '-':
            with open(args.output_path_synconst + appent_string, 'w') as output_file:
                for tree in syntree_pred:
                    output_file.write("{}\n".format(tree.pred_linearize()))
            print("Output written to:", args.output_path_synconst)

        if args.output_path_syndep != '-':
            with open(args.output_path_syndep + appent_string, 'w') as output_file:
                for heads in pred_head:
                    output_file.write("{}\n".format(heads))
            print("Output written to:", args.output_path_syndep)

        if args.output_path_synlabel != '-':
            with open(args.output_path_synlabel + appent_string, 'w') as output_file:
                for labels in pred_type:
                    output_file.write("{}\n".format(labels))
            print("Output written to:", args.output_path_synlabel)

        if args.output_path_hpsg != '-':
            with open(args.output_path_hpsg + appent_string, 'w') as output_file:

                for heads in pred_head:
                    n = len(heads)
                    childs = [[] for i in range(n + 1)]
                    left_p = [i for i in range(n + 1)]
                    right_p = [i for i in range(n + 1)]

                    def dfs(x):
                        for child in childs[x]:
                            dfs(child)
                            left_p[x] = min(left_p[x], left_p[child])
                            right_p[x] = max(right_p[x], right_p[child])

                    for i, head in enumerate(heads):
                        childs[head].append(i + 1)

                    dfs(0)
                    hpsg_list = []
                    for i in range(1, n + 1):
                        hpsg_list.append((left_p[i], right_p[i]))
                    output_file.write("{}\n".format(hpsg_list))

            print("Output written to:", args.output_path_hpsg)

        if args.output_path_srlspan != '-':
            with open(args.output_path_srlspan + appent_string, 'w') as output_file:
                for srlspan in srlspan_pred:
                    output_file.write("{}\n".format(srlspan))
            print("Output written to:", args.output_path_srlspan)

        if args.output_path_srldep != '-':
            with open(args.output_path_srldep + appent_string, 'w') as output_file:
                for srldep in srldep_pred:
                    output_file.write("{}\n".format(srldep))
            print("Output written to:", args.output_path_srldep)

    syntree_pred = []
    srlspan_pred = []
    srldep_pred = []
    cun = 0
    for start_index in range(0, len(sentences), args.eval_batch_size):
        subbatch_sentences = sentences[start_index:start_index+args.eval_batch_size]

        subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
        syntree, srlspan_dict, srldep_dict = parser.parse_batch(subbatch_sentences)
        syntree_pred.extend(syntree)
        srlspan_pred.extend(srlspan_dict)
        srldep_pred.extend(srldep_dict)
        if args.save_per_sentences <= len(syntree_pred) and args.save_per_sentences > 0:
            save_data(syntree_pred, srlspan_pred, srldep_pred, cun)
            syntree_pred = []
            srlspan_pred = []
            srldep_pred = []
            cun += 1

    if 0 < len(syntree_pred):
        save_data(syntree_pred, srlspan_pred, srldep_pred, cun)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", required=True)
    subparser.add_argument("--embedding-type", default="random")

    subparser.add_argument("--model-name", default="char_parser")
    subparser.add_argument("--evalb-dir", default="EVALB/")

    subparser.add_argument("--dataset", default="char")

    subparser.add_argument("--synconst-char-train-path", default="chardata/train_sdctconst.txt")
    subparser.add_argument("--synconst-char-dev-path", default="chardata/dev_sdctconst.txt")
    subparser.add_argument("--syndep-char-train-path", default="chardata/train_sdctdep.txt")
    subparser.add_argument("--syndep-char-dev-path", default="chardata/dev_sdctdep.txt")

    subparser.add_argument("--synconst-zms-train-path", default="chardata/train_zms.txt")
    subparser.add_argument("--synconst-zms-dev-path", default="chardata/dev_zms.txt")
    subparser.add_argument("--syndep-zms-train-path", default="chardata/train_zmsdep.txt")
    subparser.add_argument("--syndep-zms-dev-path", default="chardata/dev_zmsdep.txt")

    subparser.add_argument("--synconst-word-train-path", default="data/train_ctbc.txt")
    subparser.add_argument("--synconst-word-dev-path", default="data/dev_ctbc.txt")
    subparser.add_argument("--syndep-word-train-path", default="data/train_ctbc.conll")
    subparser.add_argument("--syndep-word-dev-path", default="data/dev_ctbc.conll")


    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=30)
    subparser.add_argument("--epochs", type=int, default=150)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--synconst-char-test-path", default="chardata/test_sdctconst.txt")
    subparser.add_argument("--syndep-char-test-path", default="chardata/test_sdctdep.txt")

    subparser.add_argument("--synconst-zms-test-path", default="chardata/test_zms.txt")
    subparser.add_argument("--syndep-zms-test-path", default="chardata/test_zmsdep.txt")

    subparser.add_argument("--synconst-word-test-path", default="data/test_ctbc.txt")
    subparser.add_argument("--syndep-word-test-path", default="data/test_ctbc.conll")

    subparser.add_argument("--test-path-raw", type=str)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--dataset", default="char")
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--embedding-path", default="data/csskip.gz")
    subparser.add_argument("--lambda-l", type=float, default=0.8)
    subparser.add_argument("--lambda-r", type=float, default=0.8)
    subparser.add_argument("--synconst-char-train-path", default="chardata/train_sdctconst.txt")
    subparser.add_argument("--synconst-char-dev-path", default="chardata/dev_sdctconst.txt")
    subparser.add_argument("--syndep-char-train-path", default="chardata/train_sdctdep.txt")
    subparser.add_argument("--syndep-char-dev-path", default="chardata/dev_sdctdep.txt")

    subparser.add_argument("--synconst-zms-train-path", default="chardata/train_zms.txt")
    subparser.add_argument("--synconst-zms-dev-path", default="chardata/dev_zms.txt")
    subparser.add_argument("--syndep-zms-train-path", default="chardata/train_zmsdep.txt")
    subparser.add_argument("--syndep-zms-dev-path", default="chardata/dev_zmsdep.txt")

    subparser.add_argument("--synconst-word-train-path", default="data/train_ctbc.txt")
    subparser.add_argument("--synconst-word-dev-path", default="data/dev_ctbc.txt")
    subparser.add_argument("--syndep-word-train-path", default="data/train_ctbc.conll")
    subparser.add_argument("--syndep-word-dev-path", default="data/dev_ctbc.conll")

    subparser.add_argument("--synconst-char-test-path", default="chardata/test_sdctconst.txt")
    subparser.add_argument("--syndep-char-test-path", default="chardata/test_sdctdep.txt")

    subparser.add_argument("--synconst-zms-test-path", default="chardata/test_zms.txt")
    subparser.add_argument("--syndep-zms-test-path", default="chardata/test_zmsdep.txt")

    subparser.add_argument("--synconst-word-test-path", default="data/test_ctbc.txt")
    subparser.add_argument("--syndep-word-test-path", default="data/test_ctbc.conll")
    subparser.add_argument("--eval-batch-size", type=int, default=30)

    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=run_parse)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", default="data/glove.6B.100d.txt.gz")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--save-per-sentences", type=int, default=-1)
    subparser.add_argument("--input-path", type=str, required=True)
    subparser.add_argument("--output-path-synconst", type=str, default="-")
    subparser.add_argument("--output-path-syndep", type=str, default="-")
    subparser.add_argument("--output-path-synlabel", type=str, default="-")
    subparser.add_argument("--output-path-hpsg", type=str, default="-")
    subparser.add_argument("--output-path-srlspan", type=str, default="-")
    subparser.add_argument("--output-path-srldep", type=str, default="-")
    subparser.add_argument("--eval-batch-size", type=int, default=50)

    args = parser.parse_args()
    args.callback(args)

# %%
if __name__ == "__main__":
    main()
