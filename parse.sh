#!/usr/bin/env bash
python src_srl_syn/main.py parse \
--dataset ptb \
--save-per-sentences 1000 \
--eval-batch-size 50 \
--input-path input_s.txt \
--output-path-synconst output_synconst.txt \
--output-path-syndep output_syndephead.txt \
--output-path-synlabel output_syndeplabel.txt \
--output-path-hpsg output_hpsg.txt \
--output-path-srlspan output_srlspan.txt \
--output-path-srldep output_srldep.txt \
--embedding-path data/glove.gz \
--model-path-base /media/ubuntu/e2c34c12-66c4-4e1c-b4e2-b2260909114f/Bert_train/0428bert_jointpos.pt
