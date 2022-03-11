#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python src_chardep/main.py train \
 --model-path-base models/1014_scdt8layer_notag \
 --epochs 200 \
 --joint-syn-dep \
 --joint-syn-const \
 --pos-layer 4 \
 --use-words \
 --const-lada 0.7 \
 --dataset char \
 --num-layers 8 \
 --num-heads 8 \
 --learning-rate 0.0005 \
 --batch-size 200 \
 --eval-batch-size 30 \
 --subbatch-max-tokens 1000 \
 --embedding-path data/csskip.gz \
 --model-name cl/1014_scdt8layer_notag \
 --embedding-type sskip \
 --checks-per-epoch 1
