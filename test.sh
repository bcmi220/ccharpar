#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python src_chardep/main.py test \
--dataset char \
--lambda-l 0.5 \
--lambda-r 0.9 \
--model-path-base models/1003_scdt12layer_predtag_best_dev=91.00_devuas=92.86_devlas=87.97_devpos=94.85.pt
