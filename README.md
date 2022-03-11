# CCharPar

This is the main code of paper "Neural Character-Level Syntactic Parsing for Chinese", which is published in JAIR 2022.

## Contents
- [CCharPar](#ccharpar)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Training](#training)
  - [Citation](#citation)

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 1.0.0. This code has not been tested with PyTorch 1.6.0, but it should work.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. 
* [transformers](https://github.com/huggingface/transformers) PyTorch 1.0.0+ or any compatible version (only required when using BERT, XLNet, etc.)

## Training

```bash
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
```

## Citation
If you use this software for research, please cite our paper as follows:
```
@article{li2022neural,
  title={Neural Character-Level Syntactic Parsing for Chinese},
  author={Li, Zuchao and Zhou, Junru and Zhao, Hai and Zhang, Zhisong and Li, Haonan and Ju, Yuqi},
  journal={Journal of Artificial Intelligence Research},
  volume={73},
  pages={461--509},
  year={2022}
}
```
