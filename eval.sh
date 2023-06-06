#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/PCA5/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400