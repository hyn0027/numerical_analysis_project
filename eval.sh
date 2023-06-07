#!/bin/sh
fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/PCA1/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 2048 \
    --context-window 400