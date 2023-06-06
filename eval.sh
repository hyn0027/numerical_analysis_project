#!/bin/sh
fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/PCA5/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400