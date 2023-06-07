#!/bin/sh

rm checkpoints/linformer4/*

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/linformer4 \
    --low-rank Linformer --low-rank-scale 50 --low-rank-threshold 50 \
    --arch transformerLowRank_lm --share-decoder-input-output-embed \
    --log-interval 20 \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 1024 --sample-break-mode none \
    --max-tokens 2048 --update-freq 2 \
    --max-update 40000 \
    --ddp-backend=legacy_ddp \
    --max-epoch 40