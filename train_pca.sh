#!/bin/sh

rm checkpoints/pca/*
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/PCA \
    --low-rank PCA --low-rank-scale 20 --low-rank-threshold 20 \
    --arch transformerLowRank_lm --share-decoder-input-output-embed \
    --log-interval 20 \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 2048 --sample-break-mode none \
    --max-tokens 2048 --update-freq 2 \
    --max-update 50000

# fairseq-train \
#     data-bin/iwslt14.tokenized.de-en \
#     --save-dir checkpoints/svd \
#     --arch transformerLowRank_iwslt_de_en --share-decoder-input-output-embed \
#     --low-rank SVD \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr-scheduler polynomial_decay --lr 5e-5 --total-num-update 30000 --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --max-epoch 85 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric