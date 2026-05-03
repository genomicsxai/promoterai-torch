#!/usr/bin/env bash
set -euo pipefail

models=(
    promoterAI_v1_hg38
    promoterAI_v1_hg38_finetune
    promoterAI_v1_hg38_mm10
    promoterAI_v1_hg38_mm10_finetune
)

mkdir -p examples/data/track_parity

# Compare full predicted tracks on deterministic random one-hot sequences.
for model in "${models[@]}"; do
    echo time python examples/compare_tf_torch_tracks_random.py -v \
        --keras_model models/${model} \
        --torch_checkpoint models/${model}.pt \
        --n_sequences 16 \
        --batch_size 1 \
        --tf_device gpu \
        --output_csv examples/data/track_parity/${model}.random.csv
done | simple_gpu_scheduler --gpus 0,1

# Compare full predicted tracks on selected real promoters.
for model in "${models[@]}"; do
    echo time python examples/compare_tf_torch_tracks_promoters.py -v \
        --keras_model models/${model} \
        --torch_checkpoint models/${model}.pt \
        --promoter DNAJC9:chr10:73247254:- \
        --promoter TERT:chr5:1294988:- \
        --promoter SFSWAP:chr12:131710589:+ \
        --batch_size 1 \
        --tf_device gpu \
        --output_csv examples/data/track_parity/${model}.DNAJC9_TERT_SFSWAP.csv \
        --fasta "${HG38_FASTA:-/home2/ayh8/data/hg38.fa}"
done | simple_gpu_scheduler --gpus 0,1
