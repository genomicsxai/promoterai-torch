# Convert all official PromoterAI models to pytorch.
for model in "promoterAI_v1_hg38" "promoterAI_v1_hg38_finetune" "promoterAI_v1_hg38_mm10" "promoterAI_v1_hg38_mm10_finetune"; do
    echo promoterai-torch convert \
        --keras_model models/${model} \
        --output models/${model}.pt \
        --input_length 20480 \
        --output_length 4096;
done | parallel

# Extract promoter variants for a couple of genes to test on
zgrep -E "chrom|SFSWAP" precomputed_scores/promoterAI_tss500.tsv.gz > precomputed_scores/promoterAI_tss500_SFSWAP.tsv
zgrep -E "chrom|DNAJC9" precomputed_scores/promoterAI_tss500.tsv.gz > precomputed_scores/promoterAI_tss500_DNAJC9.tsv
zgrep -E "chrom|TERT" precomputed_scores/promoterAI_tss500.tsv.gz > precomputed_scores/promoterAI_tss500_TERT.tsv
zcat precomputed_scores/promoterAI_tss500.tsv.gz | head -501 > precomputed_scores/promoterAI_tss500_head500.tsv
zcat precomputed_scores/promoterAI_tss500.tsv.gz | head -101 > precomputed_scores/promoterAI_tss500_head100.tsv

# Predict using PyTorch models.
# simple_gpu_scheduler is a separate python package to parallelize GPU jobs.
for model in "promoterAI_v1_hg38_finetune" "promoterAI_v1_hg38_mm10_finetune"; do
    for subset in "DNAJC9" "TERT" "SFSWAP" "head100" "head500" ; do 
        echo time promoterai-torch score --verbose \
            --model_checkpoint models/${model}.pt \
            --var_file data/precomputed_scores/promoterAI_tss500_${subset}.tsv \
            --output data/torch/promoterAI_tss500_${subset}.${model}.tsv.gz \
            --fasta_file data/hg38.fa \
            --input_length 20480 \
            --batch_size 4;
    done;
done | simple_gpu_scheduler --gpus 0 1

# Predict using TF install. Obviously run in appropriate env.
for model in "promoterAI_v1_hg38_finetune" "promoterAI_v1_hg38_mm10_finetune"; do
    for subset in "DNAJC9" "TERT" "SFSWAP" "head100" "head500"; do 
        echo time promoterai \
            --model_folder models/promoterAI_v1_hg38_mm10 \
            --var_file data/precomputed_scores/promoterAI_tss500_${subset}.tsv \
            --fasta_file /home2/ayh8/data/hg38.fa \
            --input_length 20480
    done;
done | simple_gpu_scheduler --gpus 0 1
