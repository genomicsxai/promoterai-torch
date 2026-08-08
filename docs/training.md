# Training models

> [!Warning]
> I have verified that this code executes and can train/finetune models. I have also endeavored to make sure that it aligns with the training/finetuning behavior of the official PromoterAI repo. However, I have not attempted to reproduce their models, and cannot ensure there isn't some hidden divergence in some low-level PyTorch vs Tensorflow/Keras behavior.

Fine-tuning or training from scratch using the built-in scripts requires the
`train` extra described in [Install](../README.md#install).

Weights & Biases logging is optional for training and fine-tuning:

```sh
pip install "promoterai-torch[train,wandb]"
uv add promoterai-torch --extra train --extra wandb
```

## Data preprocessing

We follow the original PromoterAI's repo and preprocess track and sequence data into chunks for training. This needs to be done separately for each chromosome, e.g.

```sh
promoterai-torch preprocess \
    --hdf5_folder data/hdf5/human --tss_file tss_hg38.tsv \
    --fasta_file hg38.fa --bigwig_files hg38_tracks.tsv \
    --chrom chr1 --input_length 32768 --output_length 16384
```

Once done, you can then train on this dataset.

Check the generated chunks before launching a long run:

```sh
promoterai-torch check-hdf5 \
    --paths data/hdf5/human data/hdf5/mouse
```

Add `--full-read` to read every `x` and `y` dataset value instead of the default
first/last-row check.

```sh
promoterai-torch train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32 \
    --wandb_project promoterai-torch
```

Training shows rank-0 train/validation progress bars by default. Use
`--log_every_batches 100` for explicit batch-loss lines, or `--no_progress` in
non-interactive logs. W&B always receives epoch loss/LR/weight-decay metrics
when enabled; use `--wandb_log_every_batches 100` to also log batch losses.
If omitted, `--log_every_batches` is reused as the W&B batch logging cadence.

Training writes `best_model.pt` when validation improves and `latest_model.pt`
after every epoch. `best_model.pt` contains only model weights and architecture
arguments for inference, while `latest_model.pt` also contains optimizer and
other training state needed to resume. Resume a pre-empted run explicitly:

```sh
promoterai-torch train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32 \
    --resume_checkpoint checkpoints/run1/latest_model.pt
```

Or opt into automatic epoch-level resumption from
`<checkpoint_folder>/latest_model.pt` when it exists:

```sh
promoterai-torch train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32 \
    --auto_resume
```

Multi-GPU training is supported via `torchrun`:

```sh
torchrun --nproc_per_node=4 -m promoterai_torch.train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --hdf5_nonhuman_folders data/hdf5/mouse \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32
```

For training, `--batch_size` is the global batch size. In multi-GPU runs it must
divide evenly by the number of ranks; the script uses `batch_size / world_size`
per GPU. For example, `--batch_size 32` with 8 GPUs uses 4 samples per GPU and
keeps `steps_per_epoch=int(sum(dataset_sizes) / 10)` independent of GPU count.

## Fine-tune on variants

```sh
promoterai-torch finetune \
    --model_checkpoint best_model.pt \
    --var_file data/annotation/finetune_gtex.tsv \
    --fasta_file hg38.fa \
    --input_length 20480 \
    --batch_size 8 \
    --epochs 100 \
    --amp_dtype bf16 \
    --wandb_project promoterai-torch --wandb_run_name gtex-finetune
```

Mixed precision is opt-in via `--amp_dtype bf16` or `--amp_dtype fp16`;
the default is full-precision training. The fine-tuned base model is saved to
`best_model_finetune/best_model.pt`. Only the first output head is trained; all
other weights are frozen.

Finetuning also writes `best_model_finetune/latest_model.pt` after every
completed epoch. Resume automatically after a preemption with `--auto_resume`,
or select a checkpoint explicitly with `--resume_checkpoint PATH`. An explicit
checkpoint takes precedence, and `--epochs` remains the total target epoch
count.

Multi-GPU finetuning is supported through `torchrun`:

```sh
torchrun --nproc_per_node=4 -m promoterai_torch.finetune \
    --model_checkpoint best_model.pt \
    --var_file data/annotation/finetune_gtex.tsv \
    --fasta_file hg38.fa \
    --input_length 20480 \
    --batch_size 8 \
    --amp_dtype bf16
```

`--batch_size` is global and must divide evenly across ranks. The frozen
backbone, including BatchNorm statistics, remains in inference mode; only the
first output head is updated.

To strip optimizer and other training state from an existing checkpoint:

```sh
promoterai-torch export-inference \
    --checkpoint checkpoints/run1/latest_model.pt \
    --output checkpoints/run1/model_inference.pt
```

The variant TSV must include `chrom`, `pos`, `ref`, `alt`, `strand`, `z` (expression z-score target), `in_cds`, `spliceai`, `p_under`, `p_over`, `gene` columns (matching the GTEx outlier format).
