"""
Unified CLI entry point.

Usage:
    promoterai-torch preprocess  [args]
    promoterai-torch train       [args]
    promoterai-torch finetune    [args]
    promoterai-torch export-inference [args]
    promoterai-torch score       [args]
    promoterai-torch check-hdf5  [args]
"""

import argparse
import sys

from promoterai_torch.utils import add_wandb_args


def main():
    """Dispatch to the appropriate subcommand (preprocess, train, finetune, convert, score)."""
    parser = argparse.ArgumentParser(
        prog="promoterai-torch",
        description="PyTorch port of PromoterAI — promoter variant effect prediction",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── preprocess ────────────────────────────────────────────────────────────
    p_pre = sub.add_parser(
        "preprocess", help="Build HDF5 training data from TSS/FASTA/BigWig"
    )
    p_pre.add_argument(
        "--hdf5_folder", required=True, help="Output folder for preprocessed HDF5 chunks"
    )
    p_pre.add_argument(
        "--tss_file",
        required=True,
        help="TSV of TSS positions to center training windows on",
    )
    p_pre.add_argument(
        "--fasta_file", required=True, help="Reference genome FASTA (indexed with pyfaidx)"
    )
    p_pre.add_argument(
        "--bigwig_files",
        required=True,
        help="TSV mapping track names to BigWig file paths",
    )
    p_pre.add_argument("--chrom", required=True, help="Chromosome to preprocess (e.g. chr1)")
    p_pre.add_argument(
        "--input_length", type=int, required=True, help="Input sequence length in bp"
    )
    p_pre.add_argument(
        "--output_length", type=int, required=True, help="Output track length in bp"
    )
    p_pre.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Number of samples per output HDF5 file (default: %(default)s)",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train PromoterAI (torchrun-compatible for multi-GPU)",
    )
    p_train.add_argument(
        "--checkpoint_folder",
        required=True,
        help="Directory to write best_model.pt/latest_model.pt and logs.csv to",
    )
    p_train.add_argument(
        "--hdf5_human_folder",
        required=True,
        help="Preprocessed human HDF5 folder (chr1-20 train, chr21-22 val)",
    )
    p_train.add_argument(
        "--hdf5_nonhuman_folders",
        nargs="+",
        default=[],
        help="Additional per-species preprocessed HDF5 folders (all chroms used for training)",
    )
    p_train.add_argument(
        "--input_length",
        type=int,
        required=True,
        help="Input sequence length in bp (must match preprocessed HDF5 chunks)",
    )
    p_train.add_argument(
        "--output_length",
        type=int,
        required=True,
        help="Output track length in bp (must match preprocessed HDF5 chunks)",
    )
    p_train.add_argument(
        "--num_blocks", type=int, required=True, help="Number of MetaFormer blocks (model depth)"
    )
    p_train.add_argument(
        "--model_dim", type=int, required=True, help="Channel width of the MetaFormer backbone"
    )
    p_train.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Global training batch size; divided evenly across DDP ranks",
    )
    p_train.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Peak learning rate (default: %(default)s)",
    )
    p_train.add_argument(
        "--weight_decay",
        type=float,
        default=5e-6,
        help="Peak weight decay (default: %(default)s)",
    )
    p_train.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for (default: %(default)s)"
    )
    p_train.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: %(default)s)",
    )
    p_train.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched per DataLoader worker (default: %(default)s)",
    )
    p_train.add_argument(
        "--profile_batches",
        type=int,
        default=0,
        help="Batches to profile after warmup, then exit before validation; 0 disables profiling",
    )
    p_train.add_argument(
        "--profile_warmup_batches",
        type=int,
        default=10,
        help="Unprofiled warmup batches before the profiling window starts (default: %(default)s)",
    )
    p_train.add_argument(
        "--no_sync_batchnorm",
        action="store_true",
        default=False,
        help="Disable SyncBatchNorm conversion in multi-GPU runs",
    )
    p_train.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the model (adds startup warmup, speeds up steady-state throughput)",
    )
    p_train.add_argument(
        "--amp_dtype",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="Mixed-precision dtype for autocast; 'none' trains in full precision",
    )
    p_train.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Explicit checkpoint path to resume from (overrides --auto_resume)",
    )
    p_train.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint_folder/latest_model.pt when it exists",
    )
    p_train.add_argument(
        "--no_progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars (recommended for non-interactive logs)",
    )
    p_train.add_argument(
        "--log_every_batches",
        type=int,
        default=0,
        help="Print a batch-loss line every N batches; 0 disables",
    )
    p_train.add_argument(
        "--wandb_log_every_batches",
        type=int,
        default=0,
        help="Log batch-level metrics to W&B every N batches; 0 reuses --log_every_batches "
        "(epoch metrics are always logged when W&B is enabled)",
    )
    add_wandb_args(p_train)

    # ── finetune ──────────────────────────────────────────────────────────────
    p_ft = sub.add_parser("finetune", help="Fine-tune on GTEx rare variant outliers")
    p_ft.add_argument(
        "--model_checkpoint",
        required=True,
        help="Base PyTorch checkpoint (.pt) to fine-tune",
    )
    p_ft.add_argument(
        "--var_file",
        required=True,
        help="TSV of GTEx-outlier-format variants to fine-tune on",
    )
    p_ft.add_argument(
        "--fasta_file", required=True, help="Reference genome FASTA (indexed with pyfaidx)"
    )
    p_ft.add_argument(
        "--input_length",
        type=int,
        required=True,
        help="Input sequence length in bp (must match the base checkpoint)",
    )
    p_ft.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Global batch size; divided evenly across torchrun ranks",
    )
    p_ft.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Peak learning rate (default: %(default)s)",
    )
    p_ft.add_argument(
        "--weight_decay",
        type=float,
        default=5e-6,
        help="Peak weight decay (default: %(default)s)",
    )
    p_ft.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total target epoch count (default: %(default)s)",
    )
    p_ft.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: %(default)s)",
    )
    p_ft.add_argument(
        "--amp_dtype",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="Mixed-precision dtype for autocast; 'none' trains in full precision",
    )
    p_ft.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the twin model (adds startup warmup, speeds up steady-state throughput)",
    )
    p_ft.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Explicit checkpoint path to resume from (overrides --auto_resume)",
    )
    p_ft.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Resume from the finetune output folder's latest_model.pt when present",
    )
    add_wandb_args(p_ft)

    # ── convert ───────────────────────────────────────────────────────────────
    p_conv = sub.add_parser(
        "convert", help="Convert a pretrained Keras PromoterAI model to PyTorch"
    )
    p_conv.add_argument(
        "--keras_model",
        required=True,
        metavar="PATH",
        help="Path to the Keras SavedModel directory",
    )
    p_conv.add_argument(
        "--output", required=True, metavar="PATH", help="Output .pt checkpoint path"
    )
    p_conv.add_argument(
        "--input_length",
        type=int,
        default=None,
        help="Input sequence length (metadata only, optional)",
    )
    p_conv.add_argument(
        "--output_length",
        type=int,
        default=None,
        help="Output sequence length (metadata only, optional)",
    )

    # ── export-inference ─────────────────────────────────────────────────────
    p_export = sub.add_parser(
        "export-inference",
        help="Strip training state from a PyTorch checkpoint",
    )
    p_export.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Input training checkpoint",
    )
    p_export.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output inference-only .pt checkpoint",
    )

    # ── score ─────────────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Score variants and write output TSV")
    p_score.add_argument(
        "--model_checkpoint",
        required=True,
        help="Path to a trained/converted PyTorch checkpoint (.pt)",
    )
    p_score.add_argument(
        "--var_file",
        required=True,
        help="TSV of variants to score, with chrom/pos/ref/alt/strand columns",
    )
    p_score.add_argument(
        "--fasta_file", required=True, help="Reference genome FASTA (indexed with pyfaidx)"
    )
    p_score.add_argument(
        "--input_length",
        type=int,
        required=True,
        help="Input sequence length in bp (must match the model checkpoint)",
    )
    p_score.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of variants scored per forward pass (default: %(default)s)",
    )
    p_score.add_argument(
        "--device",
        default=None,
        help="Device to score on (default: cuda if available, else cpu)",
    )
    p_score.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes for sequence extraction (default: %(default)s)",
    )
    p_score.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output path (default: <var_stem>.<model_stem><ext>)",
    )
    p_score.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the model; pays off on large variant files, not small ones",
    )
    p_score.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Show a tqdm progress bar"
    )

    # ── check-hdf5 ───────────────────────────────────────────────────────────
    p_check = sub.add_parser(
        "check-hdf5", help="Check HDF5 training chunks for corruption"
    )
    p_check.add_argument(
        "--paths", nargs="+", required=True, help="HDF5 files/folders to check"
    )
    p_check.add_argument(
        "--full-read",
        action="store_true",
        default=False,
        help="Read every x/y value instead of only first and last rows",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        import os

        from promoterai_torch.preprocess import preprocess_chrom

        os.makedirs(args.hdf5_folder, exist_ok=True)
        preprocess_chrom(
            tss_file=args.tss_file,
            fasta_file=args.fasta_file,
            bigwig_files_tsv=args.bigwig_files,
            chrom=args.chrom,
            hdf5_folder=args.hdf5_folder,
            input_length=args.input_length,
            output_length=args.output_length,
            chunk_size=args.chunk_size,
        )

    elif args.command == "train":
        from promoterai_torch.train import main as _main

        # Reconstruct sys.argv so train.main() parses cleanly via its own parser
        _run_submodule_main(_main, args)

    elif args.command == "finetune":
        from promoterai_torch.finetune import main as _main

        _run_submodule_main(_main, args)

    elif args.command == "convert":
        from promoterai_torch.utils import convert_tf_weights

        convert_tf_weights(
            keras_model_path=args.keras_model,
            output_pt_path=args.output,
            input_length=args.input_length,
            output_length=args.output_length,
        )

    elif args.command == "export-inference":
        from promoterai_torch.utils import export_inference_checkpoint

        export_inference_checkpoint(args.checkpoint, args.output)

    elif args.command == "score":
        from promoterai_torch.score import main as _main

        _run_submodule_main(_main, args)

    elif args.command == "check-hdf5":
        from promoterai_torch.check_hdf5 import main as _main

        sys.exit(_main(args_to_argv(args)))


def _run_submodule_main(main_fn, args):
    """Patch sys.argv from parsed args so submodule main() functions re-parse cleanly."""
    main_fn.__globals__["_cli_args"] = args
    d = vars(args).copy()
    d.pop("command", None)
    argv = []
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, list):
            if v:
                argv += [f"--{k}"] + [str(i) for i in v]
        elif isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        else:
            argv += [f"--{k}", str(v)]
    sys.argv = [sys.argv[0]] + argv
    main_fn()


def args_to_argv(args):
    """Convert parsed args back to CLI-style argv for small utility subcommands."""
    d = vars(args).copy()
    d.pop("command", None)
    argv = []
    for k, v in d.items():
        flag = f"--{k.replace('_', '-')}"
        if v is None:
            continue
        if isinstance(v, list):
            if v:
                argv += [flag] + [str(i) for i in v]
        elif isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return argv
