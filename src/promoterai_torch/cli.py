"""
Unified CLI entry point.

Usage:
    promoterai-torch preprocess  [args]
    promoterai-torch train       [args]
    promoterai-torch finetune    [args]
    promoterai-torch score       [args]
"""

import argparse
import sys


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
    p_pre.add_argument("--hdf5_folder", required=True)
    p_pre.add_argument("--tss_file", required=True)
    p_pre.add_argument("--fasta_file", required=True)
    p_pre.add_argument("--bigwig_files", required=True)
    p_pre.add_argument("--chrom", required=True)
    p_pre.add_argument("--input_length", type=int, required=True)
    p_pre.add_argument("--output_length", type=int, required=True)
    p_pre.add_argument("--chunk_size", type=int, default=256)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train PromoterAI (torchrun-compatible for multi-GPU)",
    )
    p_train.add_argument("--checkpoint_folder", required=True)
    p_train.add_argument("--hdf5_human_folder", required=True)
    p_train.add_argument("--hdf5_nonhuman_folders", nargs="+", default=[])
    p_train.add_argument("--input_length", type=int, required=True)
    p_train.add_argument("--output_length", type=int, required=True)
    p_train.add_argument("--num_blocks", type=int, required=True)
    p_train.add_argument("--model_dim", type=int, required=True)
    p_train.add_argument("--batch_size", type=int, required=True)
    p_train.add_argument("--learning_rate", type=float, default=5e-4)
    p_train.add_argument("--weight_decay", type=float, default=5e-6)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--num_workers", type=int, default=4)

    # ── finetune ──────────────────────────────────────────────────────────────
    p_ft = sub.add_parser("finetune", help="Fine-tune on GTEx rare variant outliers")
    p_ft.add_argument("--model_checkpoint", required=True)
    p_ft.add_argument("--var_file", required=True)
    p_ft.add_argument("--fasta_file", required=True)
    p_ft.add_argument("--input_length", type=int, required=True)
    p_ft.add_argument("--batch_size", type=int, default=8)
    p_ft.add_argument("--learning_rate", type=float, default=5e-4)
    p_ft.add_argument("--weight_decay", type=float, default=5e-6)
    p_ft.add_argument("--epochs", type=int, default=100)
    p_ft.add_argument("--num_workers", type=int, default=4)

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

    # ── score ─────────────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Score variants and write output TSV")
    p_score.add_argument("--model_checkpoint", required=True)
    p_score.add_argument("--var_file", required=True)
    p_score.add_argument("--fasta_file", required=True)
    p_score.add_argument("--input_length", type=int, required=True)
    p_score.add_argument("--batch_size", type=int, default=8)
    p_score.add_argument("--device", default=None)
    p_score.add_argument("--num_workers", type=int, default=4)
    p_score.add_argument("--verbose", action="store_true", default=False)

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

    elif args.command == "score":
        from promoterai_torch.score import main as _main

        _run_submodule_main(_main, args)


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
