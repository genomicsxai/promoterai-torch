"""
Unified CLI entry point.

Usage:
    promoterai-torch preprocess  [args]
    promoterai-torch train       [args]
    promoterai-torch finetune    [args]
    promoterai-torch export-inference [args]
    promoterai-torch score       [args]
    promoterai-torch check-hdf5  [args]

Each subcommand's flags are defined once, in that subcommand's own module
(`build_parser()` in preprocess.py/train.py/finetune.py/score.py/check_hdf5.py),
and reused here rather than redefined — this file only builds the subparser
tree and dispatches. `convert`/`export-inference` are simple enough that they
don't have a standalone module of their own, so their flags live here.
"""

import argparse
import sys

from promoterai_torch import check_hdf5, finetune, preprocess, score, train


def main():
    """Dispatch to the appropriate subcommand (preprocess, train, finetune, convert, score)."""
    parser = argparse.ArgumentParser(
        prog="promoterai-torch",
        description="PyTorch port of PromoterAI — promoter variant effect prediction",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    preprocess.build_parser(
        sub.add_parser("preprocess", help="Build HDF5 training data from TSS/FASTA/BigWig")
    )
    train.build_parser(
        sub.add_parser("train", help="Train PromoterAI (torchrun-compatible for multi-GPU)")
    )
    finetune.build_parser(
        sub.add_parser("finetune", help="Fine-tune on GTEx rare variant outliers")
    )
    score.build_parser(sub.add_parser("score", help="Score variants and write output TSV"))
    check_hdf5.build_parser(
        sub.add_parser("check-hdf5", help="Check HDF5 training files for corruption")
    )

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

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess.main(args)
    elif args.command == "train":
        train.main(args)
    elif args.command == "finetune":
        finetune.main(args)
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
        score.main(args)
    elif args.command == "check-hdf5":
        sys.exit(check_hdf5.main(args))
