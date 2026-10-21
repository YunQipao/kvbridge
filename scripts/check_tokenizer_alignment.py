#!/usr/bin/env python3
"""Check exact source/target token ID alignment on calibration text."""

import argparse
import os
import sys

from kvbridge.formal import check_tokenizer_alignment, load_calibration_texts, tokenize_records
from kvbridge.model_utils import load_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--target-tokenizer",
        help="explicit target tokenizer reference, useful when the target snapshot omits tokenizer files",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--n-calib", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    source_tok = load_tokenizer(args.source, local_files_only=args.local_files_only)
    target_tok = load_tokenizer(
        args.target_tokenizer or args.target,
        local_files_only=args.local_files_only,
    )
    records = load_calibration_texts(
        dataset=args.dataset, dataset_file=args.dataset_file, n_calib=args.n_calib,
        seed=args.seed, tokenizer=source_tok, min_tokens=args.seq_len)
    source_rows = tokenize_records(records, source_tok, args.seq_len)
    target_rows = tokenize_records(records, target_tok, args.seq_len)
    check_tokenizer_alignment(source_rows, target_rows)
    print(f"PASS aligned_rows={len(source_rows)}", flush=True)
    # Older datasets/fsspec combinations can leave a streaming worker that
    # crashes during CPython finalization after a successful check.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
