#!/usr/bin/env python3
"""DSIR Phase A — target corpus bucket counts.

Stream benchmark parquet, hash uni+bigram, accumulate counts in a B-vector,
dump to .npy.

Memory: O(B) ~ 80 KB at B=10 000. Wall: ~5 min on CSR7 benchmark group.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsir.counting import count_buckets
from dsir.parquet_io import list_parquet_files, scan_text_column


def _count_one_file(args):
    fp, text_col, B, n_max = args
    return count_buckets(
        scan_text_column(fp, text_col=text_col),
        bucket_count=B,
        n_max=n_max,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True, help="benchmark group dir or parquet")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--bucket_count", type=int, default=10_000, help="B")
    ap.add_argument("--n_max", type=int, default=2)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    files = list_parquet_files(args.target_path)
    print(f"[Phase A] {len(files)} parquet files under {args.target_path}", flush=True)
    t0 = time.time()
    with mp.Pool(args.n_workers) as pool:
        partials = pool.map(
            _count_one_file,
            [(fp, args.text_col, args.bucket_count, args.n_max) for fp in files],
        )
    counts = np.zeros(args.bucket_count, dtype=np.int64)
    for p in partials:
        counts += p
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, counts)
    print(
        f"[Phase A] target counts sum={int(counts.sum())} "
        f"non-zero={int((counts > 0).sum())}/{args.bucket_count} "
        f"wall={time.time() - t0:.1f}s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
