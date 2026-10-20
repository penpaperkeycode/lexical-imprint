#!/usr/bin/env python3
"""DSIR Phase B — source corpus bucket counts (one pod, ~1/4 of source shards).

CLI:
    --shard_list <txt>   : line-separated parquet file paths to process
    --bucket_count, --n_max, --text_col
    --out <partial.npy>  : output partial bucket count vector
    --n_workers          : per-pod multiprocess.Pool size (default cpu_count)

Multi-pod operation:
    Pre-split with split_source_shards.py into 4 ways; each pod processes only its own shard list.
    The 4 partial .npy files are summed in Phase C.
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
from dsir.parquet_io import scan_text_column


def _count_one_file(args):
    fp, text_col, B, n_max = args
    try:
        return count_buckets(
            scan_text_column(fp, text_col=text_col),
            bucket_count=B,
            n_max=n_max,
        )
    except Exception as e:
        print(f"[ERR] {fp}: {e}", flush=True)
        return np.zeros(B, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_list", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--bucket_count", type=int, default=10_000)
    ap.add_argument("--n_max", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    with open(args.shard_list) as f:
        files = [ln.strip() for ln in f if ln.strip()]
    print(f"[Phase B] pod processes {len(files)} parquet files (workers={args.n_workers})", flush=True)
    t0 = time.time()
    with mp.Pool(args.n_workers) as pool:
        partials = pool.imap_unordered(
            _count_one_file,
            [(fp, args.text_col, args.bucket_count, args.n_max) for fp in files],
            chunksize=1,
        )
        counts = np.zeros(args.bucket_count, dtype=np.int64)
        for i, p in enumerate(partials, 1):
            counts += p
            if i % 16 == 0 or i == len(files):
                print(
                    f"[Phase B] {i}/{len(files)} files done "
                    f"sum={int(counts.sum()):,} wall={time.time() - t0:.0f}s",
                    flush=True,
                )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, counts)
    print(f"[Phase B] -> {args.out} (wall={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
