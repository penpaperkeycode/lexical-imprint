#!/usr/bin/env python3
"""Split a source corpus directory's parquet files into N round-robin shard lists.

Used by Phase B and Phase D of the multi-pod DSIR pipeline.

Output:
    <out_prefix>_0.txt  <out_prefix>_1.txt  ...  <out_prefix>_{N-1}.txt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsir.parquet_io import list_parquet_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--num_splits", type=int, required=True)
    ap.add_argument("--out_prefix", required=True, help="e.g. /tmp/fwedu_shard_list")
    args = ap.parse_args()

    files = list_parquet_files(args.source_path)
    print(f"[split] {len(files)} parquet files at {args.source_path}", flush=True)

    splits: list[list[str]] = [[] for _ in range(args.num_splits)]
    for i, fp in enumerate(files):
        splits[i % args.num_splits].append(fp)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for s, group in enumerate(splits):
        out = f"{args.out_prefix}_{s}.txt"
        with open(out, "w") as f:
            for fp in group:
                f.write(fp + "\n")
        print(f"[split] {out}: {len(group)} files", flush=True)


if __name__ == "__main__":
    main()
