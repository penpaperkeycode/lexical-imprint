#!/usr/bin/env python3
"""DSIR Phase C — merge partial source counts + compute log P_t / P_s.

Inputs:
    --target_counts target.npy
    --source_partials part0.npy part1.npy part2.npy part3.npy
    --smoothing 1.0
    --out log_ratio.npy

Wall: <1s.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsir.scoring import compute_log_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_counts", required=True)
    ap.add_argument("--source_partials", nargs="+", required=True)
    ap.add_argument("--smoothing", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target = np.load(args.target_counts)
    source = np.zeros_like(target, dtype=np.int64)
    for p in args.source_partials:
        partial = np.load(p)
        if partial.shape != target.shape:
            raise ValueError(f"shape mismatch on {p}: {partial.shape} vs {target.shape}")
        source += partial.astype(np.int64)

    log_ratio = compute_log_ratio(target, source, smoothing=args.smoothing)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, log_ratio)
    print(
        f"[Phase C] target_sum={int(target.sum()):,} source_sum={int(source.sum()):,} "
        f"log_ratio: min={log_ratio.min():.3f} max={log_ratio.max():.3f} "
        f"mean={log_ratio.mean():.3f} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
