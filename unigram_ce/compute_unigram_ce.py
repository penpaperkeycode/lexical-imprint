#!/usr/bin/env python3
"""Word-level Unigram Cross-Entropy (UCE) between a pretraining corpus and a benchmark.

Implements the metric used in §3 of the paper:

    H(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) * log P_pre(w)

where:
  - V_eval is the unigram vocabulary of the benchmark text (target distribution Q)
  - P_eval(w) is the empirical unigram probability of w in benchmark text
  - P_pre(w) is the empirical unigram probability of w in the pretraining corpus,
    with Laplace smoothing: P_pre(w) = max(c(w), 1) / (N + B), where
      c(w) = count of w in pretraining corpus
      N    = total tokens in pretraining corpus
      B    = number of benchmark words unseen in pretraining corpus
              (i.e., size of the smoothed vocabulary extension)

Notes:
  - Tokenization: NFKC normalization, punctuation stripping (non-word characters
    replaced by spaces) and whitespace splitting, without lowercasing, as in the
    paper. Identical to the filtering pipeline (`ce_filter.py`). Pass `--lowercase`
    to lowercase words first (a preprocessing ablation in the paper's appendix).
  - Smoothing avoids log(0) for OOV words; equivalent to add-1 only on the
    benchmark vocabulary (not the full pretraining vocab).
  - Unit: nats by default, optional bits via `--bits`.

Example:
    python compute_unigram_ce.py \\
        --bench_path /data/benchmarks/arc_easy/test.parquet \\
        --train_path /data/cc_dump_2026_04/ \\
        --train_format parquet \\
        --bench_col text --train_col text \\
        --n_workers 64
    > UCE = 8.7421 nats   (12.6171 bits)

Streaming I/O: pyarrow datasets are scanned in batches; counts are accumulated
in shared `collections.Counter` (single process for benchmark, parallel for
pretraining corpus). Memory is O(|V_train|).
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import Iterable, List, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# Tokenization (identical to CE filter)
# ============================================================
def normalize_text(text: str, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    if not text:
        return []
    return normalize_text(text, lowercase).split()


# ============================================================
# Benchmark vocabulary + Q distribution
# ============================================================
def build_benchmark_q(bench_path: str, text_col: str = "text",
                      lowercase: bool = False) -> tuple[dict, set]:
    """Stream all benchmark text, compute unigram probability distribution Q.

    Returns:
        q: dict[str, float] — unigram probabilities (sum to 1.0).
        vocab: set[str] — vocabulary V_eval (used to filter pretraining-corpus
               counts for memory efficiency).
    """
    counts: Counter = Counter()
    total = 0
    if os.path.isdir(bench_path):
        scan = ds.dataset(bench_path, format="parquet").to_batches(columns=[text_col])
    else:
        scan = pq.ParquetFile(bench_path).iter_batches(columns=[text_col])
    for batch in scan:
        for v in batch.column(text_col).to_pylist():
            if not v:
                continue
            toks = tokenize(v, lowercase)
            counts.update(toks)
            total += len(toks)
    if total == 0:
        raise ValueError(f"No tokens in benchmark at {bench_path}")
    q = {w: c / total for w, c in counts.items()}
    return q, set(counts.keys())


# ============================================================
# Pretraining-corpus counts (parallel)
# ============================================================
def _scan_one_fragment(args) -> tuple[Counter, int]:
    """Worker: scan one parquet fragment, return (counts, total_tokens) restricted to vocab."""
    fragment_path, text_col, vocab, lowercase = args
    counts: Counter = Counter()
    total = 0
    pf = pq.ParquetFile(fragment_path)
    for batch in pf.iter_batches(columns=[text_col]):
        for v in batch.column(text_col).to_pylist():
            if not v:
                continue
            toks = tokenize(v, lowercase)
            total += len(toks)
            for t in toks:
                if t in vocab:
                    counts[t] += 1
    return counts, total


def build_train_counts_parquet(
    train_path: str,
    text_col: str,
    vocab: set,
    n_workers: int,
    lowercase: bool = False,
) -> tuple[Counter, int]:
    """Scan pretraining-corpus parquet (file or directory) in parallel.

    Only vocabulary words are accumulated (memory ~ |V_eval|, not |V_train|).
    """
    if os.path.isdir(train_path):
        files = sorted(
            os.path.join(d, f)
            for d, _, fs in os.walk(train_path)
            for f in fs
            if f.endswith(".parquet")
        )
    else:
        files = [train_path]
    if not files:
        raise FileNotFoundError(f"No parquet files under {train_path}")

    print(f"[scan] {len(files):,} parquet files, {n_workers} workers", file=sys.stderr)
    args = [(f, text_col, vocab, lowercase) for f in files]
    total_counts: Counter = Counter()
    total_tokens = 0
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for i, (c, n) in enumerate(pool.imap_unordered(_scan_one_fragment, args, chunksize=1), 1):
            total_counts.update(c)
            total_tokens += n
            if i % max(1, len(files) // 20) == 0 or i == len(files):
                rate = i / (time.time() - t0) if time.time() > t0 else 0.0
                print(f"[scan] {i}/{len(files)} files | tokens={total_tokens:,} | "
                      f"{rate:.1f} files/s", file=sys.stderr)
    return total_counts, total_tokens


# ============================================================
# UCE = -sum_{w in V_eval} q(w) log p_pre(w)
# ============================================================
def compute_uce_nats(q: dict, counts: Counter, total_tokens: int) -> float:
    B = sum(1 for w in q if counts.get(w, 0) <= 0)
    D = total_tokens + B
    if D <= 0:
        raise ValueError(f"Invalid D={D}")
    H = 0.0
    for w, qw in q.items():
        pw = max(counts.get(w, 0), 1) / D
        H -= qw * math.log(pw)
    return H


def nats_to_bits(H: float) -> float:
    return H / math.log(2)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Word-level Unigram Cross-Entropy (UCE)")
    ap.add_argument("--bench_path", required=True,
                    help="Benchmark parquet file or directory of parquet files")
    ap.add_argument("--bench_col", default="text", help="Text column in benchmark")
    ap.add_argument("--train_path", required=True,
                    help="Pretraining-corpus parquet file or directory")
    ap.add_argument("--train_col", default="text", help="Text column in pretraining corpus")
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count(),
                    help="Number of parallel workers for pretraining-corpus scan")
    ap.add_argument("--bits", action="store_true",
                    help="Print UCE in bits in addition to nats")
    ap.add_argument("--lowercase", action="store_true",
                    help="Lowercase words before counting (off by default, as in the paper)")
    args = ap.parse_args()

    print(f"[UCE] benchmark: {args.bench_path}", file=sys.stderr)
    q, vocab = build_benchmark_q(args.bench_path, text_col=args.bench_col,
                                 lowercase=args.lowercase)
    print(f"[UCE] |V_eval| = {len(vocab):,}, sum(q)={sum(q.values()):.4f}", file=sys.stderr)

    print(f"[UCE] pretraining corpus: {args.train_path}", file=sys.stderr)
    counts, total_tokens = build_train_counts_parquet(
        args.train_path,
        text_col=args.train_col,
        vocab=vocab,
        n_workers=args.n_workers,
        lowercase=args.lowercase,
    )
    print(f"[UCE] |V_train ∩ V_eval| = {len(counts):,}, "
          f"total tokens={total_tokens:,}", file=sys.stderr)

    H_nats = compute_uce_nats(q, counts, total_tokens)
    print(f"\nUCE = {H_nats:.4f} nats", end="")
    if args.bits:
        print(f"   ({nats_to_bits(H_nats):.4f} bits)", end="")
    print()


if __name__ == "__main__":
    main()
