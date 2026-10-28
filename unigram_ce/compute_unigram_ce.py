#!/usr/bin/env python3
"""Word-level Unigram Cross-Entropy (UCE) between a pretraining corpus and a benchmark.

Implements the metric used in §3 of the paper:

    H(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) * log P_pre(w)

where:
  - V_eval is the unigram vocabulary of the benchmark text (target distribution Q)
  - P_eval(w) is the empirical unigram probability of w in benchmark text
  - P_pre(w) is the unigram probability of w in the pretraining corpus with the
    zero-count floor of appendix A.1: P_pre(w) = max(c(w), 1) / (T + B), where
      c(w) = count of w in pretraining corpus
      T    = total word tokens in pretraining corpus (all words, not only V_eval)
      B    = number of benchmark words with c(w) = 0

Notes:
  - Tokenization: NFKC normalization, punctuation stripping (non-word characters
    replaced by spaces) and whitespace splitting, without lowercasing, as in the
    paper. Identical to the filtering pipeline (`ce_filter.py`). Pass `--lowercase`
    to lowercase words first (a preprocessing ablation in the paper's appendix).
  - Smoothing: only zero counts are raised to one, which avoids log(0). Positive
    counts are unchanged, so this is not Laplace (add-1) smoothing.
  - Benchmark directory (for example the pooled CSR7 group
    /data/csr7/benchmark_group_en/): the text of all its parquet files is pooled
    into one Q, so each file is weighted by its number of word tokens. This is
    not the same as averaging per-file UCE values.
  - Unit: nats. `--bits` also prints the value in bits, the unit of the paper.

Example, with the benchmark files written by
    python benchmarks/preprocess_csr7.py \\
        --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag \\
        --output_dir /data/csr7/
(one file per task, /data/csr7/<task>/<task>.parquet, and the pooled group
/data/csr7/benchmark_group_en/) and a pretraining corpus at /data/cc_dump/:
    python unigram_ce/compute_unigram_ce.py \\
        --bench_path /data/csr7/arc_easy/arc_easy.parquet \\
        --train_path /data/cc_dump/ \\
        --bench_col text --train_col text \\
        --n_workers 64 --bits
    > UCE = 8.7421 nats   (12.6122 bits)

Input files: --bench_path and --train_path take a parquet file or a directory.
A directory is searched recursively with pyarrow dataset discovery, the same rule
as `ce_filter.py`: names starting with '.' or '_' are skipped and every other
file must be parquet. For a Hugging Face download, pass its data directory, not
the repository root, whose README.md is not parquet. Symlinks are followed. A
missing path, a broken symlink, a file reached through more than one path (a
symlink loop), a file without the text column, an unreadable file, or a corpus
without word tokens stops the run with an error naming the path.

Streaming I/O: files are read in batches. The benchmark is counted in one
process; each corpus file is counted by one worker and the counts are summed.
--n_workers defaults to the CPU count, and at most one worker per corpus file is
started. Only benchmark words are counted in the corpus, so memory is O(|V_eval|).
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
# Input files (same discovery rule as CE filter)
# ============================================================
def _check_listing(path: str, files: list[str]):
    """Refuse inputs that pyarrow discovery reads wrongly without an error.

    A broken symlink is left out of the listing, and a symlinked directory that
    points back up the tree (a loop) lists the same files many times.
    """
    if os.path.isdir(path):
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
            for name in names:
                p = os.path.join(root, name)
                if not name.startswith((".", "_")) and not os.path.exists(p):
                    raise FileNotFoundError(f"broken symlink: {p}")
    if len({os.path.realpath(f) for f in files}) != len(files):
        raise ValueError(f"{path}: the same file is reached through more than one path "
                         f"(symlink loop?)")


def list_parquet_files(path: str) -> list[str]:
    """Return the parquet files at `path`: the file itself, or all files under a directory.

    Directories use pyarrow dataset discovery, as `ce_filter.py` does, so both tools
    read the same files: subdirectories are searched recursively, names starting
    with '.' or '_' are skipped (hidden files, Spark _SUCCESS/_temporary), and
    every other file must be parquet. Symlinks are followed; a broken symlink or a
    file listed under two paths (symlink loop) is an error (`_check_listing`).
    """
    try:
        files = sorted(ds.dataset(path, format="parquet").files)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {path}") from None
    _check_listing(path, files)
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    return files


def _open_parquet(path: str, text_col: str) -> pq.ParquetFile:
    """Open one parquet file and check that it has `text_col`; errors name the file."""
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        raise ValueError(f"{path}: {type(e).__name__}: {e}") from e
    names = pf.schema_arrow.names
    if text_col not in names:
        raise ValueError(f"{path}: no column '{text_col}' (columns: {', '.join(names)})")
    return pf


# ============================================================
# Benchmark vocabulary + Q distribution
# ============================================================
def build_benchmark_q(bench_path: str, text_col: str = "text",
                      lowercase: bool = False) -> tuple[dict, set]:
    """Stream all benchmark text, compute unigram probability distribution Q.

    A directory is pooled: the rows of all its parquet files form one Q, so each
    file is weighted by its number of word tokens.

    Returns:
        q: dict[str, float] — unigram probabilities (sum to 1.0).
        vocab: set[str] — vocabulary V_eval (used to filter pretraining-corpus
               counts for memory efficiency).
    """
    files = list_parquet_files(bench_path)
    counts: Counter = Counter()
    total = 0
    for path in files:
        pf = _open_parquet(path, text_col)
        try:
            for batch in pf.iter_batches(columns=[text_col]):
                for v in batch.column(text_col).to_pylist():
                    if not v:
                        continue
                    toks = tokenize(v, lowercase)
                    counts.update(toks)
                    total += len(toks)
        except Exception as e:
            raise ValueError(f"{path}: {type(e).__name__}: {e}") from e
    print(f"[bench] {len(files):,} parquet files, {total:,} tokens", file=sys.stderr)
    if total == 0:
        raise ValueError(f"No tokens in benchmark at {bench_path}")
    q = {w: c / total for w, c in counts.items()}
    return q, set(counts.keys())


# ============================================================
# Pretraining-corpus counts (parallel)
# ============================================================
def _scan_one_fragment(args) -> tuple[Counter, int]:
    """Worker: scan one parquet fragment, return (counts, total_tokens) restricted to vocab.

    Errors are raised with the file name (not swallowed); the pool re-raises them
    in the parent, which stops the run.
    """
    fragment_path, text_col, vocab, lowercase = args
    counts: Counter = Counter()
    total = 0
    pf = _open_parquet(fragment_path, text_col)
    try:
        for batch in pf.iter_batches(columns=[text_col]):
            for v in batch.column(text_col).to_pylist():
                if not v:
                    continue
                toks = tokenize(v, lowercase)
                total += len(toks)
                for t in toks:
                    if t in vocab:
                        counts[t] += 1
    except Exception as e:
        raise ValueError(f"{fragment_path}: {type(e).__name__}: {e}") from e
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
    Files are listed by `list_parquet_files` (same rule as the benchmark side).
    Each file is read by one worker, so at most one worker per file is started.
    """
    files = list_parquet_files(train_path)
    n_workers = max(1, min(n_workers, len(files)))

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
    if total_tokens == 0:
        # UCE would silently come out as ln|V_eval| (every word floored to count 1)
        raise ValueError(f"No word tokens in column '{text_col}' of pretraining corpus "
                         f"at {train_path}")
    return total_counts, total_tokens


# ============================================================
# UCE = -sum_{w in V_eval} q(w) log p_pre(w)
# ============================================================
def compute_uce_nats(q: dict, counts: Counter, total_tokens: int) -> float:
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
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
                    help="Benchmark parquet file, or a directory whose parquet files are pooled "
                         "into one token-weighted Q (do not average per-file UCE values instead)")
    ap.add_argument("--bench_col", default="text", help="Text column in benchmark")
    ap.add_argument("--train_path", required=True,
                    help="Pretraining-corpus parquet file or directory. Directories (here and in "
                         "--bench_path) are searched recursively; names starting with '.' or '_' "
                         "are skipped")
    ap.add_argument("--train_col", default="text", help="Text column in pretraining corpus")
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count(),
                    help="Number of parallel workers for pretraining-corpus scan "
                         "(default: the CPU count; at most one worker per corpus file is started)")
    ap.add_argument("--bits", action="store_true",
                    help="Also print UCE in bits, the unit of the paper "
                         "(without --bits only nats are printed)")
    ap.add_argument("--lowercase", action="store_true",
                    help="Lowercase words before counting (off by default, as in the paper)")
    args = ap.parse_args()
    if args.n_workers < 1:
        ap.error("--n_workers must be at least 1")

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
    try:
        main()
    except (OSError, ValueError) as e:
        # Unusable input (missing/unreadable file, missing text column, no tokens):
        # one line naming the path, non-zero exit.
        msg = str(e).strip().replace("\n", " ")
        sys.exit(f"[UCE] error: {msg}")
