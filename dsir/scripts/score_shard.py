#!/usr/bin/env python3
"""DSIR Phase D — score rows + maintain local top-K min-heap (one pod).

Per-pod output: parquet with columns (logiw float64, file_id int32, row_idx int64).
file_id is the ordinal index of the file within shard_list (used in Phase E to recover the file path).

Multi-pod orchestration:
    split_source_shards.py splits the shard_list 4 ways and hands one slice to each pod.
    The 4 pods each maintain a K-sized heap; Phase E merges the 4 heaps into the global top-K.
"""
from __future__ import annotations

import argparse
import heapq
import multiprocessing as mp
import os
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsir.hashing import hash_ngrams
from dsir.parquet_io import scan_text_column


def _score_one_file(args):
    """Worker: stream one parquet file, return [(score, file_id, row_idx), ...] sorted desc top-k_local.

    score = sum_log_iw (default, paper's DSIR) — biased toward short docs when mean log_ratio < 0.
    score = mean_log_iw (when length_normalize=True) — removes length bias, top-K has normal length distribution.
    """
    fp, file_id, log_ratio_path, text_col, n_max, k_local, length_normalize = args
    log_ratio = np.load(log_ratio_path, mmap_mode="r")
    B = log_ratio.shape[0]
    heap: List[Tuple[float, int, int]] = []
    for row_idx, t in enumerate(scan_text_column(fp, text_col=text_col)):
        buckets = hash_ngrams(t, bucket_count=B, n_max=n_max)
        if not buckets:
            continue
        if length_normalize:
            s = float(log_ratio[buckets].mean())
        else:
            s = float(log_ratio[buckets].sum())
        item = (s, file_id, row_idx)
        if len(heap) < k_local:
            heapq.heappush(heap, item)
        elif s > heap[0][0]:
            heapq.heapreplace(heap, item)
    return heap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_list", required=True, help="this pod's slice of source files")
    ap.add_argument("--log_ratio", required=True)
    ap.add_argument("--K", type=int, required=True, help="GLOBAL K (per-pod heap also = K)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--n_max", type=int, default=2)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    ap.add_argument("--file_id_offset", type=int, default=0,
                    help="Offset to avoid file_id collisions when Phase E merges the 4 pod heaps")
    ap.add_argument("--length_normalize", action="store_true",
                    help="Score = mean log_iw per ngram (removes length bias). Default = sum (paper).")
    args = ap.parse_args()

    with open(args.shard_list) as f:
        files = [ln.strip() for ln in f if ln.strip()]
    # per-pod heap size = global K (a local heap smaller than global K risks losing candidates)
    k_local = args.K

    print(
        f"[Phase D] {len(files)} files | K={k_local:,} per-pod heap | "
        f"workers={args.n_workers} file_id_offset={args.file_id_offset} "
        f"length_normalize={args.length_normalize}",
        flush=True,
    )
    t0 = time.time()
    # Each worker returns its file's heap (trimmed to K); the main process keeps a unified K-sized heap.
    with mp.Pool(args.n_workers) as pool:
        global_heap: List[Tuple[float, int, int]] = []
        done = 0
        for partial in pool.imap_unordered(
            _score_one_file,
            [(fp, args.file_id_offset + i, args.log_ratio, args.text_col,
              args.n_max, k_local, args.length_normalize) for i, fp in enumerate(files)],
            chunksize=1,
        ):
            for item in partial:
                if len(global_heap) < k_local:
                    heapq.heappush(global_heap, item)
                elif item[0] > global_heap[0][0]:
                    heapq.heapreplace(global_heap, item)
            done += 1
            if done % 8 == 0 or done == len(files):
                heap_min = global_heap[0][0] if global_heap else float("nan")
                print(
                    f"[Phase D] {done}/{len(files)} files | heap_size={len(global_heap):,} "
                    f"min_score={heap_min:.3f} wall={time.time() - t0:.0f}s",
                    flush=True,
                )

    # Dump to parquet
    sorted_heap = sorted(global_heap, reverse=True)
    logiw = np.array([s for s, _, _ in sorted_heap], dtype=np.float64)
    file_ids = np.array([fi for _, fi, _ in sorted_heap], dtype=np.int32)
    row_idxs = np.array([ri for _, _, ri in sorted_heap], dtype=np.int64)
    table = pa.table({"logiw": logiw, "file_id": file_ids, "row_idx": row_idxs})
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    pq.write_table(table, args.out_parquet, compression="snappy")
    print(
        f"[Phase D] wrote {len(sorted_heap):,} top-K rows -> {args.out_parquet} "
        f"(wall={time.time() - t0:.1f}s)",
        flush=True,
    )

    # Persist the shard_list / file_id mapping as a sidecar (used by Phase E)
    map_path = args.out_parquet + ".file_map.txt"
    with open(map_path, "w") as f:
        for i, fp in enumerate(files):
            f.write(f"{args.file_id_offset + i}\t{fp}\n")
    print(f"[Phase D] file_map -> {map_path}", flush=True)


if __name__ == "__main__":
    main()
