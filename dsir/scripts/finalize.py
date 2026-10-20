#!/usr/bin/env python3
"""DSIR Phase E — merge 4 local top-K + write filtered parquet.

Steps:
    1. Load 4 local_topk parquets + their .file_map.txt sidecars.
    2. Build a unified file_id → file_path lookup, merge to global top-K (heap merge).
    3. Group selected (file_id, row_idx) by source file → re-stream that file → emit
       only the selected rows into output parquet shards.

Parallelized per source file via mp.Pool (each file's selected_rows is small, so this is fast).
"""
from __future__ import annotations

import argparse
import heapq
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_local_topk(parquet_path: str) -> List[Tuple[float, int, int]]:
    t = pq.read_table(parquet_path)
    return list(zip(t["logiw"].to_pylist(), t["file_id"].to_pylist(), t["row_idx"].to_pylist()))


def load_file_map(map_path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(map_path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            fid, fp = ln.split("\t", 1)
            out[int(fid)] = fp
    return out


def _emit_one_file(args):
    """Stream a source parquet, keep only selected row indices, return as table dict."""
    fp, selected_rows, text_col = args
    selected_rows = sorted(selected_rows)
    selected_set = set(selected_rows)
    keep_text = []
    keep_idx = []
    seen = 0
    pf = pq.ParquetFile(fp)
    for batch in pf.iter_batches(batch_size=8192, columns=[text_col]):
        col = batch.column(text_col)
        if col.type == pa.string():
            col = col.cast(pa.large_string())
        vals = col.to_pylist()
        for v in vals:
            if seen in selected_set and v is not None:
                keep_text.append(v)
                keep_idx.append(seen)
            seen += 1
            if len(keep_text) >= len(selected_set):
                break
        if len(keep_text) >= len(selected_set):
            break
    return fp, keep_text, keep_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_topk", nargs="+", required=True,
                    help="4 local_topk parquet paths (each has .file_map.txt sidecar)")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out_dir", required=True, help="output dir for filtered parquet shards")
    ap.add_argument("--out_shards", type=int, default=16, help="# of output parquet files")
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    t0 = time.time()
    # 1. Merge the 4 local heaps -> global top-K
    all_items: List[Tuple[float, int, int]] = []
    file_map: Dict[int, str] = {}
    for lp in args.local_topk:
        all_items.extend(load_local_topk(lp))
        file_map.update(load_file_map(lp + ".file_map.txt"))
    print(f"[Phase E] merging {len(args.local_topk)} local heaps, total candidates={len(all_items):,}", flush=True)
    # heapq.nlargest is more memory-efficient than a partial sort here.
    global_top = heapq.nlargest(args.K, all_items, key=lambda x: x[0])
    print(
        f"[Phase E] global top-K size={len(global_top):,} "
        f"score range=[{global_top[-1][0]:.3f}, {global_top[0][0]:.3f}]",
        flush=True,
    )

    # 2. Group row_idx by file_id
    by_file: Dict[int, List[int]] = defaultdict(list)
    for _, fid, ri in global_top:
        by_file[fid].append(ri)
    print(f"[Phase E] selected rows span {len(by_file)} source files", flush=True)

    # 3. Emit per file in parallel
    os.makedirs(args.out_dir, exist_ok=True)
    job_args = [(file_map[fid], rows, args.text_col) for fid, rows in by_file.items()]
    all_text: List[str] = []
    with mp.Pool(args.n_workers) as pool:
        done = 0
        for fp, keep_text, _keep_idx in pool.imap_unordered(_emit_one_file, job_args, chunksize=1):
            all_text.extend(keep_text)
            done += 1
            if done % 32 == 0 or done == len(job_args):
                print(
                    f"[Phase E] emitted {done}/{len(job_args)} source files "
                    f"total_rows={len(all_text):,} wall={time.time() - t0:.0f}s",
                    flush=True,
                )

    # 4. Write output parquet per shard (large_string)
    shard_size = (len(all_text) + args.out_shards - 1) // args.out_shards
    os.makedirs(args.out_dir, exist_ok=True)
    for s in range(args.out_shards):
        start = s * shard_size
        end = min(start + shard_size, len(all_text))
        if start >= end:
            break
        chunk = all_text[start:end]
        col = pa.array(chunk, type=pa.large_string())
        tbl = pa.table({"text": col})
        out = os.path.join(args.out_dir, f"part-{s:05d}.parquet")
        pq.write_table(tbl, out, compression="snappy")
        print(f"[Phase E] shard {s+1}/{args.out_shards}: {len(chunk):,} rows -> {out}", flush=True)

    print(
        f"[Phase E] DONE  total_rows={len(all_text):,} target_K={args.K:,} "
        f"({len(all_text) / args.K * 100:.2f}% of K) wall={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
