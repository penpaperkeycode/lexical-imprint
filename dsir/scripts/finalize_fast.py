#!/usr/bin/env python3
"""DSIR Phase E (fast) — write filtered parquets directly from workers, no main-process serialization.

Each worker dumps its source file's selected rows directly to its own partial parquet.
The main process only collects partial parquet paths and concatenates them at the end (or leaves them as-is).

Faster than finalize.py because:
- No `extend` of huge text list in main process
- No pickling of keep_text per file through pool
- True 120-way parallelism on file reads
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


def _emit_one_file_to_disk(args):
    """Stream source parquet, keep selected row indices, dump to a partial parquet."""
    fp, selected_rows, text_col, partial_out = args
    selected_set = set(selected_rows)
    keep_text: List[str] = []
    seen = 0
    try:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=8192, columns=[text_col]):
            col = batch.column(text_col)
            if col.type == pa.string():
                col = col.cast(pa.large_string())
            vals = col.to_pylist()
            for v in vals:
                if seen in selected_set and v is not None:
                    keep_text.append(v)
                seen += 1
            if len(keep_text) >= len(selected_set):
                break
    except Exception as e:
        return fp, 0, str(e)

    if keep_text:
        col = pa.array(keep_text, type=pa.large_string())
        tbl = pa.table({"text": col})
        pq.write_table(tbl, partial_out, compression="snappy")
    return fp, len(keep_text), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_topk", nargs="+", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_shards", type=int, default=24,
                    help="# of final consolidated shards (after worker partials)")
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    ap.add_argument("--shard_index", type=int, default=-1,
                    help="If >=0, only process files where file_id %% num_shards == shard_index")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total # of finalize pods (used with --shard_index)")
    ap.add_argument("--skip_consolidate", action="store_true",
                    help="Skip step 4 consolidation (multi-pod mode handles it separately)")
    args = ap.parse_args()

    t0 = time.time()
    # 1. Merge local heaps → global top-K
    all_items: List[Tuple[float, int, int]] = []
    file_map: Dict[int, str] = {}
    for lp in args.local_topk:
        all_items.extend(load_local_topk(lp))
        file_map.update(load_file_map(lp + ".file_map.txt"))
    print(
        f"[Phase E] merged {len(args.local_topk)} local heaps, "
        f"total candidates={len(all_items):,}",
        flush=True,
    )
    global_top = heapq.nlargest(args.K, all_items, key=lambda x: x[0])
    print(
        f"[Phase E] global top-K={len(global_top):,} "
        f"score range=[{global_top[-1][0]:.3f}, {global_top[0][0]:.3f}]",
        flush=True,
    )

    # 2. group by file_id (apply shard filter if multi-pod)
    by_file: Dict[int, List[int]] = defaultdict(list)
    for _, fid, ri in global_top:
        if args.shard_index >= 0 and fid % args.num_shards != args.shard_index:
            continue
        by_file[fid].append(ri)
    if args.shard_index >= 0:
        print(
            f"[Phase E] shard {args.shard_index}/{args.num_shards}: "
            f"{len(by_file)} files (of total span)",
            flush=True,
        )
    else:
        print(f"[Phase E] {len(by_file)} source files have selected rows", flush=True)

    # 3. parallel emit, each worker writes its own partial parquet
    partial_dir = os.path.join(args.out_dir, "_partials")
    os.makedirs(partial_dir, exist_ok=True)
    job_args = [
        (file_map[fid], rows, args.text_col,
         os.path.join(partial_dir, f"part-{fid:06d}.parquet"))
        for fid, rows in by_file.items()
    ]
    print(f"[Phase E] dispatching {len(job_args)} files to {args.n_workers} workers", flush=True)

    total_rows = 0
    done = 0
    failed: List[Tuple[str, str]] = []
    with mp.Pool(args.n_workers) as pool:
        for fp, n, err in pool.imap_unordered(_emit_one_file_to_disk, job_args, chunksize=8):
            if err:
                failed.append((fp, err))
            else:
                total_rows += n
            done += 1
            if done % 200 == 0 or done == len(job_args):
                print(
                    f"[Phase E] {done}/{len(job_args)} files | "
                    f"total_rows={total_rows:,} wall={time.time()-t0:.0f}s",
                    flush=True,
                )

    if failed:
        print(f"[Phase E] WARNING — {len(failed)} files failed:", flush=True)
        for fp, err in failed[:5]:
            print(f"  {fp}: {err}", flush=True)

    if args.skip_consolidate:
        print(f"[Phase E] skip_consolidate set → exit (run consolidate_partials.py separately)", flush=True)
        return

    # 4. consolidate partials into final out_shards
    partials = sorted([f for f in os.listdir(partial_dir) if f.endswith(".parquet")])
    print(f"[Phase E] consolidating {len(partials)} partials → {args.out_shards} final shards", flush=True)
    # bucket partials round-robin
    buckets: List[List[str]] = [[] for _ in range(args.out_shards)]
    for i, p in enumerate(partials):
        buckets[i % args.out_shards].append(os.path.join(partial_dir, p))

    def _consolidate_one(args):
        s, paths, out_path = args
        if not paths:
            return s, 0
        tbls = [pq.read_table(p) for p in paths]
        merged = pa.concat_tables(tbls)
        pq.write_table(merged, out_path, compression="snappy")
        return s, merged.num_rows

    cons_args = [
        (s, paths, os.path.join(args.out_dir, f"part-{s:05d}.parquet"))
        for s, paths in enumerate(buckets)
    ]
    final_total = 0
    with mp.Pool(min(args.out_shards, args.n_workers)) as pool:
        for s, n in pool.imap_unordered(_consolidate_one, cons_args, chunksize=1):
            final_total += n
    print(
        f"[Phase E] DONE  final_rows={final_total:,} target_K={args.K:,} "
        f"({final_total / args.K * 100:.2f}% of K) wall={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
