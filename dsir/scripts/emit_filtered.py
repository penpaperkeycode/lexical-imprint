#!/usr/bin/env python3
"""Phase E.2 (multi-pod) — emit selected text rows from source parquets.

Reads global_top.parquet (file_id, row_idx) + file_map.txt sidecar (any of the
4 local_topks works since they share the same file_id space).

Each pod handles file_id %% num_shards == shard_index. Each worker writes
per-file partial parquet directly to disk.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    ap.add_argument("--global_top", required=True, help="merge_global_topk.py output")
    ap.add_argument("--file_map", nargs="+", required=True,
                    help="one or more local_topk_*.parquet.file_map.txt sidecars")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--partial_dir", required=True)
    ap.add_argument("--shard_index", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    t0 = time.time()
    # Load global_top
    gt = pq.read_table(args.global_top)
    fids = gt["file_id"].to_pylist()
    rids = gt["row_idx"].to_pylist()
    print(
        f"[emit] shard {args.shard_index}/{args.num_shards}: "
        f"global_top has {len(fids):,} rows ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Load combined file map
    file_map: Dict[int, str] = {}
    for mp_path in args.file_map:
        file_map.update(load_file_map(mp_path))

    # Group by file_id, applying shard filter
    by_file: Dict[int, List[int]] = defaultdict(list)
    for fi, ri in zip(fids, rids):
        if fi % args.num_shards != args.shard_index:
            continue
        by_file[fi].append(ri)
    print(
        f"[emit] shard {args.shard_index}: {len(by_file):,} files assigned",
        flush=True,
    )

    os.makedirs(args.partial_dir, exist_ok=True)
    job_args = [
        (file_map[fid], rows, args.text_col,
         os.path.join(args.partial_dir, f"part-{fid:06d}.parquet"))
        for fid, rows in by_file.items()
        if fid in file_map
    ]
    print(
        f"[emit] dispatching {len(job_args)} files to {args.n_workers} workers",
        flush=True,
    )

    total_rows = 0
    done = 0
    failed: List = []
    with mp.Pool(args.n_workers) as pool:
        for fp, n, err in pool.imap_unordered(_emit_one_file_to_disk, job_args, chunksize=8):
            if err:
                failed.append((fp, err))
            else:
                total_rows += n
            done += 1
            if done % 100 == 0 or done == len(job_args):
                print(
                    f"[emit] {done}/{len(job_args)} | "
                    f"rows={total_rows:,} wall={time.time()-t0:.0f}s",
                    flush=True,
                )

    if failed:
        print(f"[emit] {len(failed)} files failed", flush=True)
        for fp, err in failed[:5]:
            print(f"  {fp}: {err}", flush=True)
    print(
        f"[emit] DONE shard {args.shard_index}: rows={total_rows:,} wall={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
