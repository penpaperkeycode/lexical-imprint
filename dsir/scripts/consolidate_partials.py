#!/usr/bin/env python3
"""Consolidate per-file _partials/*.parquet into N final parquet shards.

Stream-writes via pq.ParquetWriter (one writer per output shard) to keep peak
memory bounded by per-partial size, not per-shard size.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq


def _consolidate_one(args):
    s, paths, out_path = args
    if not paths:
        return s, 0

    total_rows = 0
    writer = None
    try:
        for p in paths:
            try:
                t = pq.read_table(p)
            except Exception as e:
                print(f"  [shard {s}] failed to read {p}: {e}", flush=True)
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="snappy")
            else:
                # ensure schema match
                if t.schema != writer.schema:
                    t = t.cast(writer.schema)
            writer.write_table(t)
            total_rows += t.num_rows
            del t
    finally:
        if writer is not None:
            writer.close()
    return s, total_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_shards", type=int, required=True)
    ap.add_argument("--n_workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    partials = sorted(
        os.path.join(args.partial_dir, f)
        for f in os.listdir(args.partial_dir)
        if f.endswith(".parquet")
    )
    print(f"[consolidate] {len(partials)} partials → {args.out_shards} shards", flush=True)

    buckets = [[] for _ in range(args.out_shards)]
    for i, p in enumerate(partials):
        buckets[i % args.out_shards].append(p)

    os.makedirs(args.out_dir, exist_ok=True)
    cons_args = [
        (s, paths, os.path.join(args.out_dir, f"part-{s:05d}.parquet"))
        for s, paths in enumerate(buckets)
    ]
    # Limit worker count to avoid OOM — each worker streams 1 partial at a time
    # but pa.read_table can spike to several × file size.
    n_w = min(args.out_shards, args.n_workers, 16)
    print(f"[consolidate] using {n_w} workers (capped to limit memory)", flush=True)
    final_total = 0
    t0 = time.time()
    done = 0
    with mp.Pool(n_w) as pool:
        for s, n in pool.imap_unordered(_consolidate_one, cons_args, chunksize=1):
            final_total += n
            done += 1
            print(f"  shard {s}: {n:,} rows ({done}/{len(cons_args)} wall={time.time()-t0:.0f}s)", flush=True)
    print(f"[consolidate] DONE  total_rows={final_total:,} wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
