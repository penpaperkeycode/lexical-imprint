#!/usr/bin/env python3
"""Merge 4 local_topk parquets → global top-K → write global_top.parquet.

Uses numpy.argpartition for fast top-K (vs Python heapq.nlargest).
Output cols: file_id (int32), row_idx (int64). Single pod, ~30-60s on 500M items.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_topk", nargs="+", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gumbel_seed", type=int, default=-1,
                    help="If >=0, add Gumbel(0,1) noise to logiw before top-K. "
                         "Equivalent to Gumbel top-K sampling without replacement "
                         "(DSIR paper's exact protocol).")
    args = ap.parse_args()

    t0 = time.time()
    # Load directly into numpy via pyarrow (much faster than Python list)
    logiw_chunks = []
    fid_chunks = []
    rid_chunks = []
    for lp in args.local_topk:
        t = pq.read_table(lp)
        logiw_chunks.append(t["logiw"].to_numpy(zero_copy_only=False))
        fid_chunks.append(t["file_id"].to_numpy(zero_copy_only=False))
        rid_chunks.append(t["row_idx"].to_numpy(zero_copy_only=False))
    logiw = np.concatenate(logiw_chunks)
    fid = np.concatenate(fid_chunks)
    rid = np.concatenate(rid_chunks)
    n = len(logiw)
    print(f"[merge] loaded {n:,} candidates from {len(args.local_topk)} files ({time.time()-t0:.1f}s)", flush=True)

    # Apply Gumbel noise (Gumbel-top-K = weighted sampling without replacement, DSIR paper)
    if args.gumbel_seed >= 0:
        rng = np.random.default_rng(args.gumbel_seed)
        u = rng.uniform(low=1e-20, high=1.0, size=n)
        gumbel = -np.log(-np.log(u))
        score = logiw + gumbel
        print(f"[merge] applied Gumbel(0,1) noise (seed={args.gumbel_seed}) ({time.time()-t0:.1f}s)", flush=True)
    else:
        score = logiw

    if args.K >= n:
        idx = np.arange(n)
        print(f"[merge] K={args.K} >= n={n}, keeping all", flush=True)
    else:
        # argpartition: O(n) — gives indices of top-K (unordered)
        # Negate score because argpartition picks SMALLEST K. We want largest K.
        idx = np.argpartition(-score, args.K)[: args.K]
        print(f"[merge] argpartition top-{args.K:,} done ({time.time()-t0:.1f}s)", flush=True)

    sel_logiw = logiw[idx]
    sel_fid = fid[idx]
    sel_rid = rid[idx]
    print(
        f"[merge] selected: score range=[{sel_logiw.min():.3f}, {sel_logiw.max():.3f}] "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Sort by file_id for downstream locality (groupby)
    sort_idx = np.lexsort((sel_rid, sel_fid))
    sel_fid = sel_fid[sort_idx]
    sel_rid = sel_rid[sort_idx]
    print(f"[merge] sorted by (file_id, row_idx) ({time.time()-t0:.1f}s)", flush=True)

    file_ids = pa.array(sel_fid.astype(np.int32))
    row_idxs = pa.array(sel_rid.astype(np.int64))
    tbl = pa.table({"file_id": file_ids, "row_idx": row_idxs})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pq.write_table(tbl, args.out, compression="snappy")
    print(f"[merge] wrote {args.out} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
