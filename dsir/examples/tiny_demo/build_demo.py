#!/usr/bin/env python3
"""Build the tiny_demo source/target parquets + run DSIR end-to-end.

Run:  python3 examples/tiny_demo/build_demo.py
"""
from __future__ import annotations

import json
import os
import sys
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))  # repo root
from dsir.select import select_topk_indices  # noqa: E402


SOURCE_TEXTS = (
    [f"mathematical equations and proof of theorem {i}" for i in range(200)]
    + [f"celebrity gossip article number {i} about pop stars" for i in range(200)]
    + [f"weather forecast for tuesday in city {i}" for i in range(200)]
    + [f"sports recap of game {i} between teams" for i in range(200)]
    + [f"polynomial integral calculus practice problem {i}" for i in range(200)]
)
TARGET_TEXTS = [
    f"prove that the polynomial of degree {i} satisfies the equation"
    for i in range(50)
] + [
    f"compute the derivative and integral of expression {i}"
    for i in range(50)
]


def main():
    src_path = os.path.join(ROOT, "source_corpus.parquet")
    tgt_path = os.path.join(ROOT, "target_corpus.parquet")
    out_json = os.path.join(ROOT, "expected_output.json")

    pq.write_table(pa.table({"text": pa.array(SOURCE_TEXTS, type=pa.large_string())}), src_path)
    pq.write_table(pa.table({"text": pa.array(TARGET_TEXTS, type=pa.large_string())}), tgt_path)
    print(f"[demo] wrote {src_path} ({len(SOURCE_TEXTS)} rows)")
    print(f"[demo] wrote {tgt_path} ({len(TARGET_TEXTS)} rows)")

    K = 100
    chosen = select_topk_indices(SOURCE_TEXTS, TARGET_TEXTS, k=K, bucket_count=4096)
    chosen_idx = sorted(i for _, i in chosen)
    # Sanity: most top-K rows should fall in the math/calculus block (idx 0-199 or 800-999).
    in_target_aligned = sum(
        1 for i in chosen_idx if (0 <= i < 200) or (800 <= i < 1000)
    )
    print(f"[demo] selected K={len(chosen_idx)}, target-aligned={in_target_aligned}/{len(chosen_idx)}")
    assert in_target_aligned >= int(0.85 * K), \
        f"expected >=85% target-aligned, got {in_target_aligned}/{K}"

    # Checksums for reproducibility test
    src_sha = sha256(open(src_path, "rb").read()).hexdigest()
    tgt_sha = sha256(open(tgt_path, "rb").read()).hexdigest()
    out = {
        "K": K,
        "selected_indices": chosen_idx,
        "selected_count": len(chosen_idx),
        "target_aligned_count": in_target_aligned,
        "source_corpus_sha256": src_sha,
        "target_corpus_sha256": tgt_sha,
        "bucket_count": 4096,
        "n_max": 2,
        "smoothing": 1.0,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[demo] wrote {out_json}")


if __name__ == "__main__":
    main()
