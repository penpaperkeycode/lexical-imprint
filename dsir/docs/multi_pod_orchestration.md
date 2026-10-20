# Multi-pod Orchestration

## Why split

Source corpora here are large: 339M rows / ~280B tokens (FineWeb-Edu) and 630M
rows / ~500B tokens (DCLM). Single-pod scoring would cost ~1-2 h per source for
phases B/D. We split phases B and D across 4 pods per source for ~4× wall-clock
reduction.

## Splitting rule (deterministic)

`scripts/split_source_shards.py`:
1. Walk source directory, collect all `*.parquet` paths.
2. Sort lexicographically.
3. Round-robin across $P$ pods: pod $i$ gets files at index $i, i+P, i+2P, \dots$

This is run **once per source** and the resulting shard lists are reused by both
phase B (source counts) and phase D (scoring). Same files seen by the same pod
in both phases.

## Phase-by-phase

| Phase | Pods | Inputs | Outputs | Wall (estimate) |
|---|---|---|---|---|
| A target counts | 1 | benchmark dir | `target_counts.npy` | 5 min |
| B source counts | 4 (parallel) | shard lists 0..3 | `source_counts_{0..3}.npy` | 15 min |
| C log ratio | 1 | target + 4 source partials | `log_ratio.npy` | 1 s |
| D score + local top-K | 4 (parallel) | shard lists + log_ratio | `local_topk_{0..3}.parquet` + sidecar `.file_map.txt` | 15 min |
| E finalize | 1 | 4 local_topks | filtered parquet shards | 10 min |

Total per source: ~45 min wall (vs. ~3 h single-pod).

## Why 4-pod top-K is exact (not approximate)

Per-pod heap of size $K$ keeps every potential global-top-K candidate. After
merging 4 heaps of size $K$ each (4K total), we re-sort and keep the top $K$
overall (Phase E uses `heapq.nlargest`). This is provably equivalent to a
single-machine top-K because:

- Suppose row $r$ is in the global top-K. It belongs to exactly one pod's shard
  list. That pod's local top-K must contain $r$ (otherwise the pod has $K$
  rows scoring higher than $r$, so a fortiori the global has at least $K$ such
  rows, so $r$ is not in the global top-K — contradiction).

- Hence the union of 4 pod top-Ks is a superset of the global top-K, and Phase E
  re-sorts to extract the exact top-K.

## Memory per pod

- Per-pod heap of size $K$: $K \times 24$ bytes (logiw + file_id + row_idx) ≈
  3 GB for DCLM ($K=126\,M$), 1.6 GB for FWEDU ($K=68\,M$). Within the 100 Gi
  per-pod cluster limit.
- Bucket counts: $B \times 8$ bytes = 80 KB at $B=10\,000$. Negligible.

## Scaling beyond 4 pods

`split_source_shards.py --num_splits 8` (or 16) and matching K8s YAMLs gives
linear scaling for B/D. Phase C/E remain single-pod (their work is bound by
4 .npy / 4 parquet files merge, both sub-minute even at 16 pods).
