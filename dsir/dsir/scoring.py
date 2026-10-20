"""Importance-weight scoring + heap-based top-K (DSIR paper Eq. 4 + Alg. 1).

Per-row score:
    log iw(x) = sum_{ngram in x}  log P_t(ngram) - log P_s(ngram)

P_t / P_s are Laplace add-1 smoothed bucket multinomials.
"""
from __future__ import annotations

import heapq
from typing import Iterable, List, Tuple

import numpy as np

from .hashing import hash_ngrams


def compute_log_ratio(
    target_counts: np.ndarray,
    source_counts: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    """log P_t / P_s with Laplace add-`smoothing` smoothing.

    Returns float64 array of shape (B,).
    """
    if target_counts.shape != source_counts.shape:
        raise ValueError(
            f"shape mismatch: target {target_counts.shape} vs source {source_counts.shape}"
        )
    t = target_counts.astype(np.float64) + smoothing
    s = source_counts.astype(np.float64) + smoothing
    t /= t.sum()
    s /= s.sum()
    # log P_t - log P_s
    return np.log(t) - np.log(s)


def score_text(
    text: str,
    log_ratio: np.ndarray,
    n_max: int = 2,
) -> float:
    """Sum log-ratio over the row's ngrams. 0 if text has no ngrams."""
    buckets = hash_ngrams(text, bucket_count=log_ratio.shape[0], n_max=n_max)
    if not buckets:
        return float("-inf")
    return float(log_ratio[buckets].sum())


def top_k_heap(
    items: Iterable[Tuple[float, int]],
    k: int,
) -> List[Tuple[float, int]]:
    """Return top-K by score, descending. Items are (score, row_idx) pairs.

    Memory: O(k). Stable for ties via row_idx (smaller idx kept on tie).
    """
    heap: List[Tuple[float, int]] = []
    for score, idx in items:
        if len(heap) < k:
            heapq.heappush(heap, (score, idx))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, idx))
    return sorted(heap, reverse=True)
