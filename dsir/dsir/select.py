"""High-level single-machine API: select_topk_indices(source, target, K).

For the multi-pod / large-corpus path, use the 5-phase CLI under scripts/.
The tiny demo and unit tests verify behavior via this function.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .counting import count_buckets
from .scoring import compute_log_ratio, score_text, top_k_heap


def select_topk_indices(
    source_texts: List[str],
    target_texts: Iterable[str],
    k: int,
    bucket_count: int = 10_000,
    n_max: int = 2,
    smoothing: float = 1.0,
    length_normalize: bool = False,
) -> List[Tuple[float, int]]:
    """Select top-K indices into source_texts by DSIR importance weight.

    Args:
        source_texts: list of source documents
        target_texts: iterable of target (benchmark) documents
        k: number of source rows to select
        bucket_count: B (default 10 000)
        n_max: maximum n-gram order (default 2)
        smoothing: Laplace smoothing constant (default 1.0)
        length_normalize: if True, score = mean log_iw per ngram (removes length bias).
            Default False = sum (paper's DSIR).

    Returns:
        List of (score, row_idx) sorted by score descending, length min(k, len(source)).
    """
    from .hashing import hash_ngrams

    target_counts = count_buckets(target_texts, bucket_count, n_max)
    source_counts = count_buckets(source_texts, bucket_count, n_max)
    log_ratio = compute_log_ratio(target_counts, source_counts, smoothing)

    def scored():
        for i, t in enumerate(source_texts):
            buckets = hash_ngrams(t, bucket_count=bucket_count, n_max=n_max)
            if not buckets:
                yield float("-inf"), i
                continue
            arr = log_ratio[buckets]
            s = float(arr.mean()) if length_normalize else float(arr.sum())
            yield s, i

    return top_k_heap(scored(), k=k)
