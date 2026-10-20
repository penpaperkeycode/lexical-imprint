"""Bucket-count accumulation for source/target corpora (Phase A/B).

Single-pass streaming with constant memory (B int64 = 80 KB at B=10 000).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .hashing import hash_ngrams


def count_buckets(
    texts: Iterable[str],
    bucket_count: int,
    n_max: int = 2,
) -> np.ndarray:
    """Iterate texts, hash all ngrams, accumulate counts in a B-vector.

    Returns int64 array of shape (bucket_count,).
    """
    counts = np.zeros(bucket_count, dtype=np.int64)
    for t in texts:
        for b in hash_ngrams(t, bucket_count=bucket_count, n_max=n_max):
            counts[b] += 1
    return counts


def count_buckets_streaming(
    text_stream,
    bucket_count: int,
    n_max: int = 2,
    batch_size: int = 8192,
):
    """Same as count_buckets but yields a partial counts array every batch.

    Useful when a K8s pod is long-running and needs progress / checkpoint.
    """
    buf = []
    for t in text_stream:
        buf.append(t)
        if len(buf) >= batch_size:
            yield count_buckets(buf, bucket_count, n_max)
            buf.clear()
    if buf:
        yield count_buckets(buf, bucket_count, n_max)
