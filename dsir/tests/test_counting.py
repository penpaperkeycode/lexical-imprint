"""Bucket counts: streaming == naive."""
import numpy as np

from dsir.counting import count_buckets, count_buckets_streaming
from dsir.hashing import hash_ngrams


def test_count_matches_naive():
    texts = ["alpha beta gamma", "alpha beta", "delta epsilon zeta"]
    B = 1024
    counts = count_buckets(texts, bucket_count=B, n_max=2)
    expected = np.zeros(B, dtype=np.int64)
    for t in texts:
        for b in hash_ngrams(t, bucket_count=B, n_max=2):
            expected[b] += 1
    assert (counts == expected).all()


def test_count_streaming_equals_full():
    texts = ["foo bar"] * 5 + ["baz"] * 3
    B = 256
    full = count_buckets(texts, bucket_count=B)
    streamed = sum(count_buckets_streaming(iter(texts), bucket_count=B, batch_size=2))
    assert (full == streamed).all()


def test_count_empty_input():
    assert count_buckets([], bucket_count=64).sum() == 0
