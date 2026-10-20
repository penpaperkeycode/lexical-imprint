"""Hashed n-gram feature extraction (DSIR paper § 3.2).

Determinism:
- mmh3.hash(.., signed=False) is byte-stable across Python 3.x (mmh3 ≥ 4.0).
- Whitespace tokenize after NFKC + lowercase.
- Unigram (n=1) + bigram (n=2) by default (paper default n_max=2).
"""
from __future__ import annotations

import unicodedata
from typing import Iterator, List

import mmh3


def normalize_text(text: str) -> str:
    """NFKC + lowercase + collapse whitespace.

    This normalization must be applied identically on both source and target sides.
    """
    return unicodedata.normalize("NFKC", text).lower()


def ngram_iter(tokens: List[str], n_max: int = 2) -> Iterator[str]:
    """Yield unigrams, bigrams, ..., n_max-grams in sequence.

    Bigrams are space-joined (e.g., "hello world").
    """
    for n in range(1, n_max + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


def hash_ngrams(text: str, bucket_count: int, n_max: int = 2) -> List[int]:
    """Text → list of bucket ids (length = unigram + bigram count).

    Args:
        text: raw text
        bucket_count: B in paper (default 10 000)
        n_max: maximum n-gram order (default 2 = uni + bi)

    Returns:
        List of bucket indices in [0, bucket_count).
    """
    norm = normalize_text(text)
    tokens = norm.split()
    return [
        mmh3.hash(ng, signed=False) % bucket_count
        for ng in ngram_iter(tokens, n_max=n_max)
    ]
