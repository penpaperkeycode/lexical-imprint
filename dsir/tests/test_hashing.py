"""Hash determinism + ngram extraction sanity."""
import pytest

from dsir.hashing import hash_ngrams, ngram_iter, normalize_text


def test_normalize_lowercase_nfkc():
    assert normalize_text("Hello World") == "hello world"
    # half-width / full-width
    assert normalize_text("Ｈello") == "hello"


def test_ngram_iter_unibi():
    tokens = ["a", "b", "c"]
    grams = list(ngram_iter(tokens, n_max=2))
    assert grams == ["a", "b", "c", "a b", "b c"]


def test_ngram_iter_short_input():
    assert list(ngram_iter(["x"], n_max=2)) == ["x"]
    assert list(ngram_iter([], n_max=2)) == []


def test_hash_deterministic_across_calls():
    h1 = hash_ngrams("the quick brown fox", bucket_count=10000)
    h2 = hash_ngrams("the quick brown fox", bucket_count=10000)
    assert h1 == h2


def test_hash_bucket_range():
    h = hash_ngrams("hello world testing", bucket_count=128)
    assert all(0 <= b < 128 for b in h)
    # 3 unigrams + 2 bigrams
    assert len(h) == 5


def test_hash_unigram_only():
    h = hash_ngrams("a b c", bucket_count=1024, n_max=1)
    assert len(h) == 3


def test_hash_empty():
    assert hash_ngrams("", bucket_count=128) == []
    assert hash_ngrams("   ", bucket_count=128) == []
