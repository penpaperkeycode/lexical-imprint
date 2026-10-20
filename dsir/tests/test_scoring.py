"""log_iw / Laplace smoothing / score_text."""
import numpy as np
import pytest

from dsir.counting import count_buckets
from dsir.scoring import compute_log_ratio, score_text


def test_log_ratio_shape_match():
    target = np.array([10, 5, 0, 0, 1], dtype=np.int64)
    source = np.array([1, 1, 100, 50, 1], dtype=np.int64)
    lr = compute_log_ratio(target, source)
    assert lr.shape == (5,)
    # bucket 0/1: target heavy → positive log ratio
    assert lr[0] > 0
    assert lr[1] > 0
    # bucket 2/3: source heavy → negative
    assert lr[2] < 0
    assert lr[3] < 0


def test_log_ratio_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_log_ratio(np.zeros(5), np.zeros(7))


def test_log_ratio_smoothing_avoids_inf():
    target = np.array([1, 0], dtype=np.int64)
    source = np.array([0, 1], dtype=np.int64)
    lr = compute_log_ratio(target, source, smoothing=1.0)
    # add-1 smoothing keeps zero-count buckets finite
    assert np.isfinite(lr).all()


def test_score_text_uses_buckets():
    target_texts = ["benchmark style content"] * 10
    source_texts = ["unrelated source text"] * 10
    B = 1024
    t = count_buckets(target_texts, B)
    s = count_buckets(source_texts, B)
    lr = compute_log_ratio(t, s)
    # benchmark-style text should score higher than source-style text
    sb = score_text("benchmark style content", lr)
    su = score_text("unrelated source text", lr)
    assert sb > su


def test_score_text_empty():
    lr = np.zeros(64)
    assert score_text("", lr) == float("-inf")
