"""Top-K via heap == sorted truth on small corpora."""
import numpy as np

from dsir.scoring import top_k_heap
from dsir.select import select_topk_indices


def test_top_k_heap_matches_sort():
    items = [(s, i) for i, s in enumerate([3.0, 1.0, 5.0, 4.0, 2.0])]
    top3 = top_k_heap(items, k=3)
    # descending by score
    assert [i for _, i in top3] == [2, 3, 0]
    assert [s for s, _ in top3] == [5.0, 4.0, 3.0]


def test_top_k_heap_k_larger_than_input():
    items = [(1.0, 0), (2.0, 1)]
    top5 = top_k_heap(items, k=5)
    assert len(top5) == 2


def test_select_topk_picks_target_aligned_rows():
    # Source: 5 rows where rows 0,2 are target-aligned; rows 1,3,4 are off-distribution.
    target = ["mathematical equations and derivatives"] * 20
    source = [
        "mathematical equations and derivatives",   # idx 0 → high score
        "celebrity gossip news headlines",          # idx 1 → low
        "mathematical equations integral calculus", # idx 2 → high
        "weather forecast tomorrow",                # idx 3 → low
        "sports game last night final score",       # idx 4 → low
    ]
    top2 = select_topk_indices(source, target, k=2, bucket_count=2048)
    chosen = sorted([i for _, i in top2])
    assert chosen == [0, 2]
