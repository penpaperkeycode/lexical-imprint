"""Cross-entropy data-filtering pipeline (paper §5.1).

Selects the top X% of pretraining-corpus rows whose removal would most
increase the unigram cross-entropy against a benchmark target distribution.
"""
