"""UCE-filter (named CE-filter in the code), paper §5.1.

Scores every pre-training-corpus row by the analytic change ΔH in word-level
unigram cross-entropy against a benchmark target distribution Q if that row were
removed, then removes the --top_pct percent of rows with the smallest ΔH (the
least aligned with Q) in one shot and writes the rest. The paper uses
--top_pct 80 (removes 80%, keeps 20%), the default. Corpus counts of benchmark
words are floored at one, P(w) = max(c(w), 1) / (T + B); this is not Laplace
add-one smoothing.
"""
