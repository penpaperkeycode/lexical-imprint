"""CSR7 + out-of-target (OOT-4) benchmark preprocessing.

Downloads the 11 benchmarks of the paper's UCE-filter study (7 CSR7 targets,
4 out-of-target tasks) and writes them as parquet files with a single `text`
column, the benchmark text that defines the target distribution Q for
unigram-frequency analysis or CE filtering.
"""
