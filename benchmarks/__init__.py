"""CSR7 + OOD benchmark preprocessing.

Downloads the 11 evaluation benchmarks used in the paper and writes them
as parquet files with a single `text` column suitable for unigram-frequency
analysis or CE filtering.
"""
