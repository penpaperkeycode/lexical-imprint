"""Pyarrow streaming I/O helpers, robust to large_string offset overflow on >2GB files.

`scan_text_column` is a generator that yields the text column from either a
dataset directory or a single parquet path (row-level stream, constant memory).
"""
from __future__ import annotations

import os
from typing import Iterator, List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def list_parquet_files(path: str) -> List[str]:
    """If `path` is a directory, return a sorted list of .parquet files; if a single file, return [path]."""
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in names:
                if n.endswith(".parquet"):
                    files.append(os.path.join(root, n))
        return sorted(files)
    return [path]


def scan_text_column(
    path: str,
    text_col: str = "text",
    batch_size: int = 8192,
) -> Iterator[str]:
    """Stream rows from `text_col`. Casts string→large_string defensively."""
    files = list_parquet_files(path)
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_col]):
            col = batch.column(text_col)
            if col.type == pa.string():
                col = col.cast(pa.large_string())
            for v in col.to_pylist():
                if v is not None:
                    yield v


def write_filtered_parquet(
    rows: List[dict],
    out_path: str,
    schema: pa.Schema | None = None,
) -> None:
    """Convenience: write a list of dicts to a single parquet (large_string-safe)."""
    if not rows:
        # Empty row list still produces an empty parquet as long as a schema is given.
        tbl = pa.table({k: [] for k in (schema.names if schema else ["text"])}, schema=schema)
    else:
        tbl = pa.Table.from_pylist(rows, schema=schema)
        # cast any string column to large_string
        cols = []
        for i, f in enumerate(tbl.schema):
            c = tbl.column(i)
            if c.type == pa.string():
                c = c.cast(pa.large_string())
            cols.append(c)
        tbl = pa.table(
            {tbl.schema.names[i]: cols[i] for i in range(len(cols))},
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(tbl, out_path, compression="snappy")
