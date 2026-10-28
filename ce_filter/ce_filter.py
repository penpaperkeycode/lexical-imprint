"""
ce_filter.py  —  Cross-entropy data filtering (120-core optimized version)

Optimization summary:
  1. Parallel scan: distribute work per parquet fragment; each batch adds its row and token counts to shared counters.
  2. Analytical ΔH: per-row cost O(unique words); no Counter copy or full-vocab traversal.
  3. Parallel scoring: imap_unordered + worker_batch, leveraging Linux fork zero-copy sharing.

Progress reporting (scan stage): workers add to shared mp.RawValue counters (rows, tokens,
finished fragments), and a thread in the main process prints them.
"""
from __future__ import annotations

import argparse
import json
import math
import heapq
import re
import unicodedata
import sys
import time
import os
import shutil
import tempfile
import pickle
import threading
from collections import Counter
import multiprocessing as mp

import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Errors
# ============================================================
def _fail(msg: str):
    """Stop the run with a one-line error message and exit code 1."""
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def _file_error(path: str, e: Exception) -> str:
    """One-line description of an error raised while reading `path`."""
    lines = str(e).strip().splitlines()
    return f"{path}: {type(e).__name__}: {lines[0] if lines else ''}"


def _require_column(dataset, column: str):
    """Raise if a parquet dataset/file has no `column` (instead of silently reading nothing)."""
    names = dataset.schema.names
    if column not in names:
        raise ValueError(f"no column '{column}' (columns: {', '.join(names)})")


# ============================================================
# Text preprocessing
# ============================================================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return normalize_text(text).split()


# ============================================================
# Progress reporter — sequential (scoring / file write)
# ============================================================
class Progress:
    def __init__(self, name: str, total: int | None, every: int = 50_000):
        self.name = name
        self.total = total
        self.every = max(1, every)
        self.start_t = time.time()
        self.count = 0

    def update(self, n: int = 1):
        self.count += n
        if self.count % self.every != 0:
            return
        self._print()

    def _print(self):
        elapsed = time.time() - self.start_t
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if self.total:
            pct = 100.0 * self.count / self.total
            eta = max(0, self.total - self.count) / rate if rate > 0 else float("inf")
            print(f"[{self.name}] {self.count:,}/{self.total:,} ({pct:5.1f}%) "
                  f"| {rate:,.0f} rows/s | ETA {eta:.0f}s", file=sys.stderr)
        else:
            print(f"[{self.name}] {self.count:,} rows | {rate:,.0f} rows/s", file=sys.stderr)

    def done(self):
        self._print()
        elapsed = time.time() - self.start_t
        rate = self.count / elapsed if elapsed > 0 else 0.0
        total_str = f"{self.count:,}/{self.total:,}" if self.total else f"{self.count:,} rows"
        print(f"[{self.name}] done: {total_str} in {elapsed:.1f}s ({rate:,.0f} rows/s)",
              file=sys.stderr)


# ============================================================
# Progress reporter — parallel scan only (shared-memory Value, deadlock-free)
# ============================================================
def _make_shared_counters():
    """Create shared counters backed by mp.RawValue (lock-free, OOM-safe)."""
    return (
        mp.RawValue("l", 0),  # shared_rows  (long)
        mp.RawValue("l", 0),  # shared_tokens (long)
        mp.RawValue("i", 0),  # shared_frags  (int)
    )


def _progress_printer(name, total_rows, total_frags, shared_rows, shared_tokens,
                      shared_frags, stop_event, every=50_000, interval=2.0):
    """
    Background thread: poll the shared Values and emit progress output.
    Exits with a final report once stop_event is set.
    """
    start_t = time.time()
    last_print_rows = 0
    last_frags = 0

    while not stop_event.is_set():
        stop_event.wait(timeout=interval)
        rows = shared_rows.value
        tokens = shared_tokens.value
        frags = shared_frags.value

        # Detect fragment-done events
        if frags > last_frags:
            last_frags = frags
            _scan_print(name, rows, tokens, frags, total_rows, total_frags, start_t,
                        frag_done=True)
            last_print_rows = rows
        elif rows - last_print_rows >= every:
            _scan_print(name, rows, tokens, frags, total_rows, total_frags, start_t)
            last_print_rows = rows

    # Final output
    rows = shared_rows.value
    tokens = shared_tokens.value
    frags = shared_frags.value
    elapsed = time.time() - start_t
    rate = rows / elapsed if elapsed > 0 else 0.0
    print(f"[{name}] ✓ done: {rows:,} rows | {tokens:,} tokens "
          f"| {frags}/{total_frags} frags "
          f"| {elapsed:.1f}s ({rate:,.0f} rows/s)", file=sys.stderr)


def _scan_print(name, rows, tokens, frags_done, total_rows, total_frags, start_t,
                frag_done=False):
    elapsed = time.time() - start_t
    rate = rows / elapsed if elapsed > 0 else 0.0
    frag_tag = f" ← frag {frags_done}/{total_frags} done" if frag_done else \
               f" | {frags_done}/{total_frags} frags"
    if total_rows and total_rows > 0:
        pct = 100.0 * rows / total_rows
        eta = max(0, total_rows - rows) / rate if rate > 0 else float("inf")
        print(f"[{name}] {rows:,}/{total_rows:,} ({pct:5.1f}%) "
              f"| {rate:,.0f} rows/s | ETA {eta:.0f}s{frag_tag}", file=sys.stderr)
    else:
        print(f"[{name}] {rows:,} rows | {tokens:,} tokens "
              f"| {rate:,.0f} rows/s{frag_tag}", file=sys.stderr)


# ============================================================
# Dataset helpers
# ============================================================
def _import_warcio():
    """Lazy import of warcio ArchiveIterator."""
    try:
        from warcio.archiveiterator import ArchiveIterator
        return ArchiveIterator
    except ImportError:
        raise ImportError(
            "warcio is required for WET format support. "
            "Install with: pip install warcio"
        )


def detect_format(path: str, explicit: str | None = None) -> str:
    """Detect dataset format from file extensions. Returns 'parquet' or 'wet'."""
    if explicit is not None:
        return explicit
    has_parquet = False
    has_wet = False
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in files:
            if f.endswith(".parquet"):
                has_parquet = True
            elif f.endswith(".warc.wet.gz"):
                has_wet = True
            if has_parquet and has_wet:
                raise ValueError(
                    f"Mixed formats in {path}: found both .parquet and .warc.wet.gz files"
                )
    if has_wet:
        return "wet"
    if has_parquet:
        return "parquet"
    # Fallback: try PyArrow probe
    try:
        ds.dataset(path, format="parquet")
        return "parquet"
    except Exception:
        raise ValueError(
            f"Cannot detect format for {path}: no .parquet or .warc.wet.gz files found"
        )


def get_wet_fragment_paths(path: str) -> list[str]:
    """Get sorted list of .warc.wet.gz file paths for deterministic ordering."""
    result = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".warc.wet.gz"):
                result.append(os.path.join(root, f))
    return result


def iter_wet_texts(wet_path: str, column: str = "text"):
    """Generator over one .warc.wet.gz file. Yields text for each 'conversion' record.

    Read errors propagate (a truncated file must not end the scan silently)."""
    ArchiveIterator = _import_warcio()
    with open(wet_path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "conversion":
                continue
            if column == "url":
                yield record.rec_headers.get_header("WARC-Target-URI") or ""
            else:
                content = record.content_stream().read()
                yield content.decode("utf-8", errors="replace") if content else ""


def count_rows_for_path(path: str, fmt: str) -> int | None:
    """Fast row count. Returns count for parquet, None for WET."""
    if fmt == "parquet":
        return make_dataset(path).count_rows()
    return None


def make_dataset(path: str, fmt: str = "parquet"):
    """Create PyArrow dataset. Returns None for WET format."""
    if fmt == "wet":
        return None
    return ds.dataset(path, format="parquet")


def get_fragment_paths(path: str, fmt: str = "parquet") -> list[str]:
    """Return the list of fragment (file) paths in the dataset."""
    if fmt == "wet":
        return get_wet_fragment_paths(path)
    return [frag.path for frag in make_dataset(path).get_fragments()]


def _check_listing(path: str, files: list[str]):
    """Refuse inputs that pyarrow discovery reads wrongly without an error.

    A broken symlink is left out of the listing, and a symlinked directory that
    points back up the tree (a loop) lists the same files many times.
    """
    if os.path.isdir(path):
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
            for name in names:
                p = os.path.join(root, name)
                if not name.startswith((".", "_")) and not os.path.exists(p):
                    raise FileNotFoundError(f"broken symlink: {p}")
    if len({os.path.realpath(f) for f in files}) != len(files):
        raise ValueError(f"{path}: the same file is reached through more than one path "
                         f"(symlink loop?)")


def iter_indexed_texts(path, column, batch_size, fmt="parquet", stats: dict | None = None):
    """
    Sequential (physical_row_idx, text) iterator.

    The index counts every row, null ones included, so it is the same row number the
    writer (Step 5) uses. Null rows are not yielded: they are never scored and always kept.
    If `stats` is given, it receives physical_rows and null_rows once the iterator is exhausted.
    """
    row_idx = 0
    n_null = 0
    if fmt == "wet":
        # WET records have no null text: every 'conversion' record is one row
        for wet_path in get_wet_fragment_paths(path):
            for text in iter_wet_texts(wet_path, column):
                yield row_idx, text
                row_idx += 1
    else:
        dataset = make_dataset(path)
        if hasattr(dataset, "scanner"):
            batches = dataset.scanner(columns=[column], batch_size=batch_size).to_batches()
        else:
            batches = dataset.to_batches(columns=[column], batch_size=batch_size)
        for batch in batches:
            for x in batch.column(0).to_pylist():
                if x is None:
                    n_null += 1
                else:
                    yield row_idx, (x if isinstance(x, str) else str(x))
                row_idx += 1
    if stats is not None:
        stats["physical_rows"] = row_idx
        stats["null_rows"] = n_null


# ============================================================
# Scan worker — benchmark side (shared Value, deadlock-free)
# ============================================================
_g_shared_rows = None
_g_shared_tokens = None
_g_shared_frags = None


def _bench_scan_worker_init(shared_rows, shared_tokens, shared_frags):
    global _g_shared_rows, _g_shared_tokens, _g_shared_frags
    _g_shared_rows = shared_rows
    _g_shared_tokens = shared_tokens
    _g_shared_frags = shared_frags


def _bench_scan_worker(args: tuple) -> tuple[str, int]:
    """Returns (temp_file_path, total_tokens). Counter saved to temp file to avoid pipe deadlock."""
    fragment_path, column, batch_size, tmp_dir = args
    counts = Counter()
    total_tokens = 0
    try:
        frag_ds = ds.dataset(fragment_path, format="parquet")
        _require_column(frag_ds, column)
        for batch in frag_ds.scanner(columns=[column], batch_size=batch_size).to_batches():
            rows_list = batch.column(0).to_pylist()
            batch_tokens = 0
            for x in rows_list:
                if x is None:
                    continue
                toks = tokenize(x if isinstance(x, str) else str(x))
                counts.update(toks)
                batch_tokens += len(toks)
            total_tokens += batch_tokens
            _g_shared_rows.value += len(rows_list)
            _g_shared_tokens.value += batch_tokens
    except Exception as e:
        # Never drop a file from Q silently: the parent stops the run and names the file
        raise RuntimeError(_file_error(fragment_path, e)) from None
    finally:
        _g_shared_frags.value += 1
    # Write Counter to temp file instead of returning it through pipe
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="ce_bench_", dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp_path, total_tokens


# ============================================================
# Scan worker — training data side (restricted to bench_vocab)
# ============================================================
_g_bench_vocab_for_scan: set = None


def _train_scan_worker_init(bench_vocab, shared_rows, shared_tokens, shared_frags):
    global _g_bench_vocab_for_scan, _g_shared_rows, _g_shared_tokens, _g_shared_frags
    _g_bench_vocab_for_scan = bench_vocab
    _g_shared_rows = shared_rows
    _g_shared_tokens = shared_tokens
    _g_shared_frags = shared_frags


def _train_scan_worker(args: tuple) -> tuple[str, int, int]:
    """Returns (temp_file_path, total_tokens, null_rows). Counter saved to temp file to avoid pipe deadlock."""
    fragment_path, column, batch_size, tmp_dir = args
    counts = Counter()
    total_tokens = 0
    null_rows = 0
    bench_vocab = _g_bench_vocab_for_scan
    try:
        frag_ds = ds.dataset(fragment_path, format="parquet")
        _require_column(frag_ds, column)
        for batch in frag_ds.scanner(columns=[column], batch_size=batch_size).to_batches():
            rows_list = batch.column(0).to_pylist()
            batch_tokens = 0
            for x in rows_list:
                if x is None:
                    null_rows += 1
                    continue
                toks = tokenize(x if isinstance(x, str) else str(x))
                batch_tokens += len(toks)
                for w in toks:
                    if w in bench_vocab:
                        counts[w] += 1
            total_tokens += batch_tokens
            _g_shared_rows.value += len(rows_list)
            _g_shared_tokens.value += batch_tokens
    except Exception as e:
        # Never drop a file from the counts silently: the parent stops the run and names the file
        raise RuntimeError(_file_error(fragment_path, e)) from None
    finally:
        _g_shared_frags.value += 1
    # Write Counter to temp file instead of returning it through pipe
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="ce_train_", dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp_path, total_tokens, null_rows


# ============================================================
# Scan worker — WET benchmark side
# ============================================================
def _bench_scan_worker_wet(args: tuple) -> tuple[str, int, int]:
    """WET bench scan worker. Returns (temp_file_path, total_tokens, rows_scanned)."""
    fragment_path, column, batch_size, tmp_dir = args
    counts = Counter()
    total_tokens = 0
    rows_scanned = 0
    batch_rows = 0
    batch_tokens = 0
    try:
        for text in iter_wet_texts(fragment_path, column):
            toks = tokenize(text)
            counts.update(toks)
            batch_tokens += len(toks)
            rows_scanned += 1
            batch_rows += 1
            if batch_rows >= batch_size:
                total_tokens += batch_tokens
                _g_shared_rows.value += batch_rows
                _g_shared_tokens.value += batch_tokens
                batch_rows = 0
                batch_tokens = 0
        if batch_rows > 0:
            total_tokens += batch_tokens
            _g_shared_rows.value += batch_rows
            _g_shared_tokens.value += batch_tokens
    except Exception as e:
        raise RuntimeError(_file_error(fragment_path, e)) from None
    finally:
        _g_shared_frags.value += 1
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="ce_bench_wet_", dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp_path, total_tokens, rows_scanned


# ============================================================
# Scan worker — WET training data side (restricted to bench_vocab)
# ============================================================
def _train_scan_worker_wet(args: tuple) -> tuple[str, int, int]:
    """WET train scan worker. Returns (temp_file_path, total_tokens, rows_scanned)."""
    fragment_path, column, batch_size, tmp_dir = args
    counts = Counter()
    total_tokens = 0
    rows_scanned = 0
    bench_vocab = _g_bench_vocab_for_scan
    batch_rows = 0
    batch_tokens = 0
    try:
        for text in iter_wet_texts(fragment_path, column):
            toks = tokenize(text)
            batch_tokens += len(toks)
            for w in toks:
                if w in bench_vocab:
                    counts[w] += 1
            rows_scanned += 1
            batch_rows += 1
            if batch_rows >= batch_size:
                total_tokens += batch_tokens
                _g_shared_rows.value += batch_rows
                _g_shared_tokens.value += batch_tokens
                batch_rows = 0
                batch_tokens = 0
        if batch_rows > 0:
            total_tokens += batch_tokens
            _g_shared_rows.value += batch_rows
            _g_shared_tokens.value += batch_tokens
    except Exception as e:
        raise RuntimeError(_file_error(fragment_path, e)) from None
    finally:
        _g_shared_frags.value += 1
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="ce_train_wet_", dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp_path, total_tokens, rows_scanned


def build_benchmark_q_parallel(
    path, column, n_workers, batch_size=8192, progress_every=50_000,
    fmt="parquet",
) -> tuple[dict, Counter, int, int]:
    """Parallel-scan the benchmark distribution q(w) per fragment. Returns (q, counts, total_tokens, rows_scanned)."""
    fragment_paths = get_fragment_paths(path, fmt=fmt)
    n_frags = len(fragment_paths)
    effective_workers = min(n_workers, n_frags)
    total_rows = count_rows_for_path(path, fmt)

    print(f"[bench-scan] total_rows={'?' if total_rows is None else f'{total_rows:,}'} | "
          f"fragments={n_frags} | workers={effective_workers}", file=sys.stderr)

    if n_frags == 0:
        raise ValueError(f"No fragments found in {path}. Check path and format.")

    # The workers' Counter pickles go to this scan's own temp directory, removed as a whole
    # (after a failure it may also hold pickles that were written but never received)
    scan_tmp = tempfile.mkdtemp(prefix="ce_scan_")

    # shared Value counters + pool, then the progress thread (no fork after a thread starts)
    shared_rows, shared_tokens, shared_frags = _make_shared_counters()
    pool = mp.Pool(
        processes=effective_workers,
        initializer=_bench_scan_worker_init,
        initargs=(shared_rows, shared_tokens, shared_frags),
    )
    stop_event = threading.Event()
    printer = threading.Thread(
        target=_progress_printer,
        args=("bench-scan", total_rows, n_frags, shared_rows, shared_tokens,
              shared_frags, stop_event, progress_every),
        daemon=True,
    )
    printer.start()

    args_list = [(fp, column, batch_size, scan_tmp) for fp in fragment_paths]
    total_counts = Counter()
    total_tokens = 0
    rows_scanned = 0
    worker_fn = _bench_scan_worker_wet if fmt == "wet" else _bench_scan_worker

    tmp_files = []
    failure = None
    try:
        for result in pool.imap_unordered(worker_fn, args_list):
            if fmt == "wet":
                tmp_path, partial_tokens, partial_rows = result
                rows_scanned += partial_rows
            else:
                tmp_path, partial_tokens = result
            total_tokens += partial_tokens
            tmp_files.append(tmp_path)
    except Exception as e:
        # A benchmark file could not be read: stop instead of building Q without it
        failure = e
        pool.terminate()
    finally:
        pool.close()
        pool.join()
        stop_event.set()
        printer.join(timeout=10)
    if failure is not None:
        shutil.rmtree(scan_tmp, ignore_errors=True)
        _fail(f"bench-scan: {failure}")

    # Merge counters from temp files (outside pool to avoid pipe issues)
    try:
        for tmp_path in tmp_files:
            with open(tmp_path, "rb") as f:
                partial_counts = pickle.load(f)
            total_counts.update(partial_counts)
    finally:
        shutil.rmtree(scan_tmp, ignore_errors=True)

    if total_tokens == 0:
        raise ValueError(f"Benchmark has zero tokens in column '{column}' of {path}. "
                         f"Check column/path.")

    if fmt != "wet":
        rows_scanned = total_rows

    q = {w: c / total_tokens for w, c in total_counts.items()}
    print(f"[bench-scan] vocab={len(q):,} total_tokens={total_tokens:,} "
          f"rows={rows_scanned:,}", file=sys.stderr)
    return q, total_counts, total_tokens, rows_scanned


def build_train_counts_parallel(
    path, column, bench_vocab, n_workers,
    batch_size=8192, progress_name="train-scan", progress_every=50_000,
    fmt="parquet",
) -> tuple[Counter, int, int, int]:
    """Parallel-scan the training-data counts per fragment. Returns (counts, total_tokens, rows_scanned, null_rows)."""
    fragment_paths = get_fragment_paths(path, fmt=fmt)
    n_frags = len(fragment_paths)
    effective_workers = min(n_workers, n_frags)
    total_rows = count_rows_for_path(path, fmt)

    print(f"[{progress_name}] total_rows={'?' if total_rows is None else f'{total_rows:,}'} | "
          f"fragments={n_frags} | workers={effective_workers}", file=sys.stderr)

    if n_frags == 0:
        print(f"[{progress_name}] WARNING: no fragments found, returning empty counts.",
              file=sys.stderr)
        return Counter(), 0, 0, 0

    # The workers' Counter pickles go to this scan's own temp directory, removed as a whole
    # (after a failure it may also hold pickles that were written but never received)
    scan_tmp = tempfile.mkdtemp(prefix="ce_scan_")

    # shared Value counters + pool, then the progress thread (no fork after a thread starts)
    shared_rows, shared_tokens, shared_frags = _make_shared_counters()
    pool = mp.Pool(
        processes=effective_workers,
        initializer=_train_scan_worker_init,
        initargs=(bench_vocab, shared_rows, shared_tokens, shared_frags),
    )
    stop_event = threading.Event()
    printer = threading.Thread(
        target=_progress_printer,
        args=(progress_name, total_rows, n_frags, shared_rows, shared_tokens,
              shared_frags, stop_event, progress_every),
        daemon=True,
    )
    printer.start()

    args_list = [(fp, column, batch_size, scan_tmp) for fp in fragment_paths]
    total_counts = Counter()
    total_tokens = 0
    rows_scanned = 0
    null_rows = 0
    worker_fn = _train_scan_worker_wet if fmt == "wet" else _train_scan_worker

    tmp_files = []
    failure = None
    try:
        for result in pool.imap_unordered(worker_fn, args_list):
            if fmt == "wet":
                tmp_path, partial_tokens, partial_rows = result
                rows_scanned += partial_rows
            else:
                tmp_path, partial_tokens, partial_null = result
                null_rows += partial_null
            total_tokens += partial_tokens
            tmp_files.append(tmp_path)
    except Exception as e:
        # A corpus file could not be read: stop instead of counting without it
        failure = e
        pool.terminate()
    finally:
        pool.close()
        pool.join()
        stop_event.set()
        printer.join(timeout=10)
    if failure is not None:
        shutil.rmtree(scan_tmp, ignore_errors=True)
        _fail(f"{progress_name}: {failure}")

    # Merge counters from temp files (outside pool to avoid pipe issues)
    try:
        for tmp_path in tmp_files:
            with open(tmp_path, "rb") as f:
                partial_counts = pickle.load(f)
            total_counts.update(partial_counts)
    finally:
        shutil.rmtree(scan_tmp, ignore_errors=True)

    if fmt != "wet":
        rows_scanned = total_rows

    print(f"[{progress_name}] bench_vocab_hits={sum(total_counts.values()):,} "
          f"rows={rows_scanned:,}", file=sys.stderr)
    return total_counts, total_tokens, rows_scanned, null_rows


# ============================================================
# Cross-entropy computation
# ============================================================
def compute_cross_entropy_nats(q, counts, total_tokens) -> float:
    B = sum(1 for w in q if counts.get(w, 0) <= 0)
    D = total_tokens + B
    if D <= 0:
        raise ValueError(f"Invalid D={D}")
    H = 0.0
    for w, qw in q.items():
        pw = max(counts.get(w, 0), 1) / D
        H -= qw * math.log(pw)
    return H


def nats_to_bits(H: float) -> float:
    return H / math.log(2)


# ============================================================
# Analytical ΔH computation — O(|row unique words|)
# ============================================================
def compute_delta_h_bits_analytical(
    row_counter: Counter,
    n_row_tokens: int,
    q: dict,
    train_counts: Counter,
    train_total_tokens: int,
    B: int,
    D: float,
) -> float:
    """
    ΔH = H_new - H_before for a hypothetical removal of one row.

    Derivation:
        H(q,p) = -Σ_w q(w)·log(max(c(w),1)/D)

        ΔH = log(D_new/D)                                              <- global term (scalar)
           + Σ_{w ∈ row} q(w)·log(max(c(w),1)/max(c(w)-c_row(w),1))    <- only words inside the row!

        Words not in the row have unchanged counts -> log-ratio = 0 -> no Counter copy / full-vocab traversal needed.
    """
    B_new = B
    for w, c in row_counter.items():
        old_c = train_counts.get(w, 0)
        if old_c > 0 and (old_c - c) <= 0:
            B_new += 1

    D_new = (train_total_tokens - n_row_tokens) + B_new
    if D_new <= 0 or D <= 0:
        return 0.0

    delta_nats = math.log(D_new / D)

    for w, c in row_counter.items():
        qw = q.get(w, 0.0)
        if qw == 0.0:
            continue
        old_c = train_counts.get(w, 0)
        new_c = old_c - c
        old_eff = old_c if old_c > 0 else 1
        new_eff = new_c if new_c > 0 else 1
        if old_eff != new_eff:
            delta_nats += qw * math.log(old_eff / new_eff)

    return delta_nats / math.log(2)


# ============================================================
# Scoring worker (120-core parallel)
# ============================================================
_g_q: dict = None
_g_train_counts: Counter = None
_g_train_total_tokens: int = None
_g_B: int = None
_g_D: float = None
_g_bench_vocab: set = None


def _score_worker_init(q, train_counts, train_total_tokens, B, D, bench_vocab):
    global _g_q, _g_train_counts, _g_train_total_tokens, _g_B, _g_D, _g_bench_vocab
    _g_q = q
    _g_train_counts = train_counts
    _g_train_total_tokens = train_total_tokens
    _g_B = B
    _g_D = D
    _g_bench_vocab = bench_vocab


def _score_batch(indexed_texts: list[tuple[int, str]]) -> list[tuple[float, int]]:
    """(row_idx, text) batch -> [(delta_bits, row_idx), ...]"""
    results = []
    for row_idx, text in indexed_texts:
        toks = tokenize(text)
        row_counter = Counter(w for w in toks if w in _g_bench_vocab)
        delta_bits = compute_delta_h_bits_analytical(
            row_counter, len(toks),
            _g_q, _g_train_counts, _g_train_total_tokens,
            _g_B, _g_D,
        )
        results.append((delta_bits, row_idx))
    return results


def score_rows_parallel(
    train_path, train_col,
    q, train_counts, train_total_tokens, B, D, bench_vocab, k,
    batch_size=8192, n_workers=120, worker_batch=1024, progress_every=50_000,
    fmt="parquet", total_rows_hint=None, null_rows_hint=None,
) -> list[tuple[float, int]]:
    """
    120-core parallel ΔH scoring.
    Progress is emitted via the sequential Progress class (the main thread
    collects results, so it can track progress directly).

    Recommended worker_batch: 1024-2048 (for ~120 cores).
    Returns the heap of the K removed rows as (-ΔH bits, physical row index).
    """
    if fmt == "parquet":
        total_rows = make_dataset(train_path).count_rows()
    else:
        total_rows = total_rows_hint
    prog = Progress("deltaH-score", total_rows, every=progress_every)
    heap: list[tuple[float, int]] = []
    scan_stats: dict = {}

    def _batch_generator():
        buf = []
        # Physical row indices (null rows skipped but counted): the numbering the writer uses
        for row_idx, text in iter_indexed_texts(train_path, train_col, batch_size,
                                                fmt=fmt, stats=scan_stats):
            buf.append((row_idx, text))
            if len(buf) >= worker_batch:
                yield buf
                buf = []
        if buf:
            yield buf

    pool = mp.Pool(
        processes=n_workers,
        initializer=_score_worker_init,
        initargs=(q, train_counts, train_total_tokens, B, D, bench_vocab),
    )
    try:
        for scored_batch in pool.imap_unordered(
            _score_batch, _batch_generator(), chunksize=4
        ):
            for delta_bits, row_idx in scored_batch:
                # Removal order is the total order on (-ΔH, row_idx): the heap keeps the K
                # largest pairs. Among rows with equal ΔH at the cutoff the larger row index
                # is removed first (earlier rows are kept), whatever order batches arrive in.
                item = (-delta_bits, row_idx)
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif k > 0 and item > heap[0]:
                    heapq.heapreplace(heap, item)
            prog.update(len(scored_batch))
    finally:
        pool.close()
        pool.join()

    prog.done()

    # Guard: every corpus row was either scored or null, so the scoring indices are the
    # writer's physical row numbers (checked here because --skip_seq_apply skips seq-apply)
    n_null = scan_stats.get("null_rows", 0)
    n_physical = scan_stats.get("physical_rows", 0)
    if prog.count + n_null != total_rows or n_physical != total_rows:
        _fail(f"deltaH-score: scored {prog.count:,} + null {n_null:,} rows "
              f"(physical {n_physical:,}) != corpus rows {total_rows:,} in {train_path}")
    if null_rows_hint is not None and n_null != null_rows_hint:
        _fail(f"deltaH-score: {n_null:,} null rows in {train_path}, but the Step 2 scan "
              f"counted {null_rows_hint:,} (did the corpus change during the run?)")
    if n_null:
        print(f"[deltaH-score] null text rows: {n_null:,} (not scored, always kept)",
              file=sys.stderr)
    return heap


# ============================================================
# Filtered-output helpers
# ============================================================
def _write_filtered_parquet(train_path, remove_indices, total_rows, rows_per_shard,
                            n_shards, shard_path, batch_size=8192, progress_every=50_000):
    """Write filtered parquet from parquet source. Returns (rows_seen, rows_written)."""
    train_dataset = make_dataset(train_path)
    prog_write = Progress("write-filtered", total_rows, every=progress_every)
    writers: list[pq.ParquetWriter | None] = [None] * n_shards
    shard_idx = 0
    shard_row_count = 0
    row_idx = 0
    rows_written = 0

    batches = (train_dataset.scanner(batch_size=batch_size).to_batches()
               if hasattr(train_dataset, "scanner")
               else train_dataset.to_batches(batch_size=batch_size))

    for batch in batches:
        n = batch.num_rows
        # Physical row numbers, null rows included (they are never in remove_indices)
        keep_mask = pa.array(
            [(row_idx + i) not in remove_indices for i in range(n)],
            type=pa.bool_(),
        )
        filtered_batch = batch.filter(keep_mask)
        rows_written += filtered_batch.num_rows

        if filtered_batch.num_rows > 0:
            if n_shards == 1:
                if writers[0] is None:
                    writers[0] = pq.ParquetWriter(shard_path(0), filtered_batch.schema)
                writers[0].write_batch(filtered_batch)
            else:
                remaining = filtered_batch
                while remaining.num_rows > 0:
                    capacity = rows_per_shard - shard_row_count
                    if capacity <= 0 and shard_idx < n_shards - 1:
                        if writers[shard_idx] is not None:
                            writers[shard_idx].close()
                        shard_idx += 1
                        shard_row_count = 0
                        capacity = rows_per_shard
                    elif capacity <= 0:
                        # Last shard: absorb all remaining rows
                        capacity = remaining.num_rows

                    take_n = min(remaining.num_rows, capacity)
                    chunk = remaining.slice(0, take_n)
                    remaining = remaining.slice(take_n)

                    if writers[shard_idx] is None:
                        writers[shard_idx] = pq.ParquetWriter(
                            shard_path(shard_idx), chunk.schema
                        )
                    writers[shard_idx].write_batch(chunk)
                    shard_row_count += take_n

        row_idx += n
        prog_write.update(n)

    for w in writers:
        if w is not None:
            w.close()
    prog_write.done()
    return row_idx, rows_written


def _write_filtered_wet(train_path, remove_indices, total_rows, rows_per_shard,
                        n_shards, shard_path, batch_size=8192, progress_every=50_000):
    """Write filtered parquet from WET source. Output columns: text, url. Returns (rows_seen, rows_written)."""
    ArchiveIterator = _import_warcio()
    schema = pa.schema([("text", pa.string()), ("url", pa.string())])
    prog_write = Progress("write-filtered", total_rows, every=progress_every)
    writers: list[pq.ParquetWriter | None] = [None] * n_shards
    shard_idx = 0
    shard_row_count = 0
    row_idx = 0
    rows_written = 0

    # Buffer for batch writing
    buf_texts: list[str] = []
    buf_urls: list[str] = []

    def flush_buffer():
        nonlocal shard_idx, shard_row_count
        if not buf_texts:
            return
        batch = pa.RecordBatch.from_pydict(
            {"text": buf_texts, "url": buf_urls}, schema=schema
        )
        if n_shards == 1:
            if writers[0] is None:
                writers[0] = pq.ParquetWriter(shard_path(0), schema)
            writers[0].write_batch(batch)
        else:
            remaining = batch
            while remaining.num_rows > 0:
                capacity = rows_per_shard - shard_row_count
                if capacity <= 0 and shard_idx < n_shards - 1:
                    if writers[shard_idx] is not None:
                        writers[shard_idx].close()
                    shard_idx += 1
                    shard_row_count = 0
                    capacity = rows_per_shard
                elif capacity <= 0:
                    # Last shard: absorb all remaining rows
                    capacity = remaining.num_rows

                take_n = min(remaining.num_rows, capacity)
                chunk = remaining.slice(0, take_n)
                remaining = remaining.slice(take_n)

                if writers[shard_idx] is None:
                    writers[shard_idx] = pq.ParquetWriter(shard_path(shard_idx), schema)
                writers[shard_idx].write_batch(chunk)
                shard_row_count += take_n
        buf_texts.clear()
        buf_urls.clear()

    for wet_path in get_wet_fragment_paths(train_path):
        with open(wet_path, "rb") as fh:
            for record in ArchiveIterator(fh):
                if record.rec_type != "conversion":
                    continue
                if row_idx not in remove_indices:
                    url = record.rec_headers.get_header("WARC-Target-URI") or ""
                    content = record.content_stream().read()
                    text = content.decode("utf-8", errors="replace") if content else ""
                    buf_texts.append(text)
                    buf_urls.append(url)
                    rows_written += 1
                    if len(buf_texts) >= batch_size:
                        flush_buffer()
                else:
                    # Consume content to advance iterator
                    record.content_stream().read()
                row_idx += 1
                prog_write.update(1)

    flush_buffer()

    for w in writers:
        if w is not None:
            w.close()
    prog_write.done()
    return row_idx, rows_written


# ============================================================
# Checkpoints
# ============================================================
def _ckpt_dir(out_parquet: str) -> str:
    """Checkpoint directory path derived from output path."""
    return out_parquet.rstrip("/") + ".ckpt"


def _save_ckpt(out_parquet: str, name: str, data: dict):
    d = _ckpt_dir(out_parquet)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"[ckpt] saved {name} → {path}", file=sys.stderr)


def _load_ckpt(out_parquet: str, name: str) -> dict | None:
    path = os.path.join(_ckpt_dir(out_parquet), name + ".pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[ckpt] loaded {name} ← {path}", file=sys.stderr)
    return data


def _clean_ckpt(out_parquet: str):
    import shutil
    d = _ckpt_dir(out_parquet)
    if os.path.isdir(d):
        shutil.rmtree(d)
        print(f"[ckpt] cleaned up {d}", file=sys.stderr)


def _path_fingerprint(path: str, fmt: str) -> dict:
    """Cheap dataset fingerprint (metadata only): sorted [file, size, mtime_ns] list + row count."""
    root = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
    files = []
    for p in get_fragment_paths(path, fmt=fmt):
        st = os.stat(p)
        files.append([os.path.relpath(p, root), st.st_size, st.st_mtime_ns])
    files.sort()
    return {"files": files, "rows": count_rows_for_path(path, fmt)}


def _run_config(args, bench_fmt: str, train_fmt: str) -> dict:
    """Settings that determine the selection; --resume refuses checkpoints made with others."""
    return {
        "bench_path": os.path.abspath(args.bench_path),
        "bench_col": args.bench_col,
        "bench_format": bench_fmt,
        "train_path": os.path.abspath(args.train_path),
        "train_col": args.train_col,
        "train_format": train_fmt,
        "top_pct": args.top_pct,
        "bench_files": _path_fingerprint(args.bench_path, bench_fmt),
        "corpus_files": _path_fingerprint(args.train_path, train_fmt),
    }


def _save_run_config(out_parquet: str, config: dict):
    d = _ckpt_dir(out_parquet)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)


def _check_resume_config(out_parquet: str, config: dict):
    """Exit non-zero if existing checkpoints were made with other settings or other files."""
    d = _ckpt_dir(out_parquet)
    if not any(os.path.exists(os.path.join(d, f"step{i}.pkl")) for i in (3, 4, 5)):
        return  # nothing to resume: the run starts from Step 1
    path = os.path.join(d, "run_config.json")
    if not os.path.exists(path):
        _fail(f"--resume: {d} has checkpoints but no run_config.json (older version?); "
              f"delete {d} or run without --resume")
    with open(path) as f:
        saved = json.load(f)
    diffs = []
    for key in sorted(set(saved) | set(config)):
        old, new = saved.get(key), config.get(key)
        if old == new:
            continue
        if key.endswith("_files"):
            diffs.append(f"{key}: file list, sizes, mtimes or row count changed")
        else:
            diffs.append(f"{key}: {old!r} -> {new!r}")
    if diffs:
        _fail(f"--resume: checkpoints in {d} were made with different settings or files "
              f"({'; '.join(diffs)}); delete {d} or run without --resume")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cross-entropy training-data filtering (120-core optimized)"
    )
    parser.add_argument("--bench_path",      required=True,
                        help="Benchmark parquet file, or a directory whose files are pooled into one token-weighted Q.")
    parser.add_argument("--bench_col",       default="text")
    parser.add_argument("--train_path",      default=None,
                        help="Pre-training corpus (parquet file or directory, or WET directory). "
                             "Required except with --verify_only.")
    parser.add_argument("--train_col",       default="text")
    parser.add_argument("--out_parquet",     required=True)
    parser.add_argument("--out_num_shards",  type=int,   default=1,
                        help="Number of output parquet shards. 1 = single file; N = N shard files under a directory.")
    parser.add_argument("--top_pct",         type=float, default=80.0,
                        help="Percent of rows to REMOVE: the rows with the smallest ΔH (the least aligned "
                             "with Q); all other rows are written. Must be in [0, 100). "
                             "Default 80 = the paper setting (removes 80%%, keeps 20%%).")
    parser.add_argument("--batch_size",      type=int,   default=8192)
    parser.add_argument("--worker_batch",    type=int,   default=1024,
                        help="Rows per scoring-worker batch (recommended: 1024-2048).")
    parser.add_argument("--n_workers",       type=int,   default=120,
                        help="Worker processes per pool, capped at the number of CPUs. Each scan also "
                             "uses at most one per file, and the Step 6 scan at most 20.")
    parser.add_argument("--progress_every",  type=int,   default=50000,
                        help="Progress reporting interval in rows; scan also emits on each batch/fragment completion.")
    parser.add_argument("--train_format",    choices=["parquet", "wet"], default=None,
                        help="Training-data format (auto-detected if omitted).")
    parser.add_argument("--bench_format",    choices=["parquet", "wet"], default=None,
                        help="Benchmark-data format (auto-detected if omitted).")
    parser.add_argument("--resume",          action="store_true", default=False,
                        help="Resume from checkpoints; skip stages already completed if any.")
    parser.add_argument("--verify_only",    action="store_true", default=False,
                        help="Only re-compute CE on the output parquet (skip Steps 2-5); --train_path is not needed.")
    parser.add_argument("--skip_step6",    action="store_true", default=False,
                        help="Skip Step 6 (CE re-computation verification + consistency check).")
    parser.add_argument("--skip_seq_apply", action="store_true", default=False,
                        help="Skip only the seq-apply consistency check in Step 6 (single-threaded, redundant; CE recompute still runs).")
    args = parser.parse_args()

    if args.train_path is None and not args.verify_only:
        parser.error("--train_path is required (except with --verify_only)")
    # --verify_only does not use --top_pct
    if not args.verify_only and not 0 <= args.top_pct < 100:
        parser.error(f"--top_pct must be in [0, 100) (percent of rows removed), got {args.top_pct}")
    if args.n_workers < 1:
        parser.error(f"--n_workers must be at least 1, got {args.n_workers}")
    if 0 < args.top_pct < 1 and not args.verify_only:
        print(f"[warn] --top_pct {args.top_pct} removes only {args.top_pct}% of rows; it is a percent "
              f"(80 removes 80%), not a fraction", file=sys.stderr)

    # Format detection (--verify_only reads only the parquet output, never --train_path)
    bench_fmt = detect_format(args.bench_path, args.bench_format)
    train_fmt = "parquet" if args.verify_only else detect_format(args.train_path, args.train_format)
    print(f"[format] bench={bench_fmt} | train={train_fmt}", file=sys.stderr)

    # Before anything reads the parquet inputs: pyarrow discovery skips broken symlinks and
    # follows symlink loops without an error, which would change the counts
    if bench_fmt == "parquet":
        _check_listing(args.bench_path, get_fragment_paths(args.bench_path))
    if not args.verify_only and train_fmt == "parquet":
        _check_listing(args.train_path, get_fragment_paths(args.train_path))

    # Validate columns for WET
    wet_valid_cols = {"text", "url"}
    if bench_fmt == "wet" and args.bench_col not in wet_valid_cols:
        raise ValueError(f"--bench_col '{args.bench_col}' not valid for WET format. "
                         f"Use one of: {wet_valid_cols}")
    if train_fmt == "wet" and args.train_col not in wet_valid_cols:
        raise ValueError(f"--train_col '{args.train_col}' not valid for WET format. "
                         f"Use one of: {wet_valid_cols}")

    # At most one worker process per CPU, in every pool (scans, scoring, Step 6)
    n_workers = max(1, min(args.n_workers, os.cpu_count() or 1))
    if n_workers < args.n_workers:
        print(f"[config] --n_workers {args.n_workers} capped at the CPU count: {n_workers}",
              file=sys.stderr)
    use_resume = args.resume
    print(f"[config] n_workers={n_workers} | worker_batch={args.worker_batch} | "
          f"batch_size={args.batch_size} | progress_every={args.progress_every:,}"
          f" | resume={use_resume}",
          file=sys.stderr)

    # -------- verify_only mode: Step 1 + CE recompute on output only --------
    if args.verify_only:
        print("\n=== [verify_only] Step 1: Benchmark distribution ===", file=sys.stderr)
        q, bench_counts, bench_total_tokens, bench_rows = build_benchmark_q_parallel(
            args.bench_path, args.bench_col,
            n_workers=n_workers, batch_size=args.batch_size,
            progress_every=args.progress_every, fmt=bench_fmt,
        )
        bench_vocab = set(q.keys())
        print(f"[bench] vocab={len(bench_vocab):,}", file=sys.stderr)

        # Determine output read path
        n_shards = args.out_num_shards
        if n_shards > 1:
            out_read_path = args.out_parquet
        else:
            out_read_path = args.out_parquet

        print(f"\n=== [verify_only] Recompute CE on filtered output ===", file=sys.stderr)
        step6_workers = min(n_workers, 20)
        filtered_counts, filtered_total_tokens, _, _ = build_train_counts_parallel(
            out_read_path, args.train_col, bench_vocab,
            n_workers=step6_workers, batch_size=args.batch_size,
            progress_name="train-scan(filtered)", progress_every=args.progress_every,
            fmt="parquet",  # output is always parquet
        )
        H_after_bits = nats_to_bits(
            compute_cross_entropy_nats(q, filtered_counts, filtered_total_tokens)
        )
        print(f"\n[verify_only] H_after (recomputed) = {H_after_bits:.6f} bits")
        print(f"[verify_only] Done.", file=sys.stderr)
        return

    # Settings + file fingerprints: saved with the checkpoints, checked by --resume
    run_config = _run_config(args, bench_fmt, train_fmt)
    if use_resume:
        _check_resume_config(args.out_parquet, run_config)

    # -------- Try loading checkpoints (newest first) --------
    ckpt5 = _load_ckpt(args.out_parquet, "step5") if use_resume else None
    ckpt4 = _load_ckpt(args.out_parquet, "step4") if use_resume and ckpt5 is None else None
    ckpt3 = _load_ckpt(args.out_parquet, "step3") if use_resume and ckpt5 is None and ckpt4 is None else None

    if ckpt5 is not None:
        # Resume from after Step 5: skip Steps 1-5, go straight to Step 6
        q              = ckpt5["q"]
        bench_vocab    = ckpt5["bench_vocab"]
        train_counts   = ckpt5["train_counts"]
        train_total_tokens = ckpt5["train_total_tokens"]
        H_before_bits  = ckpt5["H_before_bits"]
        total_rows     = ckpt5["total_rows"]
        null_rows      = ckpt5["null_rows"]
        k              = ckpt5["k"]
        remove_indices = ckpt5["remove_indices"]
        sum_selected_delta_bits = ckpt5["sum_selected_delta_bits"]
        print(f"[ckpt] ✓ Resuming from Step 6 (Steps 1-5 skipped)", file=sys.stderr)
        print(f"[ckpt]   total_rows={total_rows:,} | K={k:,} | "
              f"remove={len(remove_indices):,} | H_before={H_before_bits:.6f} bits",
              file=sys.stderr)
    elif ckpt4 is not None:
        # Resume from after Step 4: skip Steps 1-4
        q              = ckpt4["q"]
        bench_vocab    = ckpt4["bench_vocab"]
        train_counts   = ckpt4["train_counts"]
        train_total_tokens = ckpt4["train_total_tokens"]
        H_before_bits  = ckpt4["H_before_bits"]
        total_rows     = ckpt4["total_rows"]
        null_rows      = ckpt4["null_rows"]
        k              = ckpt4["k"]
        remove_indices = ckpt4["remove_indices"]
        sum_selected_delta_bits = ckpt4["sum_selected_delta_bits"]
        print(f"[ckpt] ✓ Resuming from Step 5 (Steps 1-4 skipped)", file=sys.stderr)
        print(f"[ckpt]   total_rows={total_rows:,} | K={k:,} | "
              f"remove={len(remove_indices):,} | H_before={H_before_bits:.6f} bits",
              file=sys.stderr)
    else:
        if ckpt3 is not None:
            # Resume from after Step 3: skip Steps 1-3
            q              = ckpt3["q"]
            bench_vocab    = ckpt3["bench_vocab"]
            train_counts   = ckpt3["train_counts"]
            train_total_tokens = ckpt3["train_total_tokens"]
            H_before_bits  = ckpt3["H_before_bits"]
            B              = ckpt3["B"]
            D              = ckpt3["D"]
            total_rows     = ckpt3["total_rows"]
            null_rows      = ckpt3["null_rows"]
            k              = ckpt3["k"]
            print(f"[ckpt] ✓ Resuming from Step 4 (Steps 1-3 skipped)", file=sys.stderr)
            print(f"[ckpt]   total_rows={total_rows:,} | K={k:,} | "
                  f"H_before={H_before_bits:.6f} bits", file=sys.stderr)
        else:
            # -------- 1. Benchmark distribution --------
            print("\n=== Step 1/6: Benchmark distribution (parallel scan) ===", file=sys.stderr)
            q, bench_counts, bench_total_tokens, bench_rows = build_benchmark_q_parallel(
                args.bench_path, args.bench_col,
                n_workers=n_workers, batch_size=args.batch_size,
                progress_every=args.progress_every, fmt=bench_fmt,
            )
            bench_vocab = set(q.keys())
            print(f"[bench] vocab={len(bench_vocab):,}", file=sys.stderr)

            # -------- 2. Training counts & H_before --------
            print("\n=== Step 2/6: Train counts before filtering (parallel scan) ===", file=sys.stderr)
            train_counts, train_total_tokens, train_rows_scanned, null_rows = build_train_counts_parallel(
                args.train_path, args.train_col, bench_vocab,
                n_workers=n_workers, batch_size=args.batch_size,
                progress_name="train-scan(before)", progress_every=args.progress_every,
                fmt=train_fmt,
            )
            if train_total_tokens == 0:
                # H would silently come out as log2|V_bench| (every benchmark word floored to count 1)
                _fail(f"train-scan(before): no word tokens in column '{args.train_col}' of "
                      f"pretraining corpus at {args.train_path}")

            H_before_bits = nats_to_bits(compute_cross_entropy_nats(q, train_counts, train_total_tokens))
            print(f"[before] H(q,p) = {H_before_bits:.6f} bits")

            B = sum(1 for w in q if train_counts.get(w, 0) <= 0)
            D = float(train_total_tokens + B)
            print(f"[prep] B={B:,} | D={D:,.0f}", file=sys.stderr)

            # -------- 3. Determine K --------
            if train_fmt == "parquet":
                total_rows = make_dataset(args.train_path).count_rows()
            else:
                total_rows = train_rows_scanned
            # K counts only rows with text. Null-text rows are never scored or removed and are
            # written unchanged, so they do not change which text rows are selected (and K can
            # never exceed the number of scored rows). top_pct 0 removes nothing; any
            # top_pct > 0 removes at least one row, as before.
            text_rows = total_rows - null_rows
            k = 0 if args.top_pct == 0 else max(1, int(text_rows * args.top_pct / 100.0))
            k = min(k, text_rows)
            print(f"\n=== Step 3/6: Selection target ===", file=sys.stderr)
            print(f"[select] total_rows={total_rows:,} | top_pct={args.top_pct}% | K={k:,}",
                  file=sys.stderr)
            if null_rows:
                print(f"[select] null_rows={null_rows:,} (never scored, always kept) | "
                      f"K counts the {text_rows:,} rows with text", file=sys.stderr)

            # Fresh run: drop checkpoints of any earlier run so --resume never mixes runs,
            # then record the settings --resume checks
            _clean_ckpt(args.out_parquet)
            _save_run_config(args.out_parquet, run_config)
            # Save checkpoint after Steps 1-3
            _save_ckpt(args.out_parquet, "step3", {
                "q": q, "bench_vocab": bench_vocab,
                "train_counts": train_counts,
                "train_total_tokens": train_total_tokens,
                "H_before_bits": H_before_bits,
                "B": B, "D": D,
                "total_rows": total_rows, "null_rows": null_rows, "k": k,
            })

        # -------- 4. ΔH scoring --------
        print("\n=== Step 4/6: ΔH scoring (analytical + parallel) ===", file=sys.stderr)
        heap = score_rows_parallel(
            args.train_path, args.train_col,
            q, train_counts, train_total_tokens, B, D, bench_vocab, k,
            batch_size=args.batch_size, n_workers=n_workers,
            worker_batch=args.worker_batch, progress_every=args.progress_every,
            fmt=train_fmt, total_rows_hint=total_rows, null_rows_hint=null_rows,
        )

        remove_indices = {idx for _, idx in heap}
        sum_selected_delta_bits = -sum(key for key, _ in heap)
        print(f"[select] remove_rows={len(remove_indices):,}", file=sys.stderr)

        # Save checkpoint after Step 4
        _save_ckpt(args.out_parquet, "step4", {
            "q": q, "bench_vocab": bench_vocab,
            "train_counts": train_counts,
            "train_total_tokens": train_total_tokens,
            "H_before_bits": H_before_bits,
            "total_rows": total_rows, "null_rows": null_rows, "k": k,
            "remove_indices": remove_indices,
            "sum_selected_delta_bits": sum_selected_delta_bits,
        })

    # -------- 5. Write filtered parquet --------
    n_shards = args.out_num_shards

    # Output path: directory if shard>1, otherwise a single file
    if n_shards > 1:
        out_dir = args.out_parquet
        def shard_path(i: int) -> str:
            return os.path.join(out_dir, f"part-{i:04d}.parquet")
    else:
        out_dir = None
        def shard_path(i: int) -> str:
            return args.out_parquet

    if ckpt5 is not None:
        print("\n=== Step 5/6: Writing filtered parquet (skipped — checkpoint) ===",
              file=sys.stderr)
    else:
        print("\n=== Step 5/6: Writing filtered parquet ===", file=sys.stderr)

        # Clean partial output from previous crashed run before writing
        if n_shards > 1:
            if os.path.isdir(out_dir):
                import glob as _glob
                old_parts = _glob.glob(os.path.join(out_dir, "part-*.parquet"))
                if old_parts:
                    print(f"[write] removing {len(old_parts)} partial shard files "
                          f"from previous run", file=sys.stderr)
                    for p in old_parts:
                        os.remove(p)
            os.makedirs(out_dir, exist_ok=True)

        # Compute max rows per shard (based on post-filter row count)
        rows_to_keep = total_rows - len(remove_indices)
        rows_per_shard = math.ceil(rows_to_keep / n_shards) if n_shards > 1 else rows_to_keep
        print(f"[write] n_shards={n_shards} | rows_to_keep={rows_to_keep:,} | "
              f"rows_per_shard≈{rows_per_shard:,}", file=sys.stderr)

        if train_fmt == "wet":
            # --- WET write path: re-read WET files, output as parquet ---
            rows_seen, rows_written = _write_filtered_wet(
                args.train_path, remove_indices,
                total_rows, rows_per_shard,
                n_shards, shard_path,
                batch_size=args.batch_size, progress_every=args.progress_every,
            )
        else:
            # --- Parquet write path (original) ---
            rows_seen, rows_written = _write_filtered_parquet(
                args.train_path, remove_indices,
                total_rows, rows_per_shard,
                n_shards, shard_path,
                batch_size=args.batch_size, progress_every=args.progress_every,
            )

        # Guard (also under --skip_seq_apply): the writer saw every corpus row and wrote
        # exactly the rows that were not removed
        if rows_seen != total_rows or rows_written != rows_to_keep:
            _fail(f"write: read {rows_seen:,} of {total_rows:,} corpus rows and wrote "
                  f"{rows_written:,} rows, expected {rows_to_keep:,}; {out_dir or args.out_parquet} is not valid")

        if n_shards > 1:
            print(f"[write] output dir: {out_dir}  ({n_shards} shards)", file=sys.stderr)
        else:
            print(f"[write] output file: {args.out_parquet}", file=sys.stderr)

        # Save checkpoint after Step 5
        _save_ckpt(args.out_parquet, "step5", {
            "q": q, "bench_vocab": bench_vocab,
            "train_counts": train_counts,
            "train_total_tokens": train_total_tokens,
            "H_before_bits": H_before_bits,
            "total_rows": total_rows, "null_rows": null_rows, "k": k,
            "remove_indices": remove_indices,
            "sum_selected_delta_bits": sum_selected_delta_bits,
        })

    # Path Step 6 will read from (single file or directory)
    out_read_path = out_dir if n_shards > 1 else args.out_parquet
    # Realised share of removed rows (null rows count in the total but are never removed)
    removed_pct = 100.0 * len(remove_indices) / total_rows if total_rows else 0.0

    if args.skip_step6:
        print("\n=== Step 6/6: SKIPPED (--skip_step6) ===", file=sys.stderr)
        print(f"\n[summary (partial)]")
        print(f"  H_before     = {H_before_bits:.6f} bits")
        print(f"  rows removed = {len(remove_indices):,} / {total_rows:,} ({removed_pct:.1f}%)")
        if null_rows:
            print(f"  null rows    = {null_rows:,} (not scored, kept)")
        print(f"  sum(selected ΔH_i) = {sum_selected_delta_bits:.6f} bits")
    else:
        # -------- 6. Verification: re-compute CE (cap workers to avoid OOM from accumulated memory through Step 5) --------
        step6_workers = min(n_workers, 20)
        print(f"\n=== Step 6/6: Recompute CE on filtered data (parallel scan, workers={step6_workers}) ===",
              file=sys.stderr)
        filtered_counts, filtered_total_tokens, _, _ = build_train_counts_parallel(
            out_read_path, args.train_col, bench_vocab,
            n_workers=step6_workers, batch_size=args.batch_size,
            progress_name="train-scan(filtered)", progress_every=args.progress_every,
            fmt="parquet",  # output is always parquet
        )
        H_after_bits = nats_to_bits(
            compute_cross_entropy_nats(q, filtered_counts, filtered_total_tokens)
        )
        print(f"[filtered] H(q,p_filtered) = {H_after_bits:.6f} bits")

        # -------- Summary --------
        print("\n" + "="*55)
        print("[summary]")
        print(f"  H_before     = {H_before_bits:.6f} bits")
        print(f"  H_after      = {H_after_bits:.6f} bits")
        print(f"  ΔH (actual)  = {H_after_bits - H_before_bits:.6f} bits")
        print(f"  rows removed = {len(remove_indices):,} / {total_rows:,} ({removed_pct:.1f}%)")
        if null_rows:
            print(f"  null rows    = {null_rows:,} (not scored, kept)")
        print("="*55)
        print(f"\n[check] sum(selected ΔH_i) = {sum_selected_delta_bits:.6f} bits "
              f"(may differ from actual ΔH due to row interactions)")

        # -------- Consistency check (skippable via --skip_seq_apply) --------
        if args.skip_seq_apply:
            print("\n[check] sequential-apply SKIPPED (--skip_seq_apply)", file=sys.stderr)
        else:
            print("\n[check] sequential-apply (consistency check)...", file=sys.stderr)
            prog_seq = Progress("seq-apply", len(remove_indices),
                                every=max(1, args.progress_every // 20))
            seq_counts = train_counts.copy()
            seq_total = train_total_tokens
            removed_seen = 0

            # Same physical row numbering as scoring and the writer
            for row_idx, text in iter_indexed_texts(args.train_path, args.train_col,
                                                    args.batch_size, fmt=train_fmt):
                if row_idx in remove_indices:
                    toks = tokenize(text)
                    seq_total -= len(toks)
                    for w in toks:
                        if w in bench_vocab:
                            seq_counts[w] -= 1
                    removed_seen += 1
                    prog_seq.update(1)
                if removed_seen == len(remove_indices):
                    break

            prog_seq.done()
            H_seq_bits = nats_to_bits(compute_cross_entropy_nats(q, seq_counts, seq_total))
            print(f"[check] H_seq_apply           = {H_seq_bits:.6f} bits")
            print(f"[check] H_seq_apply - H_after = {H_seq_bits - H_after_bits:.6e} bits (should be ~0)")

            if removed_seen != len(remove_indices):
                _fail(f"seq-apply: saw only {removed_seen:,}/{len(remove_indices):,} removed rows "
                      f"in {args.train_path}; row indexing mismatch")
            if abs(H_seq_bits - H_after_bits) > 1e-9:
                _fail(f"seq-apply: |H_seq_apply - H_after| = {abs(H_seq_bits - H_after_bits):.3e} bits "
                      f"> 1e-9; the written output is not the corpus minus the selected rows")

    # -------- Done: clean up checkpoints --------
    _clean_ckpt(args.out_parquet)


if __name__ == "__main__":
    mp.set_start_method("fork" if sys.platform == "linux" else "spawn", force=True)
    try:
        main()
    except (OSError, ValueError) as e:
        # Unusable input (missing path, unreadable or non-parquet file, broken symlink,
        # symlink loop): one line with the exception type and the first line of its message
        # (pyarrow's messages name the file), exit code 1
        lines = str(e).strip().splitlines()
        _fail(f"{type(e).__name__}: {lines[0] if lines else ''}")
