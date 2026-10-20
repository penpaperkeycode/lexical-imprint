#!/usr/bin/env bash
# Single-machine end-to-end DSIR selection (small / medium corpora ≤ a few hundred GB).
#
# Usage:
#   bash run_dsir_select.sh \
#       --source <source_path> \
#       --target <target_path> \
#       --out_dir <out_dir> \
#       --K <K> \
#       [--bucket_count 10000] [--n_max 2] [--workers $(nproc)]

set -euo pipefail

BUCKET_COUNT=10000
N_MAX=2
WORKERS=$(nproc)
OUT_SHARDS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --K) K="$2"; shift 2 ;;
    --bucket_count) BUCKET_COUNT="$2"; shift 2 ;;
    --n_max) N_MAX="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --out_shards) OUT_SHARDS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

: "${SOURCE:?--source required}"
: "${TARGET:?--target required}"
: "${OUT_DIR:?--out_dir required}"
: "${K:?--K required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${OUT_DIR}/_dsir_work"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

echo "[run] Phase A — target counts"
python3 "${SCRIPT_DIR}/target_count.py" \
    --target_path "${TARGET}" \
    --bucket_count "${BUCKET_COUNT}" --n_max "${N_MAX}" \
    --n_workers "${WORKERS}" \
    --out "${WORK_DIR}/target_counts.npy"

echo "[run] Phase B — source counts (single pod path: 1 shard list)"
python3 "${SCRIPT_DIR}/split_source_shards.py" \
    --source_path "${SOURCE}" --num_splits 1 \
    --out_prefix "${WORK_DIR}/shard_list"
python3 "${SCRIPT_DIR}/source_count_shard.py" \
    --shard_list "${WORK_DIR}/shard_list_0.txt" \
    --bucket_count "${BUCKET_COUNT}" --n_max "${N_MAX}" \
    --n_workers "${WORKERS}" \
    --out "${WORK_DIR}/source_counts_0.npy"

echo "[run] Phase C — log ratio"
python3 "${SCRIPT_DIR}/merge_log_ratio.py" \
    --target_counts "${WORK_DIR}/target_counts.npy" \
    --source_partials "${WORK_DIR}/source_counts_0.npy" \
    --out "${WORK_DIR}/log_ratio.npy"

echo "[run] Phase D — score + local top-K"
python3 "${SCRIPT_DIR}/score_shard.py" \
    --shard_list "${WORK_DIR}/shard_list_0.txt" \
    --log_ratio "${WORK_DIR}/log_ratio.npy" \
    --K "${K}" \
    --n_max "${N_MAX}" \
    --n_workers "${WORKERS}" \
    --file_id_offset 0 \
    --out_parquet "${WORK_DIR}/local_topk_0.parquet"

echo "[run] Phase E — finalize, write filtered parquets"
python3 "${SCRIPT_DIR}/finalize.py" \
    --local_topk "${WORK_DIR}/local_topk_0.parquet" \
    --K "${K}" \
    --out_dir "${OUT_DIR}" \
    --out_shards "${OUT_SHARDS}" \
    --n_workers "${WORKERS}"

echo "[run] DONE. Filtered parquets in: ${OUT_DIR}"
