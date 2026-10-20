#!/usr/bin/env bash
# K8s multi-pod DSIR selection driver — run for ONE source corpus.
#
# Phases A-E with shard splits + 4-pod parallel B/D + kubectl wait gates.
#
# Required env / args:
#   SOURCE_PATH    source parquet directory (e.g., HF cache of FineWeb-Edu)
#   TARGET_PATH    benchmark group directory
#   OUT_DIR        filtered parquet output dir
#   K              top-K
#   YAML_DIR       directory containing per-phase YAMLs (Phase A/B/C/D/E)
#   SOURCE_TAG     "fwedu" or "dclm" (used to pick YAML names)
#
# K8s YAMLs expected at:
#   ${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-target-cpu.yaml      (1 pod)
#   ${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-srccnt-{0..3}-cpu.yaml
#   ${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-merge-cpu.yaml       (1 pod)
#   ${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-score-{0..3}-cpu.yaml
#   ${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-finalize-cpu.yaml    (1 pod)

set -euo pipefail

: "${SOURCE_TAG:?required}"
: "${YAML_DIR:?required}"

apply_and_wait() {
  local yaml="$1"
  local name
  name=$(awk '/^  name:/{print $2; exit}' "$yaml")
  echo "[k8s] apply $name ($yaml)"
  kubectl apply -f "$yaml"
  echo "[k8s] wait condition=complete on $name (timeout 4h)"
  kubectl wait --for=condition=complete "job/${name}" --timeout=4h
}

apply_parallel_and_wait() {
  for y in "$@"; do
    name=$(awk '/^  name:/{print $2; exit}' "$y")
    echo "[k8s] apply $name ($y)"
    kubectl apply -f "$y"
  done
  for y in "$@"; do
    name=$(awk '/^  name:/{print $2; exit}' "$y")
    echo "[k8s] wait condition=complete on $name"
    kubectl wait --for=condition=complete "job/${name}" --timeout=4h
  done
}

echo "[k8s] === Phase A target counts ==="
apply_and_wait "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-target-cpu.yaml"

echo "[k8s] === Phase B source counts (4 parallel pods) ==="
apply_parallel_and_wait \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-srccnt-0-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-srccnt-1-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-srccnt-2-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-srccnt-3-cpu.yaml"

echo "[k8s] === Phase C merge log ratio ==="
apply_and_wait "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-merge-cpu.yaml"

echo "[k8s] === Phase D score + local top-K (4 parallel pods) ==="
apply_parallel_and_wait \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-score-0-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-score-1-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-score-2-cpu.yaml" \
  "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-score-3-cpu.yaml"

echo "[k8s] === Phase E finalize ==="
apply_and_wait "${YAML_DIR}/csr7-${SOURCE_TAG}-dsir-finalize-cpu.yaml"

echo "[k8s] DSIR selection complete for ${SOURCE_TAG}."
