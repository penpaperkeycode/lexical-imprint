#!/usr/bin/env bash
# 30-second tiny demo: regenerate parquets + run end-to-end DSIR + verify output.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 examples/tiny_demo/build_demo.py
echo "[demo] OK — see examples/tiny_demo/expected_output.json"
