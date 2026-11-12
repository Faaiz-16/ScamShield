#!/usr/bin/env bash
# Retrain the baseline on the seed sample plus every reviewed file in data/labels/,
# then evaluate it.
#
#   bash ml/retrain.sh
#
# Results are written to ml/results/. Compare them with the previous run using
# git diff ml/results/comparison.md ml/results/verdict_evaluation.md
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"

echo "== 1/4 Preparing data =="
"$python" -m ml.prepare_data
echo
echo "== 2/4 Training the baseline =="
"$python" -m ml.train_baseline
echo
echo "== 3/4 Scoring the classifier =="
"$python" -m ml.evaluate
echo
echo "== 4/4 Scoring the complete system =="
"$python" -m ml.evaluate_verdict
echo
echo "Done. See what changed with:"
echo "  git diff ml/results/comparison.md ml/results/verdict_evaluation.md"
