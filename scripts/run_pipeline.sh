#!/usr/bin/env bash
# Rebuild the pitch table and rerun every analysis for one season label.
# Usage: scripts/run_pipeline.sh [2026]
set -euo pipefail
cd "$(dirname "$0")/.."
SEASON="${1:-2026}"
PY=.venv/bin/python
echo "== build pitch table $SEASON =="
$PY scripts/04_build_pitch_table.py --season "$SEASON" --sport 1
echo "== judgment =="
$PY scripts/05_judgment.py --label "mlb_$SEASON"
echo "== policy =="
$PY scripts/06_policy.py --label "mlb_$SEASON"
echo "== eyes vs approach =="
$PY scripts/07_eyes_vs_approach.py --label "mlb_$SEASON" --prev-season "$((SEASON - 1))"
echo "== done =="
