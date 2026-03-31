#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Compare Pipeline (27B+9B) vs Baseline (27B-only) on the same SWE-bench set.
#
# Runs both configurations on the same subset of instances, then uses
# scripts/analyze_comparison.py to compare resolve rate, latency, and tokens.
#
# Usage:
#   bash scripts/compare_pipeline_vs_baseline.sh [OPTIONS]
#
# Options:
#   --subset   lite|verified|full   (default: lite)
#   --split    dev|test             (default: dev)
#   --slice    SLICE_SPEC           (default: :10, i.e. first 10 instances)
#   --workers  N                    (default: 2)
#   --large-url URL                 (default: http://localhost:30001/v1)
#   --small-url URL                 (default: http://localhost:30000/v1)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBSET="lite"
SPLIT="dev"
SLICE=":10"
WORKERS=2
LARGE_URL="http://localhost:30001/v1"
SMALL_URL="http://localhost:30000/v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)    SUBSET="$2";    shift 2 ;;
        --split)     SPLIT="$2";     shift 2 ;;
        --slice)     SLICE="$2";     shift 2 ;;
        --workers)   WORKERS="$2";   shift 2 ;;
        --large-url) LARGE_URL="$2"; shift 2 ;;
        --small-url) SMALL_URL="$2"; shift 2 ;;
        *)           echo "Unknown: $1"; exit 1 ;;
    esac
done

EVALUATE_FLAG="False"
if [ -n "${SWE_BENCH_API_KEY:-}" ]; then
    EVALUATE_FLAG="True"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_OUT="${REPO_ROOT}/trajectories/comparison_${TIMESTAMP}/baseline_27b"
PIPELINE_OUT="${REPO_ROOT}/trajectories/comparison_${TIMESTAMP}/pipeline_27b_9b"

echo "=============================================="
echo " Pipeline vs Baseline Comparison"
echo "  Subset:     ${SUBSET}"
echo "  Split:      ${SPLIT}"
echo "  Slice:      ${SLICE}"
echo "  Workers:    ${WORKERS}"
echo "  Evaluate:   ${EVALUATE_FLAG}"
echo "  Large URL:  ${LARGE_URL}"
echo "  Small URL:  ${SMALL_URL}"
echo "  Baseline → ${BASELINE_OUT}"
echo "  Pipeline → ${PIPELINE_OUT}"
echo "=============================================="

COMMON_ARGS=(
    "--instances.type=swe_bench"
    "--instances.subset=${SUBSET}"
    "--instances.split=${SPLIT}"
    "--instances.slice=${SLICE}"
    "--instances.shuffle=True"
    "--instances.evaluate=${EVALUATE_FLAG}"
    "--instances.deployment.docker_args=--memory=10g"
    "--num_workers=${WORKERS}"
)

# ── Run 1: Baseline (single 27B model) ──────────────────────────────
echo ""
echo ">>> [1/3] Running BASELINE (Qwen3.5-27B only) ..."
echo ""
sweagent run-batch \
    "${COMMON_ARGS[@]}" \
    --output_dir "${BASELINE_OUT}" \
    --config "${REPO_ROOT}/config/baseline_qwen27b.yaml"

# ── Run 2: Pipeline (27B planning + 9B coding + 27B verification) ───
echo ""
echo ">>> [2/3] Running PIPELINE (27B + 9B) ..."
echo ""
sweagent run-batch \
    "${COMMON_ARGS[@]}" \
    --output_dir "${PIPELINE_OUT}" \
    --config "${REPO_ROOT}/config/pipeline_qwen.yaml"

# ── Analysis ─────────────────────────────────────────────────────────
echo ""
echo ">>> [3/3] Analyzing results ..."
echo ""
python3 "${SCRIPT_DIR}/analyze_comparison.py" \
    --baseline "${BASELINE_OUT}" \
    --pipeline "${PIPELINE_OUT}"

echo ""
echo "=== Comparison complete ==="
echo "Results are in: trajectories/comparison_${TIMESTAMP}/"
