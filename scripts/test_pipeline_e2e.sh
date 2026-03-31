#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# SWE-agent Pipeline Agent — End-to-end test script
#
# Connects to SGLang model services:
#   - Qwen3.5-9B  @ http://localhost:30000/v1
#   - Qwen3.5-27B @ http://localhost:30001/v1
#
# Prerequisites:
#   1. Python >=3.11 with sweagent installed  (pip install -e .)
#   2. Docker running (for SWE-bench sandboxed environments)
#   3. SGLang services running on ports 30000 and 30001
#
# Usage:
#   # Step 1 — Smoke test (import + config + connectivity check, no SWE-bench)
#   bash scripts/test_pipeline_e2e.sh --smoke
#
#   # Step 2 — Single-instance test (1 SWE-bench Lite instance)
#   bash scripts/test_pipeline_e2e.sh --single
#
#   # Step 3 — Full SWE-bench Lite batch
#   bash scripts/test_pipeline_e2e.sh --batch [--workers N]
#
#   # With external reference repo(s) for planning
#   bash scripts/test_pipeline_e2e.sh --single \
#     --ref-repo https://github.com/example/reference-repo.git \
#     --ref-repo https://github.com/example/another-repo.git
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/config/pipeline_qwen.yaml"

SMALL_MODEL_URL="http://localhost:30000/v1"
LARGE_MODEL_URL="http://localhost:30001/v1"

# Defaults
NUM_WORKERS=4
MODE="smoke"
REFERENCE_REPOS=()
REFERENCE_REPO_ROOT="/root/reference_repos"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)       MODE="smoke";  shift ;;
        --single)      MODE="single"; shift ;;
        --batch)       MODE="batch";  shift ;;
        --workers)     NUM_WORKERS="$2"; shift 2 ;;
        --config)      CONFIG_FILE="$2"; shift 2 ;;
        --small-url)   SMALL_MODEL_URL="$2"; shift 2 ;;
        --large-url)   LARGE_MODEL_URL="$2"; shift 2 ;;
        --ref-repo)    REFERENCE_REPOS+=("$2"); shift 2 ;;
        --ref-root)    REFERENCE_REPO_ROOT="$2"; shift 2 ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " SWE-agent Pipeline Agent E2E Test"
echo " Mode:        ${MODE}"
echo " Config:      ${CONFIG_FILE}"
echo " Small model: ${SMALL_MODEL_URL}"
echo " Large model: ${LARGE_MODEL_URL}"
if [ ${#REFERENCE_REPOS[@]} -gt 0 ]; then
    echo " Ref repos:   ${REFERENCE_REPOS[*]}"
    echo " Ref root:    ${REFERENCE_REPO_ROOT}"
fi
echo "============================================"

INSTANCE_EXTRA_ARGS=()
if [ ${#REFERENCE_REPOS[@]} -gt 0 ]; then
    REFERENCE_REPOS_RAW=$(printf "%s\n" "${REFERENCE_REPOS[@]}")
    ref_json=$(REFERENCE_REPOS_RAW="$REFERENCE_REPOS_RAW" python3 - <<'PY'
import json, os
repos = os.environ.get("REFERENCE_REPOS_RAW", "")
items = [x for x in repos.split("\n") if x]
print(json.dumps(items))
PY
)
    INSTANCE_EXTRA_ARGS+=("--instances.reference_repos=${ref_json}")
    INSTANCE_EXTRA_ARGS+=("--instances.reference_repo_root=${REFERENCE_REPO_ROOT}")
fi

# ------------------------------------------------------------------
# Step 1: Smoke test
# ------------------------------------------------------------------
smoke_test() {
    echo ""
    echo ">>> [1/3] Validating Python imports..."
    python3 -c "
from sweagent.agent.agents import (
    PipelineAgentConfig,
    VerificationConfig,
    AgentConfig,
    get_agent_from_config,
)
from sweagent.agent.extra.pipeline_agent import PipelineAgent
print('  [OK] All imports successful')
"

    echo ">>> [2/3] Validating YAML config..."
    python3 -c "
import yaml
from sweagent.agent.agents import PipelineAgentConfig, DefaultAgentConfig, VerificationConfig

with open('${CONFIG_FILE}') as f:
    raw = yaml.safe_load(f)

agent_cfg = raw['agent']
assert agent_cfg['type'] == 'pipeline', f\"Expected type=pipeline, got {agent_cfg['type']}\"

planning = DefaultAgentConfig(**agent_cfg['planning_agent'])
coding   = DefaultAgentConfig(**agent_cfg['coding_agent'])
verif    = VerificationConfig(**agent_cfg['verification'])
pipeline = PipelineAgentConfig(
    planning_agent=planning,
    coding_agent=coding,
    verification=verif,
    max_verification_retries=agent_cfg.get('max_verification_retries', 3),
    accept_score=agent_cfg.get('accept_score', 7.0),
    cost_limit=agent_cfg.get('cost_limit', 5.0),
)
print(f'  [OK] Config parsed successfully')
print(f'       Planning model: {planning.model.name} @ {planning.model.api_base}')
print(f'       Coding model:   {coding.model.name} @ {coding.model.api_base}')
print(f'       Verif model:    {verif.model.name} @ {verif.model.api_base}')
print(f'       max_retries={pipeline.max_verification_retries}, accept_score={pipeline.accept_score}, cost_limit={pipeline.cost_limit}')
"

    echo ">>> [3/3] Checking SGLang service connectivity..."
    local ok=true

    if curl -sf --max-time 5 "${SMALL_MODEL_URL}/models" > /dev/null 2>&1; then
        model_name=$(curl -sf --max-time 5 "${SMALL_MODEL_URL}/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "parse-error")
        echo "  [OK] Small model (9B) reachable at ${SMALL_MODEL_URL} — model: ${model_name}"
    else
        echo "  [FAIL] Cannot reach small model at ${SMALL_MODEL_URL}"
        ok=false
    fi

    if curl -sf --max-time 5 "${LARGE_MODEL_URL}/models" > /dev/null 2>&1; then
        model_name=$(curl -sf --max-time 5 "${LARGE_MODEL_URL}/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "parse-error")
        echo "  [OK] Large model (27B) reachable at ${LARGE_MODEL_URL} — model: ${model_name}"
    else
        echo "  [FAIL] Cannot reach large model at ${LARGE_MODEL_URL}"
        ok=false
    fi

    if [ "$ok" = false ]; then
        echo ""
        echo "  WARNING: Some SGLang services are not reachable."
        echo "  The smoke test (import + config) passed, but the actual run will fail."
        echo "  Make sure SGLang is running before using --single or --batch."
    fi

    echo ""
    echo "=== Smoke test PASSED ==="
}

# ------------------------------------------------------------------
# Step 2: Single-instance test
# ------------------------------------------------------------------
single_test() {
    echo ""
    echo ">>> [Single Test] Running pipeline agent on 1 SWE-bench Lite instance..."
    echo ""

    # Use cloud evaluation only if SWE_BENCH_API_KEY is set
    local evaluate_flag="False"
    if [ -n "${SWE_BENCH_API_KEY:-}" ]; then
        evaluate_flag="True"
        echo "    SWE_BENCH_API_KEY detected, will submit to cloud evaluation."
    else
        echo "    No SWE_BENCH_API_KEY set, skipping cloud evaluation."
        echo "    (Set 'export SWE_BENCH_API_KEY=<key>' to enable cloud eval)"
    fi

    sweagent run-batch \
        --num_workers=1 \
        --instances.type=swe_bench \
        --instances.subset=lite \
        --instances.split=dev \
        --instances.slice=":1" \
        --instances.evaluate="${evaluate_flag}" \
        --instances.deployment.docker_args='--memory=10g' \
        "${INSTANCE_EXTRA_ARGS[@]}" \
        --config "${CONFIG_FILE}"

    echo ""
    echo "=== Single instance test completed ==="

    # Show the generated prediction
    local traj_dir
    traj_dir=$(find trajectories/ -name "preds.json" -newer "${CONFIG_FILE}" 2>/dev/null | head -1)
    if [ -n "$traj_dir" ]; then
        echo "    Prediction file: ${traj_dir}"
        echo "    --- Prediction content ---"
        python3 -c "
import json, sys
preds = json.load(open('${traj_dir}'))
for p in preds.values() if isinstance(preds, dict) else preds:
    pid = p.get('instance_id', 'unknown')
    patch = p.get('model_patch', '')
    print(f'  Instance: {pid}')
    print(f'  Patch length: {len(patch)} chars')
    if patch:
        # Show first 20 lines of the patch
        lines = patch.splitlines()
        for line in lines[:20]:
            print(f'    {line}')
        if len(lines) > 20:
            print(f'    ... ({len(lines)-20} more lines)')
    else:
        print('  [WARNING] Empty patch!')
" 2>/dev/null || echo "    (Could not parse prediction file)"
    fi
}

# ------------------------------------------------------------------
# Step 3: Full SWE-bench Lite batch run
# ------------------------------------------------------------------
batch_test() {
    echo ""
    echo ">>> [Batch Test] Running pipeline agent on SWE-bench Lite (test split)..."
    echo "    Workers: ${NUM_WORKERS}"
    echo ""

    local evaluate_flag="False"
    if [ -n "${SWE_BENCH_API_KEY:-}" ]; then
        evaluate_flag="True"
        echo "    SWE_BENCH_API_KEY detected, will submit to cloud evaluation."
    else
        echo "    No SWE_BENCH_API_KEY set, skipping cloud evaluation."
        echo "    (Set 'export SWE_BENCH_API_KEY=<key>' to enable cloud eval)"
    fi

    sweagent run-batch \
        --num_workers="${NUM_WORKERS}" \
        --instances.type=swe_bench \
        --instances.subset=lite \
        --instances.split=test \
        --instances.shuffle=True \
        --instances.evaluate="${evaluate_flag}" \
        --instances.deployment.docker_args='--memory=10g' \
        "${INSTANCE_EXTRA_ARGS[@]}" \
        --config "${CONFIG_FILE}"

    echo ""
    echo "=== Batch test completed ==="
    echo "    Check the output directory for preds.json with predictions."
}

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
cd "${REPO_ROOT}"

case "${MODE}" in
    smoke)
        smoke_test
        ;;
    single)
        smoke_test
        single_test
        ;;
    batch)
        smoke_test
        batch_test
        ;;
esac
