#!/usr/bin/env python3
"""Compare pipeline vs baseline SWE-agent runs.

Reads trajectory files from two output directories and produces a comparison
table covering: resolve rate, API calls, tokens, and wall-clock time.

Usage:
    python3 scripts/analyze_comparison.py \
        --baseline trajectories/comparison_xxx/baseline_27b \
        --pipeline trajectories/comparison_xxx/pipeline_27b_9b
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_traj_files(root: Path) -> dict[str, dict[str, Any]]:
    """Map instance_id -> trajectory data from all .traj files under *root*."""
    result: dict[str, dict[str, Any]] = {}
    for f in sorted(root.rglob("*.traj")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        iid = f.stem
        result[iid] = data
    return result


def _load_preds(root: Path) -> dict[str, dict[str, Any]]:
    """Load preds.json if present."""
    preds_path = root / "preds.json"
    if not preds_path.exists():
        for p in root.rglob("preds.json"):
            preds_path = p
            break
    if preds_path.exists():
        return json.loads(preds_path.read_text())
    return {}


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from a single trajectory."""
    info = data.get("info", {})
    stats = info.get("model_stats", {})
    traj = data.get("trajectory", [])

    submission = info.get("submission", "")
    has_patch = bool(submission and submission.strip())
    exit_status = info.get("exit_status", "")

    total_exec_time = sum(s.get("execution_time", 0) for s in traj)
    n_steps = len(traj)

    return {
        "has_patch": has_patch,
        "exit_status": exit_status,
        "api_calls": stats.get("api_calls", 0),
        "tokens_sent": stats.get("tokens_sent", 0),
        "tokens_received": stats.get("tokens_received", 0),
        "instance_cost": stats.get("instance_cost", 0),
        "n_steps": n_steps,
        "total_exec_time": total_exec_time,
        "patch_len": len(submission) if submission else 0,
    }


def _aggregate(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_list:
        return {}

    n = len(metrics_list)
    n_with_patch = sum(1 for m in metrics_list if m["has_patch"])
    submitted = sum(1 for m in metrics_list if "submitted" in str(m["exit_status"]))

    def _avg(key: str) -> float:
        vals = [m[key] for m in metrics_list]
        return statistics.mean(vals) if vals else 0

    def _med(key: str) -> float:
        vals = [m[key] for m in metrics_list]
        return statistics.median(vals) if vals else 0

    return {
        "n_instances": n,
        "n_submitted": submitted,
        "n_with_patch": n_with_patch,
        "submit_rate": f"{submitted / n * 100:.1f}%",
        "patch_rate": f"{n_with_patch / n * 100:.1f}%",
        "avg_api_calls": f"{_avg('api_calls'):.1f}",
        "med_api_calls": f"{_med('api_calls'):.0f}",
        "avg_tokens_sent": f"{_avg('tokens_sent'):.0f}",
        "avg_tokens_recv": f"{_avg('tokens_received'):.0f}",
        "avg_total_tokens": f"{_avg('tokens_sent') + _avg('tokens_received'):.0f}",
        "avg_exec_time_s": f"{_avg('total_exec_time'):.1f}",
        "med_exec_time_s": f"{_med('total_exec_time'):.1f}",
        "avg_steps": f"{_avg('n_steps'):.1f}",
        "avg_cost": f"{_avg('instance_cost'):.4f}",
    }


def _print_table(baseline_agg: dict, pipeline_agg: dict) -> None:
    rows = [
        ("Instances", "n_instances"),
        ("Submitted", "n_submitted"),
        ("Has non-empty patch", "n_with_patch"),
        ("Submit rate", "submit_rate"),
        ("Patch rate", "patch_rate"),
        ("Avg API calls", "avg_api_calls"),
        ("Median API calls", "med_api_calls"),
        ("Avg tokens sent", "avg_tokens_sent"),
        ("Avg tokens received", "avg_tokens_recv"),
        ("Avg total tokens", "avg_total_tokens"),
        ("Avg exec time (s)", "avg_exec_time_s"),
        ("Median exec time (s)", "med_exec_time_s"),
        ("Avg steps", "avg_steps"),
        ("Avg cost ($)", "avg_cost"),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    col_w = 18

    header = f"{'Metric':<{label_w}} {'Baseline (27B)':<{col_w}} {'Pipeline (27B+9B)':<{col_w}}"
    print(header)
    print("─" * len(header))
    for label, key in rows:
        b = str(baseline_agg.get(key, "N/A"))
        p = str(pipeline_agg.get(key, "N/A"))
        print(f"{label:<{label_w}} {b:<{col_w}} {p:<{col_w}}")


def _per_instance_comparison(
    baseline_trajs: dict[str, dict], pipeline_trajs: dict[str, dict]
) -> None:
    common = sorted(set(baseline_trajs) & set(pipeline_trajs))
    if not common:
        print("\nNo common instances found for per-instance comparison.")
        return

    print(f"\n{'='*70}")
    print(f"Per-instance comparison ({len(common)} common instances)")
    print(f"{'='*70}")

    iid_w = 40
    col_w = 15
    header = (
        f"{'Instance':<{iid_w}} "
        f"{'B calls':<{col_w}} {'P calls':<{col_w}} "
        f"{'B patch?':<{col_w}} {'P patch?':<{col_w}}"
    )
    print(header)
    print("─" * len(header))

    for iid in common[:30]:
        bm = _extract_metrics(baseline_trajs[iid])
        pm = _extract_metrics(pipeline_trajs[iid])
        print(
            f"{iid[:iid_w-1]:<{iid_w}} "
            f"{bm['api_calls']:<{col_w}} {pm['api_calls']:<{col_w}} "
            f"{'Y' if bm['has_patch'] else 'N':<{col_w}} {'Y' if pm['has_patch'] else 'N':<{col_w}}"
        )

    if len(common) > 30:
        print(f"  ... and {len(common) - 30} more instances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading baseline trajectories from: {args.baseline}")
    baseline_trajs = _load_traj_files(args.baseline)
    print(f"  Found {len(baseline_trajs)} trajectory files")

    print(f"Loading pipeline trajectories from: {args.pipeline}")
    pipeline_trajs = _load_traj_files(args.pipeline)
    print(f"  Found {len(pipeline_trajs)} trajectory files")

    if not baseline_trajs and not pipeline_trajs:
        print("No trajectories found in either directory.")
        return

    baseline_metrics = [_extract_metrics(d) for d in baseline_trajs.values()]
    pipeline_metrics = [_extract_metrics(d) for d in pipeline_trajs.values()]

    baseline_agg = _aggregate(baseline_metrics)
    pipeline_agg = _aggregate(pipeline_metrics)

    print(f"\n{'='*70}")
    print("Aggregate Comparison")
    print(f"{'='*70}\n")
    _print_table(baseline_agg, pipeline_agg)

    _per_instance_comparison(baseline_trajs, pipeline_trajs)

    # Efficiency analysis
    if baseline_metrics and pipeline_metrics:
        b_tokens = statistics.mean(
            m["tokens_sent"] + m["tokens_received"] for m in baseline_metrics
        )
        p_tokens = statistics.mean(
            m["tokens_sent"] + m["tokens_received"] for m in pipeline_metrics
        )
        if b_tokens > 0:
            token_ratio = p_tokens / b_tokens
            print(f"\nToken usage ratio (pipeline / baseline): {token_ratio:.2f}x")

        b_calls = statistics.mean(m["api_calls"] for m in baseline_metrics)
        p_calls = statistics.mean(m["api_calls"] for m in pipeline_metrics)
        if b_calls > 0:
            call_ratio = p_calls / b_calls
            print(f"API call ratio (pipeline / baseline):  {call_ratio:.2f}x")

    # SWE-bench eval results (if available)
    for label, root in [("Baseline", args.baseline), ("Pipeline", args.pipeline)]:
        preds = _load_preds(root)
        if preds:
            n_preds = len(preds)
            n_resolved = sum(
                1 for p in preds.values()
                if isinstance(p, dict) and p.get("resolved", False)
            )
            print(f"\n{label} SWE-bench evaluation: {n_resolved}/{n_preds} resolved")

    summary_path = args.baseline.parent / "comparison_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Baseline: {args.baseline}\n")
        f.write(f"Pipeline: {args.pipeline}\n")
        f.write(f"Baseline instances: {len(baseline_trajs)}\n")
        f.write(f"Pipeline instances: {len(pipeline_trajs)}\n")
        for key in baseline_agg:
            f.write(f"{key}: baseline={baseline_agg[key]}, pipeline={pipeline_agg.get(key, 'N/A')}\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
