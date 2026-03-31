#!/usr/bin/env python3
"""Quick inspection of pipeline agent trajectory files.

Usage:
    python3 scripts/inspect_pipeline_traj.py <path-to-traj-dir-or-traj-file>
"""

import json
import sys
from pathlib import Path


def inspect(traj_path: Path) -> None:
    if traj_path.is_dir():
        traj_files = list(traj_path.rglob("*.traj"))
        if not traj_files:
            print(f"No .traj files found in {traj_path}")
            return
        for f in sorted(traj_files):
            print(f"\n{'='*60}")
            print(f"File: {f}")
            print(f"{'='*60}")
            _inspect_file(f)
    else:
        _inspect_file(traj_path)


def _inspect_file(path: Path) -> None:
    data = json.loads(path.read_text())

    # Pipeline-level data
    if "pipeline_attempts" in data:
        attempts = data["pipeline_attempts"]
        print(f"\nTotal pipeline phases: {len(attempts)}")
        for i, attempt in enumerate(attempts):
            phase = attempt.get("phase", "unknown")
            info = attempt.get("info", {})
            traj = attempt.get("trajectory", [])
            submission = info.get("submission")
            exit_status = info.get("exit_status")
            print(f"\n--- Phase {i}: {phase} ---")
            print(f"  Steps: {len(traj)}")
            print(f"  Exit status: {exit_status}")
            print(f"  Submission: {'[empty]' if not submission else f'{len(submission)} chars'}")

            if traj:
                print(f"  First action: {traj[0].get('action', '???')[:100]}")
                print(f"  Last action:  {traj[-1].get('action', '???')[:100]}")

                for j, step in enumerate(traj):
                    thought = step.get("thought", "")
                    action = step.get("action", "")
                    obs = step.get("observation", "")
                    print(f"\n  Step {j+1}:")
                    print(f"    Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
                    print(f"    Action:  {action[:200]}{'...' if len(action) > 200 else ''}")
                    print(f"    Observation: {obs[:200]}{'...' if len(obs) > 200 else ''}")

        # Verification results
        verif = data.get("verification_results", [])
        if verif:
            print(f"\nVerification results ({len(verif)}):")
            for i, vr in enumerate(verif):
                print(f"  Round {i}: score={vr.get('avg_score', '?')}, scores={vr.get('scores', [])}")
                for resp in vr.get("responses", [])[:1]:
                    print(f"    Response: {resp[:300]}{'...' if len(resp) > 300 else ''}")

    # Top-level info
    info = data.get("info", {})
    print(f"\n--- Final Info ---")
    sub = info.get("submission")
    sub_desc = f"{len(sub)} chars" if sub else "[empty]"
    print(f"  submission: {sub_desc}")
    print(f"  exit_status: {info.get('exit_status')}")
    stats = info.get("model_stats", {})
    print(f"  model_stats: cost={stats.get('instance_cost', 0):.4f}, "
          f"calls={stats.get('api_calls', 0)}, "
          f"tokens_sent={stats.get('tokens_sent', 0)}, "
          f"tokens_received={stats.get('tokens_received', 0)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inspect(Path(sys.argv[1]))
