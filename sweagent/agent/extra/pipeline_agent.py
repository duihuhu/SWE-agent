"""Pipeline agent that orchestrates a multi-model workflow:
Phase 1 (Planning): A large model retrieves relevant code and creates a plan.
Phase 2 (Coding): A small model generates code based on the plan.
Phase 3 (Verification): A large model reviews the code and provides feedback.
If verification fails, Phase 2 is retried with the feedback.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template
from typing_extensions import Self

from sweagent.agent.agents import DefaultAgent, PipelineAgentConfig
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    InstanceStats,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import TotalCostLimitExceededError
from sweagent.tools.parsing import ActionParser
from sweagent.tools.tools import ToolConfig
from sweagent.types import AgentInfo, AgentRunResult, StepOutput
from sweagent.utils.log import get_logger

PLAN_FILE_PATH = "/root/plan.md"


class PipelineAgent:
    """Agent that implements a three-phase pipeline:

    1. Planning (large model): retrieves code, analyzes the problem, writes a plan
    2. Coding (small model): generates a patch based on the plan
    3. Verification (large model): reviews the patch and provides feedback

    If verification fails, Phase 2 is re-run with the feedback injected.
    """

    def __init__(self, config: PipelineAgentConfig):
        self.config = config.model_copy(deep=True)
        self._hooks: list[AbstractAgentHook] = []
        self._chook = CombinedAgentHook()
        self.logger = get_logger("swea-pipeline", emoji="🔄")

        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | None = None
        self._output_dir: Path | None = None
        self._traj_path: Path | None = None

        self._planning_agent: DefaultAgent | None = None
        self._coding_agent: DefaultAgent | None = None
        self._verification_model: AbstractModel | None = None

        self._plan: str = ""
        self._attempt_data: list[dict[str, Any]] = []
        self._verification_results: list[dict[str, Any]] = []
        self._total_stats = InstanceStats()
        self._verification_stats_snapshot = InstanceStats()
        self._best_coding_attempt_idx: int | None = None
        self.replay_config = None

    @classmethod
    def from_config(cls, config: PipelineAgentConfig) -> Self:
        return cls(config)

    def add_hook(self, hook: AbstractAgentHook) -> None:
        self._chook.add_hook(hook)
        self._hooks.append(hook)

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        self._env = env
        self._problem_statement = problem_statement
        self._output_dir = output_dir
        self._traj_path = output_dir / (self._problem_statement.id + ".traj")
        self._total_stats = InstanceStats()
        self._attempt_data = []
        self._verification_results = []
        self._plan = ""
        self._verification_stats_snapshot = InstanceStats()
        self._best_coding_attempt_idx = None

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    def _run_planning_phase(self) -> str:
        """Run the planning agent to retrieve relevant code and create a plan.

        The planning agent's prompt should instruct it to write the plan to
        ``/root/plan.md`` before submitting.

        Returns:
            The plan text produced by the planning agent.
        """
        self.logger.info("=" * 20 + " PHASE 1: PLANNING " + "=" * 20)
        assert self._env is not None
        assert self._problem_statement is not None
        assert self._output_dir is not None

        planning_config = self.config.planning_agent.model_copy(deep=True)
        if self.config.cost_limit > 0:
            remaining_budget = self.config.cost_limit - self._total_stats.instance_cost
            if remaining_budget < planning_config.model.per_instance_cost_limit:
                planning_config.model.per_instance_cost_limit = remaining_budget

        self._planning_agent = DefaultAgent.from_config(planning_config)
        for hook in self._hooks:
            self._planning_agent.add_hook(hook)

        plan_output_dir = self._output_dir / "phase_planning"
        self._planning_agent.setup(
            env=self._env,
            problem_statement=self._problem_statement,
            output_dir=plan_output_dir,
        )

        step_output = StepOutput()
        while not step_output.done:
            step_output = self._planning_agent.step()
            self._planning_agent.save_trajectory()

        self._total_stats += self._planning_agent.model.stats
        self._attempt_data.append({
            "phase": "planning",
            **self._planning_agent.get_trajectory_data(),
        })

        plan = self._extract_plan()
        self.logger.info("Plan extracted (%d chars)", len(plan))
        return plan

    def _extract_plan(self) -> str:
        """Read the plan from ``/root/plan.md``, falling back to the last
        substantial thought in the planning trajectory."""
        assert self._env is not None
        try:
            plan = self._env.read_file(PLAN_FILE_PATH)
            if plan.strip():
                return plan.strip()
        except Exception:
            self.logger.debug("Could not read plan file, extracting from trajectory")

        if self._planning_agent is None:
            return ""
        for step in reversed(self._planning_agent.trajectory):
            thought = step.get("thought", "")
            if thought and len(thought) > 50:
                return thought
        return ""

    # ------------------------------------------------------------------
    # Phase 2: Coding
    # ------------------------------------------------------------------

    def _run_coding_phase(
        self, plan: str, feedback: str = "", previous_patch: str | None = None,
    ) -> StepOutput:
        """Run the coding agent with the plan (and optional feedback) injected.

        The plan and any verification feedback are appended to the coding
        agent's ``instance_template`` inside a ``<plan>`` tag.

        Returns:
            The final StepOutput from the coding agent.
        """
        retry_label = " (with feedback)" if feedback else ""
        self.logger.info("=" * 20 + f" PHASE 2: CODING{retry_label} " + "=" * 20)
        assert self._env is not None
        assert self._problem_statement is not None
        assert self._output_dir is not None

        coding_config = self.config.coding_agent.model_copy(deep=True)
        if self.config.cost_limit > 0:
            remaining_budget = self.config.cost_limit - self._total_stats.instance_cost
            if remaining_budget < coding_config.model.per_instance_cost_limit:
                coding_config.model.per_instance_cost_limit = remaining_budget

        injected_plan = plan
        if previous_patch:
            injected_plan += (
                "\n\n--- YOUR PREVIOUS PATCH ---\n"
                "This is the patch you submitted in your last attempt. "
                "Do NOT repeat the same mistakes:\n\n"
                "```diff\n" + previous_patch + "\n```"
            )
        if feedback:
            injected_plan += (
                "\n\n--- VERIFICATION FEEDBACK ---\n"
                "Your previous attempt was reviewed and the following issues were found. "
                "Please address them:\n\n" + feedback
            )

        coding_config.templates.instance_template = (
            coding_config.templates.instance_template
            + "\n\n<plan>\n" + injected_plan + "\n</plan>"
        )

        coding_attempt = len([d for d in self._attempt_data if d.get("phase") == "coding"])
        self._coding_agent = DefaultAgent.from_config(coding_config)
        for hook in self._hooks:
            self._coding_agent.add_hook(hook)

        coding_output_dir = self._output_dir / f"phase_coding_{coding_attempt}"
        self._coding_agent.setup(
            env=self._env,
            problem_statement=self._problem_statement,
            output_dir=coding_output_dir,
        )

        step_output = StepOutput()
        while not step_output.done:
            step_output = self._coding_agent.step()
            self._coding_agent.save_trajectory()

        self._total_stats += self._coding_agent.model.stats
        self._attempt_data.append({
            "phase": "coding",
            "attempt": coding_attempt,
            **self._coding_agent.get_trajectory_data(),
        })

        return step_output

    # ------------------------------------------------------------------
    # Phase 3: Verification
    # ------------------------------------------------------------------

    def _run_verification_phase(self, submission: str | None) -> tuple[float, str]:
        """Verify the submission using the verification model.

        Returns:
            ``(score, feedback)`` where score is a float 0-10 and feedback is
            the model's textual response.
        """
        self.logger.info("=" * 20 + " PHASE 3: VERIFICATION " + "=" * 20)
        assert self._problem_statement is not None

        if not submission or not submission.strip():
            self.logger.warning("Empty submission, skipping verification")
            return 0.0, "No patch was submitted. Please generate a valid code patch."

        if self._verification_model is None:
            self._verification_model = get_model(
                self.config.verification.model,
                ToolConfig(parse_function=ActionParser()),
            )

        problem_text = self._problem_statement.get_problem_statement()
        edited_files = ""
        if self._coding_agent is not None:
            edited_files = self._coding_agent.info.get("edited_files30", "")

        instance_msg = Template(self.config.verification.instance_template).render(
            problem_statement=problem_text,
            plan=self._plan,
            submission=submission,
            edited_files30=edited_files or "Empty. No edited files found.",
            accept_score=self.config.accept_score,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.verification.system_template, "message_type": "thought"},
            {"role": "user", "content": instance_msg, "message_type": "thought"},
        ]

        scores: list[float] = []
        responses: list[str] = []
        for _ in range(self.config.verification.n_samples):
            try:
                result = self._verification_model.query(messages)
                response_text: str = result["message"]
                responses.append(response_text)
                score = self._parse_score(response_text)
                scores.append(score)
            except Exception as e:
                self.logger.warning("Verification query failed: %s", e)
                scores.append(0.0)
                responses.append(f"Verification failed: {e}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        best_response = responses[0] if responses else "No verification response"

        assert self._verification_model is not None
        new_stats = self._verification_model.stats
        delta = new_stats - self._verification_stats_snapshot
        self._verification_stats_snapshot = new_stats.model_copy(deep=True)
        self._total_stats += delta
        verification_result = {
            "scores": scores,
            "avg_score": avg_score,
            "responses": responses,
            "messages": [{"role": m["role"], "content": m["content"][:500]} for m in messages],
        }
        self._verification_results.append(verification_result)
        self.logger.info(
            "Verification score: %.2f (threshold: %.2f)", avg_score, self.config.accept_score
        )

        return avg_score, best_response

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from the verification response.

        Handles formats like "7", "7.5", "Score: 7", "7/10", "3 / 10".
        For "X/10" patterns, X is taken as the score.
        Falls back to the first number on the last non-empty line.
        """
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            slash_match = re.search(r"(\d+\.?\d*)\s*/\s*10", line)
            if slash_match:
                score = float(slash_match.group(1))
                return max(0.0, min(10.0, score))
            numbers = re.findall(r"(?<!/)\b(\d+\.?\d*)\b(?!/)", line)
            if numbers:
                score = float(numbers[-1])
                return max(0.0, min(10.0, score))
        return 0.0

    # ------------------------------------------------------------------
    # Soft reset: undo coding changes without restarting the container
    # ------------------------------------------------------------------

    def _soft_reset(self) -> None:
        """Reset the repository to the base commit without restarting
        the container, so the plan file and environment state are preserved."""
        assert self._env is not None
        if self._env.repo is not None:
            self.logger.info("Soft-resetting repository for next coding attempt")
            reset_cmds = self._env.repo.get_reset_commands()
            skip_for_soft_reset = {"git fetch", "git status"}
            filtered = [c for c in reset_cmds if c not in skip_for_soft_reset]
            if not filtered:
                return
            try:
                self._env.communicate(
                    " && ".join(filtered),
                    timeout=60,
                    check="warn",
                )
            except Exception as e:
                self.logger.warning("Soft reset failed: %s", e)

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def get_trajectory_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "pipeline_attempts": self._attempt_data,
            "verification_results": self._verification_results,
        }
        best_coding = self._get_best_coding_data()
        if best_coding is not None:
            data.update(copy.deepcopy(best_coding))
            data["info"]["model_stats"] = self._total_stats.model_dump()
        else:
            data["trajectory"] = []
            data["history"] = []
            data["info"] = AgentInfo(model_stats=self._total_stats.model_dump())
        return data

    def _get_best_coding_data(self) -> dict[str, Any] | None:
        """Return trajectory data for the best coding attempt.

        Uses ``_best_coding_attempt_idx`` (set in ``run()``) to stay aligned
        with the ``best_submission`` tracked in the main loop.  Falls back to
        the last coding attempt if the index is not set.
        """
        coding_attempts = [
            d for d in self._attempt_data if d.get("phase") == "coding"
        ]
        if not coding_attempts:
            return None

        idx = self._best_coding_attempt_idx
        if idx is not None and idx < len(coding_attempts):
            best = coding_attempts[idx]
        else:
            best = coding_attempts[-1]

        return {k: v for k, v in best.items() if k not in ("phase", "attempt")}

    def save_trajectory(self) -> None:
        data = self.get_trajectory_data()
        assert self._traj_path is not None
        self._traj_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)
        self._chook.on_run_start()

        # Phase 1: Planning
        try:
            self._plan = self._run_planning_phase()
        except TotalCostLimitExceededError:
            raise
        except Exception as e:
            self.logger.error("Planning phase failed: %s. Continuing without plan.", e)
            self._plan = ""
        if not self._plan:
            self.logger.warning(
                "Planning phase produced no plan, coding agent will work without guidance"
            )

        self._soft_reset()
        self.save_trajectory()

        # Phase 2 + 3: Coding / Verification loop
        best_submission: str | None = None
        best_score = -1.0
        best_coding_agent: DefaultAgent | None = None
        feedback = ""
        last_submission: str | None = None

        for i_retry in range(self.config.max_verification_retries + 1):
            if self._total_stats.instance_cost > self.config.cost_limit > 0:
                self.logger.info("Cost limit reached, stopping pipeline")
                break

            step_output = self._run_coding_phase(
                self._plan, feedback=feedback, previous_patch=last_submission,
            )
            self.save_trajectory()

            submission = step_output.submission
            self.logger.info(
                "Coding phase produced submission: %s",
                f"{len(submission)} chars" if submission else "[empty/None]",
            )

            if submission is not None and submission.strip():
                if best_submission is None:
                    best_submission = submission
            elif self._coding_agent is not None:
                fallback = self._coding_agent.info.get("submission")
                if fallback and fallback.strip():
                    self.logger.info("Using submission from coding agent info as fallback")
                    submission = fallback
                    if best_submission is None:
                        best_submission = submission

            try:
                score, verification_response = self._run_verification_phase(submission)
            except TotalCostLimitExceededError:
                raise
            except Exception as e:
                self.logger.error("Verification failed: %s", e)
                score = 0.0
                verification_response = str(e)

            self.save_trajectory()

            if score > best_score and submission is not None and submission.strip():
                best_score = score
                best_submission = submission
                best_coding_agent = self._coding_agent
                self._best_coding_attempt_idx = i_retry

            if score >= self.config.accept_score:
                self.logger.info(
                    "Verification passed (score=%.2f), accepting submission", score
                )
                break

            last_submission = submission

            if i_retry < self.config.max_verification_retries:
                self.logger.info(
                    "Verification failed (score=%.2f), retrying coding phase (%d/%d)",
                    score,
                    i_retry + 1,
                    self.config.max_verification_retries,
                )
                feedback = verification_response
                self._soft_reset()
            else:
                self.logger.info(
                    "Max verification retries reached, using best submission (score=%.2f)",
                    best_score,
                )

        report_agent = best_coding_agent or self._coding_agent
        self._chook.on_run_done(
            trajectory=report_agent.trajectory if report_agent else [],
            info=report_agent.info if report_agent else AgentInfo(),
        )

        self.save_trajectory()
        self.logger.info("Pipeline trajectory saved to %s", self._traj_path)

        data = self.get_trajectory_data()
        info = data.setdefault("info", {})
        if best_submission is not None and best_submission.strip():
            info["submission"] = best_submission
            info["exit_status"] = "submitted"
        else:
            self.logger.warning("Pipeline produced no valid submission")
            info.setdefault("exit_status", "exit_error")
        return AgentRunResult(info=data["info"], trajectory=data.get("trajectory", []))
