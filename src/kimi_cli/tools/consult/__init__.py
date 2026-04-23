"""ConsultAdvisor tool — executor-initiated pull-based consult mechanism.

The executor writes a structured request to /workspace/.consult_request.json.
The orchestrator's ExecutionMonitor detects the request, routes it to the
advisor model (Opus), and writes the response to /workspace/.consult_response.json.
The executor polls for the response and returns it.

Design constraints (Linus/Kai reviewer bar):
  - Advisor does NOT call tools, does NOT give patch-level output.
  - Response limited to ~500 tokens of structured guidance.
  - Max 5 consults per trial.
  - Advisor must not become a hidden solver.
"""

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any, override

from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from loguru import logger
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import load_desc

CONSULT_REQUEST_FILENAME = ".consult_request.json"
CONSULT_RESPONSE_FILENAME = ".consult_response.json"

# Maximum time to wait for advisor response (seconds)
RESPONSE_TIMEOUT_S = 120

# Poll interval when waiting for response (seconds)
POLL_INTERVAL_S = 2

# Max consults per trial (enforced client-side as a safety net;
# the orchestrator also enforces this)
MAX_CONSULTS_PER_TRIAL = 5


class Params(BaseModel):
    current_goal: str = Field(description="What you are currently trying to achieve.")
    current_hypothesis: str = Field(
        description="Your current hypothesis about the root cause or solution approach."
    )
    strongest_evidence: str = Field(
        description=(
            "The strongest piece of evidence supporting or contradicting your "
            "hypothesis (e.g., an error message, profiler output, test result)."
        )
    )
    tried_actions: list[str] = Field(
        description="List of actions you have already tried and their outcomes.",
        default_factory=list,
    )
    blocker_type: str = Field(
        description=(
            "Type of blocker: 'epistemic' (don't know what's wrong), "
            "'infra' (environment/setup issue vs target bug confusion), "
            "'hypothesis_exhaustion' (tried all ideas, none worked), "
            "or 'objective_contract' (target/objective may be unachievable or ambiguous)."
        )
    )
    concrete_question: str = Field(
        description="A specific question you want the advisor to answer."
    )


class ConsultAdvisor(CallableTool2[Params]):
    name: str = "ConsultAdvisor"
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        description = load_desc(
            Path(__file__).parent / "consult_advisor.md",
            {"MAX_CONSULTS": MAX_CONSULTS_PER_TRIAL},
        )
        super().__init__(description=description)
        self._runtime = runtime
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._consult_count = 0

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        # Enforce client-side max consult limit
        if self._consult_count >= MAX_CONSULTS_PER_TRIAL:
            return ToolError(
                message=(
                    f"You have already used all {MAX_CONSULTS_PER_TRIAL} "
                    f"advisor consults for this trial. You must proceed on "
                    f"your own from here."
                ),
                brief="Consult limit reached",
            )

        work_dir = Path(str(self._work_dir))
        request_path = work_dir / CONSULT_REQUEST_FILENAME
        response_path = work_dir / CONSULT_RESPONSE_FILENAME

        # Clean up any stale response file
        with contextlib.suppress(OSError):
            response_path.unlink(missing_ok=True)

        # Write structured request
        request_data = {
            "consult_number": self._consult_count + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "current_goal": params.current_goal,
            "current_hypothesis": params.current_hypothesis,
            "strongest_evidence": params.strongest_evidence,
            "tried_actions": params.tried_actions,
            "blocker_type": params.blocker_type,
            "concrete_question": params.concrete_question,
        }

        try:
            request_path.write_text(
                json.dumps(request_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            return ToolError(
                message=f"Failed to write consult request: {e}",
                brief="Write failed",
            )

        logger.info(
            "ConsultAdvisor: wrote request #{n} — {q}",
            n=self._consult_count + 1,
            q=params.concrete_question[:80],
        )

        # Poll for response
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        while time.monotonic() < deadline:
            if response_path.exists():
                try:
                    raw = response_path.read_text(encoding="utf-8").strip()
                    if raw:
                        response = json.loads(raw)
                        self._consult_count += 1

                        # Clean up request file
                        request_path.unlink(missing_ok=True)

                        # Format response for executor
                        return self._format_response(response)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("ConsultAdvisor: bad response file: {e}", e=e)

            await asyncio.sleep(POLL_INTERVAL_S)

        # Timeout — clean up and report
        request_path.unlink(missing_ok=True)
        return ToolError(
            message=(
                "Advisor consult timed out after "
                f"{RESPONSE_TIMEOUT_S}s. The advisor may be unavailable. "
                "Continue working on your own."
            ),
            brief="Consult timeout",
        )

    def _format_response(self, response: dict[str, Any]) -> ToolReturnValue:
        """Format the advisor response into a readable tool output."""
        parts: list[str] = []
        parts.append(f"## Advisor Response (consult #{self._consult_count})")
        parts.append("")

        if diagnosis := response.get("diagnosis"):
            parts.append(f"**Diagnosis:** {diagnosis}")
        if confidence := response.get("confidence"):
            parts.append(f"**Confidence:** {confidence}")
        if next_action := response.get("next_action"):
            parts.append(f"**Recommended next action:** {next_action}")
        if do_not_do := response.get("do_not_do"):
            parts.append(f"**Do NOT do:** {do_not_do}")
        if stop_condition := response.get("stop_condition"):
            parts.append(f"**Stop condition:** {stop_condition}")

        need_benchmark = response.get("need_benchmark_now")
        if need_benchmark is not None:
            parts.append(f"**Run benchmark now:** {'YES' if need_benchmark else 'No'}")

        remaining = MAX_CONSULTS_PER_TRIAL - self._consult_count
        parts.append("")
        parts.append(f"_({remaining} consult(s) remaining this trial)_")

        output = "\n".join(parts)
        return ToolOk(
            output=output,
            message=f"Advisor consult #{self._consult_count} complete.",
        )
