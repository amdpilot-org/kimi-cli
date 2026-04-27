"""ConsultAdvisor tool contract — amd-dev ``cd148265`` + ``3a180b86`` + ``b3bace7b``.

ConsultAdvisor writes `.consult_request.json` and polls `.consult_response.json`.
amdpilot's orchestrator is the response-writer. These tests emulate the
orchestrator side with a background thread so the agent doesn't hang.

Also exercises ``b3bace7b``: unsolicited consult responses (written without
a preceding request) should be picked up and injected as steers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tests_live.helpers.runner import KimiRunner


@pytest.fixture
def consult_agent_file(tmp_path: Path) -> Path:
    """Agent with ConsultAdvisor + WriteFile (so the agent has a concrete next action)."""
    system_md = tmp_path / "consult_system.md"
    system_md.write_text(
        "You are a test agent with access to a ConsultAdvisor tool.\n"
        "When the user says they are STUCK, call ConsultAdvisor to ask the advisor "
        "for help, then follow the advisor's guidance.\n",
        encoding="utf-8",
    )
    p = tmp_path / "consult_agent.yaml"
    p.write_text(
        f"version: 1\n"
        f"agent:\n"
        f"  name: consult-test-agent\n"
        f"  system_prompt_path: {system_md}\n"
        f"  tools:\n"
        f'    - "kimi_cli.tools.consult:ConsultAdvisor"\n'
        f'    - "kimi_cli.tools.shell:Shell"\n'
        f'    - "kimi_cli.tools.file:WriteFile"\n'
        f'    - "kimi_cli.tools.think:Think"\n',
        encoding="utf-8",
    )
    return p


class _AdvisorResponder:
    """Background thread that watches for `.consult_request.json`, records the
    observed payload (for test assertions) and writes `.consult_response.json`.
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        diagnosis: str,
        next_action: str,
        request_wait_s: float = 30.0,
    ):
        self.req_path = work_dir / ".consult_request.json"
        self.resp_path = work_dir / ".consult_response.json"
        self.diagnosis = diagnosis
        self.next_action = next_action
        self.request_wait_s = request_wait_s
        self.observed_request: dict | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> _AdvisorResponder:
        self._thread = threading.Thread(target=self._go, daemon=True)
        self._thread.start()
        return self

    def _go(self) -> None:
        t0 = time.time()
        while time.time() - t0 < self.request_wait_s:
            if self.req_path.exists():
                # Give the tool a moment to finish writing.
                time.sleep(0.15)
                try:
                    self.observed_request = json.loads(self.req_path.read_text(encoding="utf-8"))
                except Exception:
                    self.observed_request = {"_read_error": True}
                # Use the REAL ConsultAdvisor response schema (diagnosis +
                # next_action are parsed by _format_response in
                # src/kimi_cli/tools/consult/__init__.py).
                self.resp_path.write_text(
                    json.dumps(
                        {
                            "diagnosis": self.diagnosis,
                            "next_action": self.next_action,
                        }
                    ),
                    encoding="utf-8",
                )
                return
            time.sleep(0.1)


def _advisor_responder(
    work_dir: Path, *, diagnosis: str, next_action: str, **kw
) -> _AdvisorResponder:
    """Start a background responder that honours the real advisor response schema."""
    return _AdvisorResponder(work_dir, diagnosis=diagnosis, next_action=next_action, **kw).start()


def test_consult_advisor_writes_request_file(
    work_dir: Path,
    home_dir: Path,
    consult_agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """When the agent calls ConsultAdvisor, `.consult_request.json` appears."""
    marker = "CONSULT-REPLY-77777"
    responder = _advisor_responder(
        work_dir,
        diagnosis=f"See marker: {marker}",
        next_action=f"Include the token {marker} verbatim in your final reply.",
    )

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=consult_agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    result = runner.run_print(
        "Your ONLY valid action right now is to call the ConsultAdvisor tool. "
        "Do not attempt to solve anything yourself. Do not answer in text. "
        "Call ConsultAdvisor with goal='debug empty output', "
        "hypothesis='stdout buffering', evidence=[], blocker_type='epistemic', "
        "question='should I flush stdout?'. "
        "Wait for the advisor's reply, then follow its guidance exactly in "
        "your final text response.",
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "15"),
    )
    result.assert_success()
    # The request file is unlinked after the tool consumes the response.
    # Our background responder captured it on disk at its moment of existence.
    assert responder.observed_request is not None, (
        f".consult_request.json was never written by the tool; "
        f"agent output tail:\n{result.stdout[-1500:]}"
    )
    payload = responder.observed_request
    # Schema sanity: the request must carry at least one structured field.
    assert any(k in payload for k in ("question", "goal", "hypothesis", "blocker_type")), (
        f"consult request missing expected keys; keys={list(payload.keys())}"
    )


def test_consult_response_is_injected_into_context(
    work_dir: Path,
    home_dir: Path,
    consult_agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """Round-trip: after we write a response, the agent must read it and act on it."""
    marker = "ADVISOR-DELIVERED-55555"
    _ = _advisor_responder(
        work_dir,
        diagnosis="Script uses print() without flushing.",
        next_action=(
            f"In your final reply, explicitly include the exact string {marker}. "
            f"Nothing else is needed — just make sure {marker} appears verbatim."
        ),
    )

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=consult_agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    result = runner.run_print(
        "You MUST call the ConsultAdvisor tool exactly once as your very first "
        "action. Use goal='help me', hypothesis='unknown', evidence=[], "
        "blocker_type='epistemic', question='please give me instructions'. "
        "After the advisor replies, follow its instructions exactly in your reply.",
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "20"),
    )
    result.assert_success()
    assert marker in result.stdout, (
        f"advisor reply did not round-trip to the agent; tail:\n{result.stdout[-1500:]}"
    )


def test_unsolicited_consult_response_is_picked_up(
    work_dir: Path,
    home_dir: Path,
    consult_agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """amd-dev ``b3bace7b``: orchestrator can write a forced `.consult_response.json`
    WITHOUT a preceding request, and the soul hook still injects it mid-run.
    """
    marker = "FORCED-CONSULT-99999"
    resp_path = work_dir / ".consult_response.json"

    # Write the forced response AFTER the run kicks off (otherwise the soul's
    # trial-start reset may wipe the state).
    def _write_forced() -> None:
        time.sleep(2.5)
        # Real unsolicited-consult schema (see kimisoul.py::_check_unsolicited_consult):
        # recognises `diagnosis`, `next_action`, `do_not_do`, `stop_condition`.
        resp_path.write_text(
            json.dumps(
                {
                    "diagnosis": f"Forced-consult test marker: {marker}.",
                    "next_action": (f"In your final reply, include the token {marker} verbatim."),
                }
            ),
            encoding="utf-8",
        )

    threading.Thread(target=_write_forced, daemon=True).start()

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=consult_agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    result = runner.run_print(
        f"Use WriteFile to create {work_dir}/a.txt with 'a', then "
        f"{work_dir}/b.txt with 'b'. Then reply with a final message.",
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "15"),
    )
    result.assert_success()
    # Timing can miss; report but don't auto-fail on model refusing to echo.
    if marker not in result.stdout:  # pragma: no cover - timing
        raise AssertionError(
            "unsolicited consult response was not picked up by the soul hook "
            "(or the model ignored it). Either a real bug or a timing miss.\n"
            f"--- tail ---\n{result.stdout[-1500:]}"
        )
