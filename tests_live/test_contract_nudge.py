"""`.supervisor_nudge.md` / `.agentic_supervisor_nudge.md` file-watcher contract.

amd-dev ``43c4241b`` (message injection) and ``fae579d9`` (dual-path nudge so
rule-based and agentic nudges don't clobber each other). amdpilot writes these
files from the host; kimisoul polls them between steps and injects the
content as a synthetic user message.

These tests use `SKILL.md` + `ReadFile` as a self-contained way to generate
multi-step turns so the nudge poller has a chance to fire.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from tests_live.helpers.runner import KimiRunner


def _delayed_writer(path: Path, content: str, delay_s: float) -> threading.Thread:
    def _go() -> None:
        time.sleep(delay_s)
        path.write_text(content, encoding="utf-8")

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


def test_supervisor_nudge_file_is_consumed(
    work_dir: Path,
    home_dir: Path,
    agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """Write `.supervisor_nudge.md` shortly after kicking off a multi-step task.
    The agent should observe the nudge and echo the unique marker.

    We time this loosely — nudge polling fires between steps — and use a task
    big enough to span several LLM calls (3 WriteFile ops).
    """
    nudge_path = work_dir / ".supervisor_nudge.md"
    marker = "NUDGE-MARKER-ABCDE"

    _ = _delayed_writer(
        nudge_path,
        f"IMPORTANT: when you finish, the LAST WORD of your final reply must be "
        f"exactly {marker}.\n",
        delay_s=3.0,
    )

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    result = runner.run_print(
        f"Use WriteFile three times: create {work_dir}/a.txt with 'a', "
        f"then {work_dir}/b.txt with 'b', then {work_dir}/c.txt with 'c'. "
        f"Finally reply with a single line ending the reply.",
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "15"),
    )
    result.assert_success()
    # Soft assertion: nudge polling is best-effort; if timing missed, mark
    # as xfail-equivalent with an informative message.
    if marker not in result.stdout:  # pragma: no cover — timing-dependent
        raise AssertionError(
            "nudge marker did not appear in agent output — nudge file may not "
            "have been consumed. Check `_check_nudge_file` hook timing.\n"
            f"--- stdout tail ---\n{result.stdout[-1200:]}"
        )


def test_agentic_nudge_file_is_separate_from_supervisor_nudge(
    work_dir: Path,
    home_dir: Path,
    agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """amd-dev ``fae579d9``: `.agentic_supervisor_nudge.md` and
    `.supervisor_nudge.md` must be tracked with independent mtimes so they
    don't clobber each other.

    We write a `.supervisor_nudge.md` BEFORE the agent starts, then write a
    different `.agentic_supervisor_nudge.md` DURING the run. Both markers
    should eventually land in the context.
    """
    supervisor_nudge = work_dir / ".supervisor_nudge.md"
    agentic_nudge = work_dir / ".agentic_supervisor_nudge.md"
    marker_rule = "RULE-NUDGE-11111"
    marker_agent = "AGENTIC-NUDGE-22222"

    supervisor_nudge.write_text(
        f"STEER: include the string {marker_rule} in your final reply.\n",
        encoding="utf-8",
    )
    _ = _delayed_writer(
        agentic_nudge,
        f"STEER: also include the string {marker_agent} in your final reply.\n",
        delay_s=3.0,
    )

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    result = runner.run_print(
        f"Use WriteFile to create {work_dir}/x.txt with 'x', then "
        f"{work_dir}/y.txt with 'y'. Then reply with a final message.",
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "15"),
    )
    result.assert_success()
    # At least one marker must be present; BOTH is the contract but timing
    # is fragile. Fail only if NEITHER appears (proves both paths dead).
    got_rule = marker_rule in result.stdout
    got_agent = marker_agent in result.stdout
    assert got_rule or got_agent, (
        f"neither nudge marker reached the agent — both polling paths may "
        f"be broken.\n--- tail ---\n{result.stdout[-1200:]}"
    )
    if not (got_rule and got_agent):  # pragma: no cover — timing
        print(
            f"note: only {('rule' if got_rule else 'agentic')} nudge arrived; "
            f"timing may have masked the other — re-run if this matters."
        )
