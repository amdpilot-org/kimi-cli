"""Canary smoke test — proves the live-test harness itself works.

If this test fails, nothing else in tests_live/ should be trusted.
"""

from __future__ import annotations

from pathlib import Path

from tests_live.helpers.runner import KimiRunner
from tests_live.helpers.status_reader import StatusReader


def test_smoke_kimi_connects_and_exits_cleanly(runner: KimiRunner, live_timeout: int) -> None:
    """Minimal end-to-end: kimi launches, connects to sglang, answers, exits 0."""
    result = runner.run_print(
        "Reply with exactly the word PONG. Do not use any tools.",
        thinking=True,
        timeout=live_timeout,
    )
    result.assert_success()
    combined = result.stdout + result.stderr
    assert "PONG" in combined, f"'PONG' not in output; got:\n{combined[-800:]}"


def test_smoke_agent_status_file_gets_written(
    work_dir: Path,
    home_dir: Path,
    agent_file: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """Force per-step dumps and assert the file exists + has required fields.

    Validates the producer side of the `.agent_status.jsonl` contract that
    amdpilot's ExecutionMonitor consumes. Corresponds to amd-dev commits
    ``725ee9bb`` (append-only producer), ``fdfcdeed`` (content_tail +
    tool_args_summary), and ``2ba264f1`` (interval).
    """
    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    # Force at least one tool use — status dump only fires at end-of-step,
    # and early-return on `no_tool_calls` skips the dump code path.
    # Be extra insistent with OSS models that sometimes short-circuit.
    result = runner.run_print(
        "TASK: Write the file /tmp/smoke-status-probe.txt containing exactly "
        "SMOKE-OK using the WriteFile tool. You MUST call WriteFile. "
        "Do not reply in text before calling the tool. "
        "After the tool returns, reply exactly DONE and stop.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()

    reader = StatusReader(work_dir)
    assert reader.exists(), f"`.agent_status.jsonl` not at {reader.path}"
    records = reader.read_all()
    assert records, (
        f"file exists but is empty\n"
        f"--- STDOUT (last 1500) ---\n{result.stdout[-1500:]}\n"
        f"--- STDERR (last 500) ---\n{result.stderr[-500:]}"
    )
    # Every record must have step+role (our minimum contract).
    reader.assert_contract_fields_present()
    # At least one record's role should be a real producer role.
    roles = {r.get("role") for r in records}
    assert roles & {"assistant", "user", "system", "tool"}, (
        f"no recognisable roles in status dump; roles seen: {roles}"
    )
