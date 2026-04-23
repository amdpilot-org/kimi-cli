"""Contract tests for `.agent_status.jsonl` — the amdpilot nudge-agent parser.

Consumer contract lives in:
- `amdpilot/src/amdpilot/orchestrator/execution_monitor.py:240-260`
- `amdpilot/src/amdpilot/orchestrator/runtime/kimi_runtime.py:438-479`

Producer patches: ``725ee9bb`` (append-only), ``717d5e6e`` (cursor reset),
``33089173`` (event buffer decoupled from history), ``fdfcdeed`` (content_tail
+ tool_args_summary), ``2ba264f1`` (interval 30 → 5).
"""

from __future__ import annotations

from pathlib import Path

from tests_live.helpers.runner import KimiRunner
from tests_live.helpers.status_reader import StatusReader


def _run_tool_using_turn(runner: KimiRunner, work_dir: Path, live_timeout: int) -> StatusReader:
    """Helper: run a prompt that always invokes at least one tool, return the reader."""
    probe = work_dir / "contract-probe.txt"
    res = runner.run_print(
        f"TASK: Use the WriteFile tool to overwrite {probe} with exactly STATUS-PROBE. "
        f"You MUST call WriteFile — do not reply in text first. "
        f"After the tool returns, reply exactly DONE.",
        thinking=False,
        timeout=live_timeout,
    )
    res.assert_success()
    return StatusReader(work_dir)


def test_contract_required_fields_present(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """Every record has `step` and `role`."""
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir, extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    reader = _run_tool_using_turn(runner, work_dir, live_timeout)
    reader.assert_contract_fields_present()


def test_contract_tool_call_shape(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """Assistant records with tool calls must carry `tool_calls` (list[str])
    and `tool_args_summary` (list[str]). Amdpilot parser reads both
    (`execution_monitor.py:240-260`).
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir, extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    reader = _run_tool_using_turn(runner, work_dir, live_timeout)
    records = reader.read_all()
    assistant_records = [
        r for r in records if r.get("role") == "assistant" and r.get("tool_calls")
    ]
    assert assistant_records, f"no assistant record with tool_calls; records={records}"
    first = assistant_records[0]
    assert isinstance(first["tool_calls"], list), first
    assert all(isinstance(t, str) for t in first["tool_calls"]), first
    assert "tool_args_summary" in first, f"missing tool_args_summary: {first}"
    assert isinstance(first["tool_args_summary"], list), first


def test_contract_seq_monotonic(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """`seq` (monotonic sequence from amd-dev event buffer redesign, 33089173)
    must be strictly increasing across the file.
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir, extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    reader = _run_tool_using_turn(runner, work_dir, live_timeout)
    records = reader.read_all()
    seqs = [r.get("seq") for r in records if "seq" in r]
    assert seqs, "no records carry `seq`"
    assert all(isinstance(s, int) for s in seqs), seqs
    assert seqs == sorted(seqs), f"seq not monotonically increasing: {seqs}"
    # And no duplicates
    assert len(seqs) == len(set(seqs)), f"seq has duplicates: {seqs}"


def test_contract_content_preview_truncation(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """`content_preview` is bounded (amdpilot expects ~500 char cap)."""
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir, extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    # Ask for a longer response by having the agent write and then echo a file.
    probe = work_dir / "content-preview-probe.txt"
    body = "A" * 2000
    res = runner.run_print(
        f"First WriteFile to {probe} with exactly this content: {body!r}. "
        f"Then reply DONE.",
        thinking=False,
        timeout=live_timeout,
    )
    res.assert_success()
    reader = StatusReader(work_dir)
    records = reader.read_all()
    for r in records:
        preview = r.get("content_preview", "")
        assert isinstance(preview, str), r
        # Our amd-dev dump code caps at 500 chars for the preview.
        assert len(preview) <= 512, (
            f"content_preview exceeds 512 chars ({len(preview)}): {r}"
        )


def test_contract_interval_env_respected(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """`KIMI_STATUS_INTERVAL=0` disables dumps (amd-dev ``725ee9bb`` explicit opt-out).

    amdpilot sets this when it does not want the nudge file to be rewritten
    (e.g., non-nudge deployments).
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir, extra_env={"KIMI_STATUS_INTERVAL": "0"},
    )
    _run_tool_using_turn(runner, work_dir, live_timeout)
    reader = StatusReader(work_dir)
    # With interval=0, the file should either not exist or be empty.
    if reader.exists():
        assert reader.read_all() == [], (
            f"KIMI_STATUS_INTERVAL=0 but file has content: {reader.read_all()}"
        )
