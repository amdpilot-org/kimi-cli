"""Shell tool `KIMI_MAX_TIMEOUT` env var — amd-dev ``f76da46a``.

The AMD pipeline builds ROCm kernels that can take 10+ minutes. amdpilot
raises the timeout via this env var. Test validates (a) the env var is read,
(b) the tool actually returns before our outer pytest timeout.
"""

from __future__ import annotations

import time
from pathlib import Path

from tests_live.helpers.runner import KimiRunner


def test_shell_timeout_env_is_respected(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """With ``KIMI_MAX_TIMEOUT=3`` a 30s sleep must return a timeout-error result,
    and the overall subprocess finishes well within 30s.

    Note: ``KIMI_MAX_TIMEOUT`` applies per-call. The Shell tool should kill
    the subprocess at 3s, mark it as errored, and return to the agent.
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir,
        extra_env={"KIMI_MAX_TIMEOUT": "3"},
    )
    t0 = time.time()
    result = runner.run_print(
        "Use the Shell tool exactly once: run `sleep 30`. "
        "After the Shell tool returns (it will time out), reply exactly TIMED_OUT and stop.",
        thinking=False,
        timeout=live_timeout,
    )
    elapsed = time.time() - t0
    result.assert_success()
    assert elapsed < 25, (
        f"subprocess took {elapsed:.1f}s — KIMI_MAX_TIMEOUT=3 was not respected. "
        f"stdout tail: {result.stdout[-600:]}"
    )


def test_shell_timeout_defaults_to_large(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """Without ``KIMI_MAX_TIMEOUT`` a 2s sleep completes normally (default is 90min).

    Sanity check that our override doesn't accidentally shorten the default.
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir,
    )
    t0 = time.time()
    result = runner.run_print(
        "Use the Shell tool exactly once: run `sleep 2 && echo SLEEP_OK`. "
        "Then reply exactly DONE.",
        thinking=False,
        timeout=live_timeout,
    )
    elapsed = time.time() - t0
    result.assert_success()
    # 2s sleep must actually take at least 2s
    assert elapsed > 1.5, f"sleep 2 finished in {elapsed:.2f}s — tool not executed"
    assert "SLEEP_OK" in result.stdout, (
        f"expected SLEEP_OK in tool output; tail:\n{result.stdout[-800:]}"
    )
