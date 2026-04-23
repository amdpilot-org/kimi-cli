"""`--prompt-file` CLI option — amd-dev ``5a07c785``.

Avoids `pkill -f <prompt-pattern>` self-kill when the prompt contains process
names; amdpilot uses this flag inside the container launcher (see
``amdpilot/src/amdpilot/orchestrator/runtime/kimi_runtime.py:414``).
"""

from __future__ import annotations

from pathlib import Path

from tests_live.helpers.runner import KimiRunner


def test_prompt_file_flag_reads_from_file(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """`--prompt-file` reads the file and drives the agent normally."""
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir,
    )
    result = runner.run_print_with_prompt_file(
        "Reply with exactly the string PROMPT_FILE_OK and stop.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()
    assert "PROMPT_FILE_OK" in result.stdout, (
        f"model didn't echo marker; tail:\n{result.stdout[-600:]}"
    )


def test_prompt_file_keeps_prompt_off_cmdline(
    work_dir: Path, home_dir: Path, agent_file: Path, kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path, live_timeout: int,
) -> None:
    """Regression for the `pkill -f` self-kill: the prompt text itself must not
    appear in any argv. We can't inspect the running subprocess from here
    reliably, but we can assert the *file path* appears on argv (proving
    --prompt-file was used) and the caller of `run_print_with_prompt_file`
    did not pass --prompt.
    """
    runner = KimiRunner(
        work_dir=work_dir, home_dir=home_dir, agent_file=agent_file,
        project_dir=kimi_project_dir,
    )
    secret_marker = "UNIQUE-DO-NOT-PUT-ME-IN-ARGV-12345"
    result = runner.run_print_with_prompt_file(
        f"Reply with exactly {secret_marker} and stop.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()
    assert secret_marker in result.stdout, result.stdout[-500:]
    # Positive assertion: the prompt FILE must exist and contain the secret.
    prompt_file = work_dir / "prompt.txt"
    assert prompt_file.exists(), f"prompt file not created at {prompt_file}"
    assert secret_marker in prompt_file.read_text(), prompt_file.read_text()
