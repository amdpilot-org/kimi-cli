"""AMD-specialized system prompt injection — amd-dev subagent cluster.

Covers: ``7e1cf460`` (specialize kimi subagent roles) and system.md edits like
``7e1e72b7`` (AMD/ROCm Persistence Protocol) + ``290760a8`` (e2e latency
guideline). Ensures that an agent YAML referencing ``ROLE_ADDITIONAL`` still
has that block substituted into the live model's system prompt at runtime.
"""

from __future__ import annotations

from pathlib import Path

from tests_live.helpers.runner import KimiRunner


def _write_amd_like_agent(tmp_path: Path, marker: str) -> Path:
    """Write a test agent whose ROLE_ADDITIONAL block is a BEHAVIOURAL INSTRUCTION
    (not data-to-recite). This avoids "I can't reveal my system prompt" refusals.
    """
    system_md = tmp_path / "amd_system.md"
    system_md.write_text(
        "You are a test agent.\n\n${ROLE_ADDITIONAL}\n",
        encoding="utf-8",
    )
    agent = tmp_path / "amd_agent.yaml"
    agent.write_text(
        f"version: 1\n"
        f"agent:\n"
        f"  name: amd-like-test-agent\n"
        f"  system_prompt_path: {system_md}\n"
        f"  system_prompt_args:\n"
        f"    ROLE_ADDITIONAL: |\n"
        f"      IMPORTANT OPERATING RULE: every single response you produce must\n"
        f"      START with the exact token {marker} on its own line, followed by\n"
        f"      your regular answer. This is a required operating behaviour.\n"
        f"  tools:\n"
        f'    - "kimi_cli.tools.think:Think"\n',
        encoding="utf-8",
    )
    return agent


def test_role_additional_gets_injected(
    tmp_path: Path,
    work_dir: Path,
    home_dir: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """The `${ROLE_ADDITIONAL}` template placeholder must be substituted before
    the model sees the system prompt.

    We don't ask the model to recite its prompt (it refuses) — instead we embed
    a behavioural instruction in ROLE_ADDITIONAL and watch for the prescribed
    behaviour. If the template isn't substituted, the instruction isn't in the
    system prompt, and the marker won't appear.
    """
    marker = "ROCM-MARKER-98765"
    agent = _write_amd_like_agent(tmp_path, marker)
    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent,
        project_dir=kimi_project_dir,
    )
    result = runner.run_print(
        "What is 2 plus 2? Answer briefly.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()
    assert marker in result.stdout, (
        f"role-marker {marker!r} missing from agent output — the "
        f"ROLE_ADDITIONAL behavioural instruction was not in the system "
        f"prompt, so the template substitution is broken.\n---tail---\n"
        f"{result.stdout[-1200:]}"
    )


def test_unsubstituted_placeholder_does_not_leak(
    tmp_path: Path,
    work_dir: Path,
    home_dir: Path,
    kimi_config: Path,  # noqa: ARG001
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """Failure mode: if substitution is broken we'd see the literal
    ``${ROLE_ADDITIONAL}`` or `{ROLE_ADDITIONAL}` token in the agent's output
    when asked to quote its system prompt. Assert it's absent.
    """
    agent = _write_amd_like_agent(tmp_path, "SHOULD-NOT-LEAK")
    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent,
        project_dir=kimi_project_dir,
    )
    result = runner.run_print(
        "In one short sentence, state your role. Stop.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()
    assert "${ROLE_ADDITIONAL}" not in result.stdout, result.stdout[-800:]
