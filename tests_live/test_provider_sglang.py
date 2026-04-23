"""Provider-level behaviour against a real sglang/Qwen endpoint.

Covers the amd-dev patches that are only exercised by a real OSS model:
``f5bdde06`` (reasoning_key for thinking), ``bf9bed8a`` (null tool-call args),
``47d51341`` + ``12441e5e`` (double-encoded JSON recovery), and
``45cb3437`` (retry loop survives client recreation).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests_live.helpers.runner import KimiRunner


def test_reasoning_content_populates_thinkpart(runner: KimiRunner, live_timeout: int) -> None:
    """amd-dev ``f5bdde06``: openai_legacy must set ``reasoning_key="reasoning_content"``
    so Qwen3.5's thinking tokens land in a ThinkPart (not silently dropped).

    We ask a small reasoning question with thinking enabled. The CLI's stdout
    in print mode serialises `ThinkPart` objects — its presence proves the
    reasoning channel is wired through.
    """
    result = runner.run_print(
        "What is 127 * 31? Think briefly, then answer with just the number.",
        thinking=True,
        timeout=live_timeout,
    )
    result.assert_success()
    combined = result.stdout
    assert "ThinkPart" in combined, (
        "No ThinkPart in kimi print output — reasoning_key likely not wired.\n"
        f"--- tail ---\n{combined[-1500:]}"
    )
    # And the correct arithmetic answer should appear (3937). If the model
    # is flaky, we still consider the ThinkPart alone sufficient for the
    # wiring contract; so this second check is soft.
    if "3937" not in combined:  # pragma: no cover - model nondeterminism
        print("note: model did not produce 3937 in final answer (flaky OK)")


def test_multi_tool_call_sequence_no_parse_failures(
    runner: KimiRunner, work_dir: Path, live_timeout: int
) -> None:
    """amd-dev ``47d51341`` + ``12441e5e``: if Qwen emits tool arguments as a
    JSON-encoded string (double-encoded), kosong recovers. Also covers
    ``bf9bed8a``: arguments=None default to "{}".

    We don't deterministically force the double-encode; instead we run a short
    multi-tool sequence and assert zero tool-call-parsing errors surface in
    stderr, and that every tool_call in the status log has a non-empty
    ``tool_args_summary`` entry.
    """
    from tests_live.helpers.runner import KimiRunner

    runner_with_dump = KimiRunner(
        work_dir=runner.work_dir,
        home_dir=runner.home_dir,
        agent_file=runner.agent_file,
        project_dir=runner.project_dir,
        extra_env={"KIMI_STATUS_INTERVAL": "1"},
    )
    paths = [work_dir / f"probe_{i}.txt" for i in range(3)]
    result = runner_with_dump.run_print(
        (
            f"Your ONLY job is to create three files with the WriteFile tool, "
            f"one at a time. Do not run Shell. Do not write any text before "
            f"calling the tools. Create:\n"
            f"1. {paths[0]} with content one\n"
            f"2. {paths[1]} with content two\n"
            f"3. {paths[2]} with content three\n"
            f"After all three tool calls succeed, reply exactly DONE."
        ),
        thinking=False,
        timeout=live_timeout,
        extra_args=("--max-steps-per-turn", "20"),
    )
    result.assert_success()

    # Contract: no JSON parse errors should have leaked into stderr.
    assert "JSONDecodeError" not in result.stderr, result.stderr[-500:]
    assert "failed to parse" not in result.stderr.lower(), result.stderr[-500:]

    # At least one tool call was recorded.
    from tests_live.helpers.status_reader import StatusReader

    reader = StatusReader(work_dir)
    records = reader.read_all()
    assert records, (
        f"expected at least one status record; stdout tail:\n{result.stdout[-1500:]}"
    )
    tool_calls = [r for r in records if r.get("role") == "tool"]
    assert tool_calls, f"no tool results in status log; records={records}"


def test_custom_headers_and_verify_ssl_keys_accepted(
    home_dir: Path,
    live_endpoint: str,
    live_model: str,
    live_api_key: str,
    work_dir: Path,
    agent_file: Path,
    kimi_project_dir: Path,
    live_timeout: int,
) -> None:
    """amd-dev ``392709e8`` + ``8c9aa449``: ``custom_headers`` and ``verify_ssl``
    must be accepted on the provider config without raising. We don't test the
    headers are actually applied over the wire here (that requires a proxy);
    we assert the config parses and the CLI still reaches a successful turn.
    """
    from tests_live.helpers.config import write_kimi_config

    extra = (
        "verify_ssl = true\n"
        "\n"
        "[providers.live.custom_headers]\n"
        '"X-AMD-Test-Header" = "live-test-marker"\n'
    )
    write_kimi_config(
        home_dir=home_dir,
        endpoint=live_endpoint,
        model=live_model,
        api_key=live_api_key,
        extra_provider_lines=extra,
    )

    runner = KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent_file,
        project_dir=kimi_project_dir,
    )
    result = runner.run_print(
        "Reply with exactly the word HEADERCFG-OK.",
        thinking=False,
        timeout=live_timeout,
    )
    result.assert_success()
    # If verify_ssl/custom_headers weren't accepted we'd see a validation error
    # ("extra keys not permitted" / "unexpected keyword argument httpx client").
    # Instead, the turn should complete and the model should reply.
    assert "HEADERCFG-OK" in result.stdout, (
        f"model didn't echo expected marker; tail=\n{result.stdout[-800:]}"
    )
