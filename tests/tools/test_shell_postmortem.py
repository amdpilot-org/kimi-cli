"""Tests for the Shell tool's post-mortem log enrichment.

When an agent runs a command that interacts with a long-running
*server* or *benchmark* process (sglang/vllm `launch_server`,
`bench_one_batch_server`, or a `curl` against a localhost serving
port) and that command fails, the agent often does NOT see the
actual crash signature.  The signature lives in the server's own
log file (e.g. ``/workspace/output/server.log``) — but the agent's
Shell tool only returns whatever the failed command itself printed,
which is typically a one-line ``curl: (52) Empty reply from server``
or a Python traceback that does not mention the GPU-side fault.

Concrete observation that motivated this enrichment (sglang-17,
2026-05-06): across 5 trials the agent never once saw any of:

  * ``GPU core dump failed``
  * ``Subprocess scheduler_N (pid=...) crashed with exit code -6``
  * ``SIGQUIT received``
  * ``process_batch_result_decode``

…in its own tool outputs, even though every server crash in those
trials produced exactly that signature in
``/workspace/output/server.log``.  The agent kept relaunching the
server because it could not see the device fault that was killing
its child workers.

The fix is **post-mortem enrichment**: when a Shell call's command
matches a server/benchmark heuristic AND the result indicates
failure (non-zero exit, ``Connection refused``, ``Empty reply``,
``Failed to connect``), the Shell tool tails common server-log
paths and appends them to the tool result so the agent's NEXT
reasoning step sees the device-side signature directly.

These tests pin the contract.  They do NOT exercise a real sglang
server — instead they pre-seed a fake ``/workspace/output/server.log``
in the test working directory and run a command that emulates each
failure mode.

Run::

  cd kimi-cli && uv run python -m pytest tests/tools/test_shell_postmortem.py -v
"""

from __future__ import annotations

import platform

import pytest
from kaos.path import KaosPath

from kimi_cli.tools.shell import Params, Shell

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Bash-tests run only on non-Windows.",
)


_FAKE_SERVER_CRASH = """\
[2026-05-06 19:11:22] INFO:     Started server process [40984]
[2026-05-06 19:11:22] The server is fired up and ready to roll!
[2026-05-06 19:11:47 TP0] Prefill batch, #new-seq: 1, #new-token: 256, ...
GPU core dump failed
GPU core dump failed
[2026-05-06 19:12:07] Subprocess scheduler_3 (pid=22856) crashed with exit code -6.
[2026-05-06 19:12:07] SIGQUIT received. signum=None, frame=None.
"""


async def _seed_server_log(work_dir: KaosPath, content: str = _FAKE_SERVER_CRASH) -> str:
    """Seed a ``output/server.log`` under the test work dir.  The
    post-mortem helper looks for the file by RELATIVE path and by the
    canonical ``/workspace/output/server.log`` location — but in tests
    we run with cwd=temp_work_dir so the relative lookup matches.
    KaosPath's ``mkdir`` / ``write_text`` are async."""
    output_dir = work_dir / "output"
    await output_dir.mkdir(exist_ok=True)
    log = output_dir / "server.log"
    await log.write_text(content)
    return str(log)


# ---------------------------------------------------------------------------
# Trigger detection: does the heuristic fire on the right command shapes?
# ---------------------------------------------------------------------------


class TestShellPostMortemTriggers:
    """The post-mortem MUST trigger when the command interacts with a
    long-running server/bench process and the result indicates failure.
    It MUST NOT trigger on unrelated failures (e.g. ``ls /nonexistent``)."""

    @pytest.mark.parametrize(
        "command,expect_trigger",
        [
            # Should trigger — server/bench commands hitting a dead port:
            (
                "curl -sS http://127.0.0.1:31050/health",
                True,
            ),
            (
                "curl -sS -X POST http://127.0.0.1:31050/generate -d '{}'",
                True,
            ),
            (
                "/opt/venv/bin/python3 -m sglang.bench_one_batch_server "
                "--base-url http://127.0.0.1:31050 --batch-size 1",
                True,
            ),
            (
                "python3 -m sglang.launch_server --port 31050",
                True,
            ),
            (
                "python3 -m vllm.entrypoints.openai.api_server --port 8000",
                True,
            ),
            # Should NOT trigger — generic failures unrelated to a server:
            ("ls /nonexistent/directory", False),
            ("cat /etc/no-such-file", False),
            ("python3 -c 'raise SystemExit(1)'", False),
        ],
    )
    async def test_trigger_heuristic(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
        command: str,
        expect_trigger: bool,
    ):
        # Make sure there's a server log to surface — the trigger should
        # fire whether or not the log exists; we want to verify ATTACHMENT
        # below.  Here we verify the command-shape heuristic only.
        await _seed_server_log(temp_work_dir)
        result = await shell_tool(Params(command=command, timeout=10))
        # All these commands fail (port is dead, file missing, etc.).
        assert result.is_error
        out = result.output
        # Trigger leaves a header in the output:
        has_postmortem = "[POSTMORTEM]" in out or "POSTMORTEM" in out
        assert has_postmortem == expect_trigger, (
            f"command {command!r}: expected trigger={expect_trigger}, "
            f"got has_postmortem={has_postmortem}.  Output:\n{out[:500]}"
        )


# ---------------------------------------------------------------------------
# Content: does the post-mortem actually surface the device-fault signature?
# ---------------------------------------------------------------------------


class TestShellPostMortemContent:
    async def test_post_mortem_includes_server_log_tail(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
    ):
        """The seeded ``output/server.log`` lines (`GPU core dump failed`,
        `crashed with exit code -6`, `SIGQUIT received`) must appear in
        the tool result so the next reasoning step sees them."""
        await _seed_server_log(temp_work_dir)
        result = await shell_tool(
            Params(
                command="curl -sS -m 2 http://127.0.0.1:31050/generate -X POST -d '{}'",
                timeout=10,
            )
        )
        assert result.is_error
        out = result.output

        # Pin all three signature substrings — these are the strings the
        # agent failed to see in trial 4-5.
        for needle in (
            "GPU core dump failed",
            "scheduler_3",
            "exit code -6",
            "SIGQUIT received",
        ):
            assert needle in out, (
                f"post-mortem missing required signature {needle!r}.\nOutput:\n{out[:1500]}"
            )

    async def test_post_mortem_marks_log_path(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
    ):
        """The post-mortem block must annotate which log path the tail
        came from, so the agent can run a follow-up ``cat <path>`` if
        it wants more context."""
        await _seed_server_log(temp_work_dir)
        result = await shell_tool(
            Params(
                command="curl -sS -m 2 http://127.0.0.1:31050/health",
                timeout=10,
            )
        )
        assert result.is_error
        out = result.output
        # Should mention the path it tailed (server.log).
        assert "server.log" in out, (
            f"post-mortem should annotate the log path in its block.\nOutput:\n{out[:1500]}"
        )

    async def test_post_mortem_header_format(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
    ):
        """The block must be syntactically distinguishable from the
        agent's own command output (so a future agent reading the
        wire log can tell what came from where)."""
        await _seed_server_log(temp_work_dir)
        result = await shell_tool(
            Params(
                command="curl -sS -m 2 http://127.0.0.1:31050/health",
                timeout=10,
            )
        )
        assert result.is_error
        out = result.output
        # Header must start with [POSTMORTEM] or === POSTMORTEM === or similar.
        assert any(tag in out for tag in ("[POSTMORTEM]", "=== POSTMORTEM ===")), (
            f"post-mortem must be wrapped in a clear header.\nOutput:\n{out[:1500]}"
        )


# ---------------------------------------------------------------------------
# Negative path: don't fire on success.
# ---------------------------------------------------------------------------


class TestShellPostMortemNoFalsePositives:
    async def test_successful_curl_does_not_trigger(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
    ):
        """When a server-touching command SUCCEEDS, no post-mortem
        should fire (otherwise we spam the agent with stale logs)."""
        await _seed_server_log(temp_work_dir)
        # Invoke a 'server-touching' shape but make it succeed.
        result = await shell_tool(Params(command="echo 'launch_server returned 0'", timeout=5))
        assert not result.is_error
        out = result.output
        assert "POSTMORTEM" not in out, (
            f"post-mortem fired on a successful command — false positive.\nOutput:\n{out[:500]}"
        )

    async def test_unrelated_failure_does_not_trigger(
        self,
        shell_tool: Shell,
        temp_work_dir: KaosPath,
    ):
        await _seed_server_log(temp_work_dir)
        result = await shell_tool(Params(command="ls /this/path/does/not/exist", timeout=5))
        assert result.is_error
        assert "POSTMORTEM" not in result.output, (
            f"post-mortem fired on `ls` failure — too eager.\nOutput:\n{result.output[:500]}"
        )
