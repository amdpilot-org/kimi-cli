import asyncio
import os as _os
import re
from collections.abc import Callable
from pathlib import Path
from typing import override

import kaos
from kaos import AsyncReadable
from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import ShellDisplayBlock
from kimi_cli.tools.utils import ToolRejectedError, ToolResultBuilder, load_desc
from kimi_cli.utils.environment import Environment
from kimi_cli.utils.subprocess_env import get_clean_env

MAX_TIMEOUT = int(_os.environ.get("KIMI_SHELL_MAX_TIMEOUT", 5 * 60))


# ---------------------------------------------------------------------------
# Post-mortem log enrichment
# ---------------------------------------------------------------------------
#
# When an agent runs a command that interacts with a long-running server or
# benchmark process and the command FAILS, the agent's tool result usually
# does not contain the actual root cause -- the device-side fault, the
# scheduler subprocess crash, the SIGABRT -- because that signature lives
# in the SERVER's own log file, not in stdout/stderr of the agent's shell
# command.  Without seeing the signature, the agent commonly mis-diagnoses
# the failure as a transient launch issue and relaunches the server in a
# loop, never converging on the real bug.
#
# Heuristic:
#   1. The command must look like it interacts with a server / bench
#      subprocess.  We match a small set of clearly-identifiable shapes:
#        * `python3 -m {sglang,vllm}.<entrypoint>` (server / bench / serve)
#        * `bench_one_batch_server` / `bench_one_batch`
#        * `curl http://127.0.0.1:<port>` or `curl http://localhost:<port>`
#      We deliberately reject patterns that would match unrelated
#      `python3 -c '...'` or `ls /missing` -- those are NOT server work.
#   2. The result must look like a failure: non-zero exit, OR stdout/stderr
#      contains a known transport-level marker (`Connection refused`,
#      `Empty reply from server`, `Failed to connect`, `connection reset`).
#
# When both fire, we tail the standard server-log paths and append a
# clearly-marked "[POSTMORTEM]" block to the tool result.  The block is
# capped to a few KB so it doesn't blow the agent's context.

_POSTMORTEM_HEADER_BEGIN = "\n[POSTMORTEM] server/bench command failed; tailing common log paths\n"
_POSTMORTEM_HEADER_END = "[POSTMORTEM] end\n"
_POSTMORTEM_MAX_TOTAL_CHARS = 4000
_POSTMORTEM_TAIL_LINES = 30

# Files we tail.  Each entry is checked relative to cwd AND as an absolute
# path (so tests that pre-seed `output/server.log` in a tempdir AND prod
# containers that write to `/workspace/output/server.log` both work).
_POSTMORTEM_LOG_PATHS = (
    "output/server.log",
    "output/bench_result.log",
    "output/full_run.log",
    "output/prewarm.log",
    "/workspace/output/server.log",
    "/workspace/output/bench_result.log",
    "/workspace/output/full_run.log",
    "/workspace/server.log",
    "/tmp/sglang_server.log",
    "/tmp/server.log",
)


_SERVER_COMMAND_PATTERN = re.compile(
    r"\b(?:"
    r"python\d?\s+-m\s+(?:sglang|vllm)\."  # python -m sglang.* / vllm.*
    r"(?:launch_server|bench_one_batch_server|bench_one_batch|"
    r"entrypoints\.openai\.api_server|serve)"
    r"|"
    r"\bsglang\s+serve\b"  # `sglang serve` alias
    r"|"
    r"curl\s+(?:[^|]*\s+)?https?://(?:127\.0\.0\.1|localhost)(?::\d{2,5})?/"
    r")",
    re.IGNORECASE,
)

_FAILURE_MARKER_PATTERN = re.compile(
    r"(?:"
    r"curl:\s*\(\d+\)"  # any curl error: (7) Failed to connect, (52) Empty reply, ...
    r"|Connection refused"
    r"|Empty reply from server"
    r"|Failed to connect"
    r"|connection reset by peer"
    r"|address already in use"
    r")",
    re.IGNORECASE,
)


def _looks_like_server_command(command: str) -> bool:
    """Return True if the command interacts with a server/bench process."""
    if not command:
        return False
    return _SERVER_COMMAND_PATTERN.search(command) is not None


def _looks_like_failure(exitcode: int, output: str) -> bool:
    """Return True if the result indicates failure (exit !=0 OR a known
    transport-level marker in the output, even when curl returns exit 0
    via shell pipelines that swallow it)."""
    if exitcode != 0:
        return True
    return bool(output and _FAILURE_MARKER_PATTERN.search(output))


async def _collect_postmortem(
    *,
    shell_path: str,
    log_paths: tuple[str, ...] = _POSTMORTEM_LOG_PATHS,
    tail_lines: int = _POSTMORTEM_TAIL_LINES,
    max_total_chars: int = _POSTMORTEM_MAX_TOTAL_CHARS,
    timeout: float = 5.0,
) -> str:
    """Collect a tail of common server-log paths.

    Implementation note: we shell out via the same shell binary the Shell
    tool already uses -- this avoids any divergence between which paths
    the tool sees (e.g. inside a docker container vs. on the host) and
    keeps the path-expansion logic identical.
    """
    paths_arg = " ".join(f'"{p}"' for p in log_paths)
    # Per-path: print a header iff the file exists, then tail it.  We
    # cap each tail's output via `tail -n N`.
    cmd = (
        f"for f in {paths_arg}; do "
        f'  if [ -f "$f" ] && [ -s "$f" ]; then '
        f'    echo "==> $f <=="; '
        f'    tail -n {tail_lines} "$f"; '
        f"  fi; "
        f"done"
    )
    try:
        process = await kaos.exec(shell_path, "-c", cmd, env=get_clean_env())
        stdout = bytearray()
        stderr = bytearray()

        async def _drain(stream: AsyncReadable, sink: bytearray):
            while True:
                line = await stream.readline()
                if not line:
                    break
                sink.extend(line)
                if len(sink) >= max_total_chars * 2:  # generous early-stop
                    break

        await asyncio.wait_for(
            asyncio.gather(
                _drain(process.stdout, stdout),
                _drain(process.stderr, stderr),
            ),
            timeout,
        )
        await process.wait()
    except Exception:
        return ""

    text = stdout.decode("utf-8", errors="replace")
    if not text.strip():
        return ""
    if len(text) > max_total_chars:
        # Keep the TAIL of the post-mortem (most recent log lines) when
        # truncating -- crash signatures are at the bottom of server logs.
        text = "[...truncated head...]\n" + text[-max_total_chars:]
    return text


class Params(BaseModel):
    command: str = Field(description="The bash command to execute.")
    timeout: int = Field(
        description=(
            "The timeout in seconds for the command to execute. "
            "If the command takes longer than this, it will be killed."
        ),
        default=60,
        ge=1,
        le=MAX_TIMEOUT,
    )


class Shell(CallableTool2[Params]):
    name: str = "Shell"
    params: type[Params] = Params

    def __init__(self, approval: Approval, environment: Environment):
        is_powershell = environment.shell_name == "Windows PowerShell"
        super().__init__(
            description=load_desc(
                Path(__file__).parent / ("powershell.md" if is_powershell else "bash.md"),
                {"SHELL": f"{environment.shell_name} (`{environment.shell_path}`)"},
            )
        )
        self._approval = approval
        self._is_powershell = is_powershell
        self._shell_path = environment.shell_path

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder()

        if not params.command:
            return builder.error("Command cannot be empty.", brief="Empty command")

        if not await self._approval.request(
            self.name,
            "run command",
            f"Run command `{params.command}`",
            display=[
                ShellDisplayBlock(
                    language="powershell" if self._is_powershell else "bash",
                    command=params.command,
                )
            ],
        ):
            return ToolRejectedError()

        def stdout_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)

        def stderr_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)

        try:
            exitcode = await self._run_shell_command(
                params.command, stdout_cb, stderr_cb, params.timeout
            )

            # Post-mortem enrichment: when an agent's server- or bench-
            # touching command FAILS, automatically tail the server's
            # own log file and append it to the tool result so the
            # agent's NEXT reasoning step sees the device-side fault
            # signature instead of a generic ``Connection refused`` or
            # ``Empty reply from server``.
            current_output = "".join(builder._buffer)  # type: ignore[attr-defined]
            if (
                not self._is_powershell
                and _looks_like_server_command(params.command)
                and _looks_like_failure(exitcode, current_output)
            ):
                postmortem = await _collect_postmortem(shell_path=str(self._shell_path))
                if postmortem:
                    builder.write(_POSTMORTEM_HEADER_BEGIN)
                    builder.write(postmortem)
                    if not postmortem.endswith("\n"):
                        builder.write("\n")
                    builder.write(_POSTMORTEM_HEADER_END)

            if exitcode == 0:
                return builder.ok("Command executed successfully.")
            else:
                return builder.error(
                    f"Command failed with exit code: {exitcode}.",
                    brief=f"Failed with exit code: {exitcode}",
                )
        except TimeoutError:
            return builder.error(
                f"Command killed by timeout ({params.timeout}s)",
                brief=f"Killed by timeout ({params.timeout}s)",
            )

    async def _run_shell_command(
        self,
        command: str,
        stdout_cb: Callable[[bytes], None],
        stderr_cb: Callable[[bytes], None],
        timeout: int,
    ) -> int:
        async def _read_stream(stream: AsyncReadable, cb: Callable[[bytes], None]):
            while True:
                line = await stream.readline()
                if line:
                    cb(line)
                else:
                    break

        process = await kaos.exec(*self._shell_args(command), env=get_clean_env())

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(process.stdout, stdout_cb),
                    _read_stream(process.stderr, stderr_cb),
                ),
                timeout,
            )
            return await process.wait()
        except TimeoutError:
            await process.kill()
            raise

    def _shell_args(self, command: str) -> tuple[str, ...]:
        if self._is_powershell:
            return (str(self._shell_path), "-command", command)
        return (str(self._shell_path), "-c", command)
