"""Subprocess runner for `kimi` CLI."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class KimiRunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    def succeeded(self) -> bool:
        return self.returncode == 0

    def assert_success(self) -> None:
        if not self.succeeded():
            raise AssertionError(
                f"kimi exited {self.returncode}\n"
                f"--- STDOUT (last 2k) ---\n{self.stdout[-2000:]}\n"
                f"--- STDERR (last 2k) ---\n{self.stderr[-2000:]}"
            )


@dataclass
class KimiRunner:
    """Launches `kimi` in print mode and collects output.

    Mirrors how ``amdpilot.orchestrator.runtime.kimi_runtime.KimiRuntime``
    invokes kimi inside containers — shell launcher + ``--prompt-file``
    + ``--agent-file`` + ``--yolo`` + ``timeout`` wrapper.
    """

    work_dir: Path
    home_dir: Path
    agent_file: Path | None = None
    skills_dir: Path | None = None
    project_dir: Path | None = None  # path to kimi-cli repo for `uv run --project`
    extra_env: dict[str, str] = field(default_factory=dict)

    def _base_args(self) -> list[str]:
        args: list[str] = ["uv", "run"]
        if self.project_dir:
            args.extend(["--project", str(self.project_dir)])
        args.extend(["--", "kimi"])
        return args

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HOME"] = str(self.home_dir)
        env["USERPROFILE"] = str(self.home_dir)
        # Prevent the live test's provider init from picking up stray auth
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        env.update(self.extra_env)
        return env

    def run_print(
        self,
        prompt: str,
        *,
        timeout: float = 180.0,
        thinking: bool | None = None,
        extra_args: Sequence[str] = (),
    ) -> KimiRunResult:
        """One-shot prompt; wait for kimi to exit; capture stdout+stderr.

        Uses `--print` (non-interactive). Does not need to be on a TTY.
        """
        import time

        args = self._base_args()
        args.extend(["--debug", "--print", "--yolo", "--work-dir", str(self.work_dir)])
        if thinking is True:
            args.append("--thinking")
        elif thinking is False:
            args.append("--no-thinking")
        if self.agent_file:
            args.extend(["--agent-file", str(self.agent_file)])
        if self.skills_dir:
            args.extend(["--skills-dir", str(self.skills_dir)])
        args.extend(extra_args)
        args.extend(["--prompt", prompt])

        t0 = time.time()
        try:
            proc = subprocess.run(
                args,
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return KimiRunResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.time() - t0,
            )
        except subprocess.TimeoutExpired as exc:
            return KimiRunResult(
                returncode=-signal.SIGKILL,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                duration_s=time.time() - t0,
            )

    def run_print_with_prompt_file(
        self,
        prompt_text: str,
        *,
        timeout: float = 180.0,
        thinking: bool | None = None,
        extra_args: Sequence[str] = (),
    ) -> KimiRunResult:
        """Same as run_print but exercises `--prompt-file` (amd-dev patch 5a07c785)."""
        prompt_file = self.work_dir / "prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        args = self._base_args()
        args.extend(["--print", "--yolo", "--work-dir", str(self.work_dir)])
        if thinking is True:
            args.append("--thinking")
        elif thinking is False:
            args.append("--no-thinking")
        if self.agent_file:
            args.extend(["--agent-file", str(self.agent_file)])
        if self.skills_dir:
            args.extend(["--skills-dir", str(self.skills_dir)])
        args.extend(extra_args)
        args.extend(["--prompt-file", str(prompt_file)])

        import time

        t0 = time.time()
        try:
            proc = subprocess.run(
                args,
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return KimiRunResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.time() - t0,
            )
        except subprocess.TimeoutExpired as exc:
            return KimiRunResult(
                returncode=-signal.SIGKILL,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                duration_s=time.time() - t0,
            )
