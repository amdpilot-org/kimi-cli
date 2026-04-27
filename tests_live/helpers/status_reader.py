"""Tail and parse `.agent_status.jsonl` (the producer-side event buffer)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Fields the amdpilot ExecutionMonitor + KimiRuntime parser depends on.
# Changes to this set are a contract break.
STATUS_REQUIRED_FIELDS = {"step", "role"}
STATUS_OPTIONAL_FIELDS = {
    "content_preview",
    "content_tail",
    "tool_calls",
    "tool_args_summary",
    "seq",
}


class StatusReader:
    """Read the `.agent_status.jsonl` file written by kimisoul.py.

    Mirrors the parsing contract in:
    - kimi-cli amd-dev `src/kimi_cli/soul/kimisoul.py` (producer)
    - amdpilot `src/amdpilot/orchestrator/execution_monitor.py:240-260` (consumer)
    - amdpilot `src/amdpilot/orchestrator/runtime/kimi_runtime.py:438-479` (consumer)
    """

    def __init__(self, work_dir: Path) -> None:
        self.path = Path(work_dir) / ".agent_status.jsonl"

    def exists(self) -> bool:
        return self.path.exists()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
    ) -> dict[str, Any]:
        """Poll until a record matching *predicate* appears."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            for record in self.read_all():
                if predicate(record):
                    return record
            time.sleep(poll_interval)
        raise TimeoutError(
            f"No status record matched predicate within {timeout}s. "
            f"File={self.path}; exists={self.path.exists()}; "
            f"records={len(self.read_all())}"
        )

    def wait_nonempty(self, *, timeout: float = 60.0, poll_interval: float = 0.5) -> None:
        """Wait until at least one record is written."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.read_all():
                return
            time.sleep(poll_interval)
        raise TimeoutError(f"`.agent_status.jsonl` never got a record at {self.path}")

    def assert_contract_fields_present(self) -> None:
        """Raise AssertionError if any record is missing required fields."""
        records = self.read_all()
        assert records, f"no records at {self.path}"
        for i, rec in enumerate(records):
            missing = STATUS_REQUIRED_FIELDS - rec.keys()
            assert not missing, f"record #{i} missing required fields {missing}: {rec}"

    def collect_roles(self) -> list[str]:
        return [r.get("role", "") for r in self.read_all()]

    def has_tool_call(self, tool_name: str) -> bool:
        """Return True if any record's `tool_calls` list contains *tool_name*."""
        for rec in self.read_all():
            tc = rec.get("tool_calls") or []
            if isinstance(tc, list) and any(tool_name in str(t) for t in tc):
                return True
            # tool_args_summary is a list of dicts; scan names
            tas = rec.get("tool_args_summary") or []
            if isinstance(tas, list):
                for item in tas:
                    if isinstance(item, str) and tool_name in item:
                        return True
                    if isinstance(item, dict) and tool_name in str(item):
                        return True
        return False
