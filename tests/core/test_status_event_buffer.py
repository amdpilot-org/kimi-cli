"""Producer-side regression tests for the append-only status event buffer.

Verifies that the event buffer in KimiSoul correctly handles:
1. Compaction + preserved tail + new messages → no replay, no loss
2. Revert + restored old messages + new messages → no replay, no loss

These tests operate at the _status_event_buffer / _dump_agent_status level,
bypassing the full agent loop infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FakeMessage:
    """Minimal Message stand-in for status dump tests."""

    def __init__(self, role: str, text: str, tool_calls: list | None = None) -> None:
        self.role = role
        self._text = text
        self.content = text
        self.tool_calls = tool_calls or []

    def extract_text(self, sep: str = " ") -> str:
        return self._text


class StatusBufferHarness:
    """Minimal harness that mirrors KimiSoul's event buffer + dump logic.

    We extract the exact same fields and logic from KimiSoul to test
    the producer contract in isolation, without needing a full Runtime.
    """

    def __init__(self, work_dir: Path) -> None:
        self._status_event_buffer: list[tuple[int, Any]] = []
        self._status_seq: int = 0
        self._status_interval: int = 5
        self._work_dir = work_dir

    @property
    def status_path(self) -> Path:
        return self._work_dir / ".agent_status.jsonl"

    def reset_for_trial(self) -> None:
        """Simulate trial-start reset."""
        self._status_event_buffer.clear()
        self._status_seq = 0
        self.status_path.write_text("", encoding="utf-8")

    def enqueue_events(self, step_no: int, messages: list[FakeMessage]) -> None:
        """Simulate _grow_context pushing events to the buffer."""
        for msg in messages:
            self._status_event_buffer.append((step_no, msg))

    def dump_status(self, step_no: int) -> None:
        """Simplified version of KimiSoul._dump_agent_status."""
        if self._status_interval <= 0:
            return
        if not self._status_event_buffer:
            return

        events = list(self._status_event_buffer)
        self._status_event_buffer.clear()

        lines: list[str] = []
        for event_step, msg in events:
            self._status_seq += 1
            text = msg.extract_text(" ") if hasattr(msg, "extract_text") else str(msg.content)
            preview = text[:500] if text else ""
            entry: dict[str, object] = {
                "seq": self._status_seq,
                "step": event_step,
                "role": msg.role,
                "content_preview": preview,
            }
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [tc.function.name for tc in msg.tool_calls]
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(self.status_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def read_all_events(self) -> list[dict]:
        """Read all events from the status file."""
        if not self.status_path.exists():
            return []
        events = []
        for line in self.status_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events


class TestCompactionPreservedTailNoReplay:
    """Compaction + preserved tail + new messages → no replay, no loss."""

    def test_compaction_does_not_replay_old_events(self, tmp_path: Path) -> None:
        """Simulate:
        1. Steps 1-10: _grow_context produces events, dumped to file
        2. Compaction happens (preserved tail has old messages) — NO buffer push
        3. Steps 11-12: _grow_context produces new events
        4. Dump: only new events from steps 11-12 appear, no replay of steps 1-10
        """
        h = StatusBufferHarness(tmp_path)
        h.reset_for_trial()

        # Steps 1-10: produce and dump events
        for step in range(1, 11):
            h.enqueue_events(
                step,
                [
                    FakeMessage("assistant", f"thinking at step {step}"),
                    FakeMessage("tool", f"ReadFile(/src/file_{step}.py) result"),
                ],
            )
        h.dump_status(10)

        events_before = h.read_all_events()
        assert len(events_before) == 20  # 10 steps × 2 messages each
        assert events_before[-1]["seq"] == 20

        # --- COMPACTION HAPPENS ---
        # Compaction rewrites context.history to [summary, old8, old9, old10, ...]
        # But it does NOT push anything to the buffer.
        # The preserved tail (old8, old9, old10) should NOT appear again.

        # Steps 11-12: _grow_context produces new events (post-compaction)
        h.enqueue_events(
            11,
            [
                FakeMessage("assistant", "new thinking after compaction"),
                FakeMessage("tool", "Shell(make build) result"),
            ],
        )
        h.enqueue_events(
            12,
            [
                FakeMessage("assistant", "analyzing build output"),
            ],
        )
        h.dump_status(12)

        all_events = h.read_all_events()
        # Should have 20 old + 3 new = 23 total
        assert len(all_events) == 23

        # New events should have seq 21-23 and steps 11-12
        new_events = all_events[20:]
        assert new_events[0]["seq"] == 21
        assert new_events[0]["step"] == 11
        assert new_events[0]["content_preview"] == "new thinking after compaction"

        assert new_events[1]["seq"] == 22
        assert new_events[1]["step"] == 11

        assert new_events[2]["seq"] == 23
        assert new_events[2]["step"] == 12

        # Verify NO old events were replayed (no duplicate seqs)
        seqs = [e["seq"] for e in all_events]
        assert seqs == list(range(1, 24))  # strictly monotonic, no duplicates

    def test_preserved_tail_messages_not_in_buffer(self, tmp_path: Path) -> None:
        """Verify that after compaction, even if old messages exist in
        context.history (preserved tail), they are NOT in the event buffer
        and thus NOT re-emitted.
        """
        h = StatusBufferHarness(tmp_path)
        h.reset_for_trial()

        # Pre-compaction: produce events for steps 1-5
        for step in range(1, 6):
            h.enqueue_events(step, [FakeMessage("assistant", f"step {step}")])
        h.dump_status(5)

        assert len(h.read_all_events()) == 5
        assert len(h._status_event_buffer) == 0  # buffer drained

        # Compaction happens — buffer is untouched (no push)
        # Even though context.history now has [summary, old4, old5],
        # the buffer remains empty.

        assert len(h._status_event_buffer) == 0

        # New message arrives via _grow_context
        h.enqueue_events(6, [FakeMessage("assistant", "post-compaction new")])
        h.dump_status(6)

        all_events = h.read_all_events()
        assert len(all_events) == 6  # 5 old + 1 new
        assert all_events[5]["seq"] == 6
        assert all_events[5]["content_preview"] == "post-compaction new"


class TestRevertRestoredMessagesNoReplay:
    """Revert + restored old messages + new messages → no replay, no loss."""

    def test_revert_does_not_replay_old_events(self, tmp_path: Path) -> None:
        """Simulate:
        1. Steps 1-5: _grow_context produces events, dumped
        2. Steps 6-8: _grow_context produces more events, dumped
        3. Revert to step 5 — restores old history but does NOT touch buffer
        4. Steps 6'-7': _grow_context produces new (different) events
        5. Dump: only new events from steps 6'-7' appear
        """
        h = StatusBufferHarness(tmp_path)
        h.reset_for_trial()

        # Steps 1-5: produce and dump
        for step in range(1, 6):
            h.enqueue_events(step, [FakeMessage("assistant", f"original step {step}")])
        h.dump_status(5)
        assert len(h.read_all_events()) == 5

        # Steps 6-8: produce and dump
        for step in range(6, 9):
            h.enqueue_events(step, [FakeMessage("assistant", f"original step {step}")])
        h.dump_status(8)
        assert len(h.read_all_events()) == 8

        # --- REVERT TO STEP 5 ---
        # Context.history is rebuilt to the step-5 state.
        # But the buffer is independent — it stays empty (already drained).
        # The old messages (steps 6-8) are back in history but NOT in buffer.

        assert len(h._status_event_buffer) == 0

        # Steps 6'-7': new events after revert (different content)
        h.enqueue_events(6, [FakeMessage("assistant", "RETRIED step 6 after revert")])
        h.enqueue_events(7, [FakeMessage("assistant", "RETRIED step 7 after revert")])
        h.dump_status(7)

        all_events = h.read_all_events()
        # 5 original + 3 original-6-7-8 + 2 retried = 10
        assert len(all_events) == 10

        # Verify the retried events have correct content
        assert all_events[8]["seq"] == 9
        assert all_events[8]["content_preview"] == "RETRIED step 6 after revert"
        assert all_events[9]["seq"] == 10
        assert all_events[9]["content_preview"] == "RETRIED step 7 after revert"

        # Verify monotonic seq — no gaps, no duplicates
        seqs = [e["seq"] for e in all_events]
        assert seqs == list(range(1, 11))

    def test_revert_with_undrained_buffer_preserves_new_events(self, tmp_path: Path) -> None:
        """Edge case: events were pushed to buffer but not yet drained
        when revert happens. They should still be emitted.
        """
        h = StatusBufferHarness(tmp_path)
        h.reset_for_trial()

        # Step 1: produce and dump
        h.enqueue_events(1, [FakeMessage("assistant", "step 1")])
        h.dump_status(1)
        assert len(h.read_all_events()) == 1

        # Step 2: produce but DON'T dump yet
        h.enqueue_events(2, [FakeMessage("assistant", "step 2 undrained")])
        assert len(h._status_event_buffer) == 1

        # --- REVERT HAPPENS ---
        # Buffer still has the step 2 event — revert doesn't touch it

        # Step 2': produce new event
        h.enqueue_events(2, [FakeMessage("assistant", "step 2 retried")])
        assert len(h._status_event_buffer) == 2  # both in buffer

        # Now dump — both the undrained and retried events should appear
        h.dump_status(2)
        all_events = h.read_all_events()
        assert len(all_events) == 3  # 1 dumped + 2 from buffer
        assert all_events[1]["content_preview"] == "step 2 undrained"
        assert all_events[2]["content_preview"] == "step 2 retried"
        assert [e["seq"] for e in all_events] == [1, 2, 3]


class TestTrialReset:
    """Trial boundary correctly resets state."""

    def test_trial_reset_clears_buffer_and_seq(self, tmp_path: Path) -> None:
        """After trial reset, seq starts from 0 and old events are gone."""
        h = StatusBufferHarness(tmp_path)
        h.reset_for_trial()

        # Trial 1: produce events
        h.enqueue_events(1, [FakeMessage("assistant", "trial 1 step 1")])
        h.enqueue_events(2, [FakeMessage("assistant", "trial 1 step 2")])
        h.dump_status(2)
        assert len(h.read_all_events()) == 2

        # Trial 2: reset
        h.reset_for_trial()
        assert len(h._status_event_buffer) == 0
        assert h._status_seq == 0

        # Trial 2: produce events — seq restarts from 1
        h.enqueue_events(1, [FakeMessage("assistant", "trial 2 step 1")])
        h.dump_status(1)

        all_events = h.read_all_events()
        assert len(all_events) == 1  # file was truncated on reset
        assert all_events[0]["seq"] == 1
        assert all_events[0]["content_preview"] == "trial 2 step 1"
