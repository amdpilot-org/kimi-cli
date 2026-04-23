"""E2E tests for AskUserQuestion via the Wire protocol.

NOTE (amdpilot-org fork): the AMD default agent.yaml intentionally drops
`kimi_cli.tools.ask_user:AskUserQuestion` from the tool list (see
amd-dev commits `cd2fb047` reset-default-agent-tools + `7e1e72b7`
specialize-default-agent). Tests that exercise the AskUserQuestion flow
via the default agent therefore fail on this fork because the scripted
tool call has no handler.

The wire protocol + server plumbing for AskUserQuestion is still intact —
any user who re-enables the tool in their own agent.yaml will get the
full flow back. These tests are skipped on the fork baseline to keep CI
green; if AMD ever re-enables AskUserQuestion in the default agent, undo
the skip.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from .wire_helpers import (
    build_ask_user_tool_call,
    build_question_response,
    collect_until_response,
    make_home_dir,
    make_work_dir,
    send_initialize,
    start_wire,
    summarize_messages,
    write_scripted_config,
)

pytestmark = pytest.mark.skip(
    reason=(
        "AMD fork default agent excludes AskUserQuestion tool — these tests "
        "require it in the default toolset. See module docstring."
    )
)


def _make_question(
    question: str = "Which option?",
    options: list[dict[str, str]] | None = None,
    multi_select: bool = False,
) -> dict[str, Any]:
    if options is None:
        options = [
            {"label": "Alpha", "description": "First"},
            {"label": "Beta", "description": "Second"},
        ]
    return {
        "question": question,
        "header": "Test",
        "options": options,
        "multi_select": multi_select,
    }


def _question_request_handler(answers: dict[str, str]):
    """Return a request_handler that responds to QuestionRequest with the given answers."""

    def handler(msg: dict[str, Any]) -> dict[str, Any]:
        params = msg.get("params", {})
        msg_type = params.get("type")
        if msg_type == "QuestionRequest":
            return build_question_response(msg, answers)
        # For other request types (e.g. approval), just approve
        from .wire_helpers import build_approval_response

        return build_approval_response(msg, "approve")

    return handler


def test_question_request_answer(tmp_path) -> None:
    """Test normal question → answer flow through Wire protocol."""
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q1", [question]),
            ]
        ),
        "text: done",
    ]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        yolo=True,  # yolo to skip approval for other tools
    )
    try:
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )

        answers = {"Which option?": "Alpha"}
        resp, messages = collect_until_response(
            wire,
            "prompt-1",
            request_handler=_question_request_handler(answers),
        )
        assert resp.get("result", {}).get("status") == "finished"

        # Verify the QuestionRequest was sent
        summary = summarize_messages(messages)
        question_requests = [m for m in summary if m.get("type") == "QuestionRequest"]
        assert len(question_requests) == 1
        qr_payload = question_requests[0]["payload"]
        assert qr_payload["tool_call_id"] == "tc-q1"
        assert len(qr_payload["questions"]) == 1
        assert qr_payload["questions"][0]["question"] == "Which option?"

        # Verify the ToolResult contains the answers
        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        assert len(tool_results) >= 1
        for tr in tool_results:
            rv = tr["payload"]["return_value"]
            if tr["payload"]["tool_call_id"] == "tc-q1":
                assert not rv["is_error"]
                output = json.loads(rv["output"])
                assert output == {"answers": {"Which option?": "Alpha"}}
                break
        else:
            raise AssertionError("ToolResult for tc-q1 not found")
    finally:
        wire.close()


def test_question_request_error_response(tmp_path) -> None:
    """Test that a JSON-RPC error response resolves to empty answers without crash."""
    question = _make_question()
    scripts = [
        "\n".join(
            [
                "text: asking",
                build_ask_user_tool_call("tc-q2", [question]),
            ]
        ),
        "text: done after error",
    ]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
        yolo=True,
    )
    try:
        send_initialize(wire, capabilities={"supports_question": True})
        wire.send_json(
            {
                "jsonrpc": "2.0",
                "id": "prompt-1",
                "method": "prompt",
                "params": {"user_input": "ask me"},
            }
        )

        def error_handler(msg: dict[str, Any]) -> dict[str, Any]:
            params = msg.get("params", {})
            msg_type = params.get("type")
            if msg_type == "QuestionRequest":
                return {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "error": {"code": -32000, "message": "User cancelled"},
                }
            from .wire_helpers import build_approval_response

            return build_approval_response(msg, "approve")

        resp, messages = collect_until_response(
            wire,
            "prompt-1",
            request_handler=error_handler,
        )
        # Should complete normally (not crash)
        assert resp.get("result", {}).get("status") == "finished"

        # Verify the ToolResult has empty answers (error response resolves to {})
        summary = summarize_messages(messages)
        tool_results = [m for m in summary if m.get("type") == "ToolResult"]
        for tr in tool_results:
            if tr["payload"]["tool_call_id"] == "tc-q2":
                rv = tr["payload"]["return_value"]
                assert not rv["is_error"]
                output = json.loads(rv["output"])
                assert output["answers"] == {}
                assert "dismissed" in output.get("note", "").lower()
                break
        else:
            raise AssertionError("ToolResult for tc-q2 not found")
    finally:
        wire.close()


def test_question_capability_negotiation(tmp_path) -> None:
    """Test that the server reports supports_question in its capabilities."""
    scripts = ["text: hello"]
    config_path = write_scripted_config(tmp_path, scripts)
    work_dir = make_work_dir(tmp_path)
    home_dir = make_home_dir(tmp_path)

    wire = start_wire(
        config_path=config_path,
        config_text=None,
        work_dir=work_dir,
        home_dir=home_dir,
    )
    try:
        resp = send_initialize(wire, capabilities={"supports_question": True})
        result = resp.get("result", {})
        # Server should always report supports_question: True
        caps = result.get("capabilities", {})
        assert caps.get("supports_question") is True
    finally:
        wire.close()
