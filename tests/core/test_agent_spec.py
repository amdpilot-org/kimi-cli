from __future__ import annotations

import re
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from kimi_cli.agentspec import DEFAULT_AGENT_FILE, load_agent_spec
from kimi_cli.exception import AgentSpecError


def test_load_default_agent_spec():
    """Test loading the default agent specification."""
    spec = load_agent_spec(DEFAULT_AGENT_FILE)

    assert spec.name == snapshot("")
    assert spec.system_prompt_path == DEFAULT_AGENT_FILE.parent / "system.md"
    assert spec.system_prompt_args == snapshot(
        {
            "ROLE_ADDITIONAL": """\
You are the **Planning Supervisor** for this AMD optimization session. Your role is to plan, delegate, verify, and report — you do NOT write code, run scripts, or implement optimizations yourself.

Use the `Task` tool to delegate ALL implementation work to subagents. Your value is in correctly scoping each task (providing file paths, class names, benchmark commands, and the list of already-applied changes), and in rigorously verifying subagent results before accepting them.
""",
            "ROLE_EXECUTOR_CONTEXT": '> **Supervisor note:** The `Coding Conventions` and `Optimization Discipline` sections below define standards you must **enforce when evaluating subagent results** — they are not actions you perform yourself. Concretely: reject any result that substitutes a config toggle for real source-file edits, a micro-benchmark for E2E latency, or a "deferred" report with fewer than 3 documented attempts. The AMD/ROCm Environment Rules below are background knowledge for understanding and validating what subagents report.\n',
        }
    )
    assert spec.exclude_tools == snapshot([])
    assert spec.tools == snapshot(
        [
            "kimi_cli.tools.multiagent:Task",
            "kimi_cli.tools.todo:SetTodoList",
            "kimi_cli.tools.shell:Shell",
            "kimi_cli.tools.file:ReadFile",
            "kimi_cli.tools.file:Glob",
            "kimi_cli.tools.file:Grep",
            "kimi_cli.tools.file:WriteFile",
            "kimi_cli.tools.file:StrReplaceFile",
            "kimi_cli.tools.web:FetchURL",
        ]
    )
    subagents = {
        name: (spec.path.relative_to(DEFAULT_AGENT_FILE.parent).as_posix(), spec.description)
        for name, spec in spec.subagents.items()
    }
    assert subagents == snapshot(
        {
            "explorer": (
                "explorer.yaml",
                "Explores and analyzes codebases — maps repo structure, traces execution flows, identifies key components and hardware-specific code paths.",
            ),
            "porter": (
                "porter.yaml",
                "Ports NVIDIA/CUDA-only code to AMD GPUs under ROCm — resolves compatibility issues, replaces vendor-locked extensions, validates with forward pass.",
            ),
            "profiler": (
                "profiler.yaml",
                "Creates benchmarks and runs profiling — produces categorized GPU time breakdowns, identifies performance bottlenecks with rigorous measurement.",
            ),
            "optimizer": (
                "optimizer.yaml",
                "Implements specific performance optimizations and validates with benchmarks — kernel fusion, torch.compile, attention backends, quantization, etc.",
            ),
        }
    )

    subagent_specs = {name: load_agent_spec(spec.path) for name, spec in spec.subagents.items()}
    assert subagent_specs["explorer"].name == snapshot("")
    assert subagent_specs["explorer"].system_prompt_path == DEFAULT_AGENT_FILE.parent / "system.md"
    assert subagent_specs["explorer"].system_prompt_args == snapshot(
        {
            "ROLE_ADDITIONAL": """\
You are now running as a **Code Explorer** subagent. All `user` messages come from the main agent. The main agent cannot see your context — only your final message.

Your role is to explore, analyze, and understand codebases. You excel at:
- Mapping repository structure, key modules, and their relationships
- Tracing execution flows end-to-end (e.g., model loading, inference pipeline, data preprocessing)
- Identifying inner computational components (attention, MLP, normalization, custom ops) and their file locations
- Understanding configuration systems and how parameters propagate through the code
- Spotting hardware-specific code paths (CUDA guards, device checks, vendor-locked extensions)

Guidelines:
- Read broadly first to build a structural map, then dive deep into the most relevant files.
- Provide structured, detailed findings — include file paths, class/function names, and how components connect.
- When tracing execution flows, describe the call chain with concrete file paths and line references.
- If you write any helper scripts for exploration, mention them in your summary.
- Your final message must be a comprehensive report that the main agent can act on without needing to re-explore the same code.
""",
            "ROLE_EXECUTOR_CONTEXT": "> **Explorer note:** You explore and report — you do not make code changes or run optimizations. The `Coding Conventions` and `Optimization Discipline` sections below do not govern your work; skip them. Focus on producing a comprehensive structural report the main agent can act on directly.\n",
        }
    )
    assert subagent_specs["explorer"].exclude_tools == snapshot(
        [
            "kimi_cli.tools.multiagent:Task",
            "kimi_cli.tools.todo:SetTodoList",
        ]
    )
    assert subagent_specs["explorer"].tools == snapshot(
        [
            "kimi_cli.tools.multiagent:Task",
            "kimi_cli.tools.todo:SetTodoList",
            "kimi_cli.tools.shell:Shell",
            "kimi_cli.tools.file:ReadFile",
            "kimi_cli.tools.file:Glob",
            "kimi_cli.tools.file:Grep",
            "kimi_cli.tools.file:WriteFile",
            "kimi_cli.tools.file:StrReplaceFile",
            "kimi_cli.tools.web:FetchURL",
        ]
    )
    sub_subagents = {
        name: (spec.path.relative_to(DEFAULT_AGENT_FILE.parent).as_posix(), spec.description)
        for name, spec in subagent_specs["explorer"].subagents.items()
    }
    assert sub_subagents == snapshot({})


def test_load_agent_spec_basic(agent_file: Path):
    """Test loading a basic agent specification."""
    spec = load_agent_spec(agent_file)

    assert spec.name == snapshot("Test Agent")
    assert spec.system_prompt_path == agent_file.parent / "system.md"
    assert spec.tools == snapshot(["kimi_cli.tools.think:Think"])


def test_load_agent_spec_missing_name(agent_file_no_name: Path):
    """Test missing agent name raises AgentSpecError."""
    with pytest.raises(AgentSpecError, match="Agent name is required"):
        load_agent_spec(agent_file_no_name)


def test_load_agent_spec_missing_system_prompt(agent_file_no_prompt: Path):
    """Test missing system prompt path raises AgentSpecError."""
    with pytest.raises(AgentSpecError, match="System prompt path is required"):
        load_agent_spec(agent_file_no_prompt)


def test_load_agent_spec_missing_tools(agent_file_no_tools: Path):
    """Test missing tools raises AgentSpecError."""
    with pytest.raises(AgentSpecError, match="Tools are required"):
        load_agent_spec(agent_file_no_tools)


def test_load_agent_spec_with_exclude_tools(agent_file_with_tools: Path):
    """Test loading agent spec with excluded tools."""
    spec = load_agent_spec(agent_file_with_tools)

    assert spec.tools == snapshot(["kimi_cli.tools.think:Think", "kimi_cli.tools.shell:Shell"])
    assert spec.exclude_tools == snapshot(["kimi_cli.tools.shell:Shell"])


def test_load_agent_spec_extension(agent_file_extending: Path):
    """Test loading agent spec with extension."""
    spec = load_agent_spec(agent_file_extending)

    assert spec.name == snapshot("Extended Agent")
    assert spec.tools == snapshot(["kimi_cli.tools.think:Think"])


def test_load_agent_spec_default_extension():
    """Test loading agent spec with default extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create extending agent
        extending_agent = tmpdir / "extending.yaml"
        extending_agent.write_text("""
version: 1
agent:
  extend: default
  system_prompt_args:
    CUSTOM_ARG: "custom_value"
  exclude_tools:
    - "kimi_cli.tools.web:SearchWeb"
    - "kimi_cli.tools.web:FetchURL"
""")

        spec = load_agent_spec(extending_agent)

        assert spec.name == snapshot("")
        assert spec.system_prompt_path == DEFAULT_AGENT_FILE.parent / "system.md"
        assert spec.system_prompt_args == snapshot(
            {
                "ROLE_ADDITIONAL": """\
You are the **Planning Supervisor** for this AMD optimization session. Your role is to plan, delegate, verify, and report — you do NOT write code, run scripts, or implement optimizations yourself.

Use the `Task` tool to delegate ALL implementation work to subagents. Your value is in correctly scoping each task (providing file paths, class names, benchmark commands, and the list of already-applied changes), and in rigorously verifying subagent results before accepting them.
""",
                "ROLE_EXECUTOR_CONTEXT": '> **Supervisor note:** The `Coding Conventions` and `Optimization Discipline` sections below define standards you must **enforce when evaluating subagent results** — they are not actions you perform yourself. Concretely: reject any result that substitutes a config toggle for real source-file edits, a micro-benchmark for E2E latency, or a "deferred" report with fewer than 3 documented attempts. The AMD/ROCm Environment Rules below are background knowledge for understanding and validating what subagents report.\n',
                "CUSTOM_ARG": "custom_value",
            }
        )
        assert spec.tools == snapshot(
            [
                "kimi_cli.tools.multiagent:Task",
                "kimi_cli.tools.todo:SetTodoList",
                "kimi_cli.tools.shell:Shell",
                "kimi_cli.tools.file:ReadFile",
                "kimi_cli.tools.file:Glob",
                "kimi_cli.tools.file:Grep",
                "kimi_cli.tools.file:WriteFile",
                "kimi_cli.tools.file:StrReplaceFile",
                "kimi_cli.tools.web:FetchURL",
            ]
        )
        assert spec.exclude_tools == snapshot(
            ["kimi_cli.tools.web:SearchWeb", "kimi_cli.tools.web:FetchURL"]
        )
        assert "explorer" in spec.subagents


def test_load_agent_spec_unsupported_version():
    """Test loading agent spec with unsupported version raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 2
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
""")

        with pytest.raises(AgentSpecError, match="Unsupported agent spec version: 2"):
            load_agent_spec(agent_yaml)


def test_load_agent_spec_nonexistent_file():
    """Test loading nonexistent agent spec file raises AssertionError."""
    nonexistent = Path("/nonexistent/agent.yaml")
    with pytest.raises(
        AgentSpecError,
        match=re.compile(r"Agent spec file not found: [\\/]nonexistent[\\/]agent.yaml"),
    ):
        load_agent_spec(nonexistent)


# Fixtures for test files


@pytest.fixture
def agent_file() -> Generator[Path, Any, Any]:
    """Create a basic agent configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
""")

        yield agent_yaml


@pytest.fixture
def agent_file_no_name() -> Generator[Path, Any, Any]:
    """Create an agent configuration file without name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
""")

        yield agent_yaml


@pytest.fixture
def agent_file_no_prompt() -> Generator[Path, Any, Any]:
    """Create an agent configuration file without system prompt path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create agent.yaml
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  tools: ["kimi_cli.tools.think:Think"]
""")

        yield agent_yaml


@pytest.fixture
def agent_file_no_tools() -> Generator[Path, Any, Any]:
    """Create an agent configuration file without tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
""")

        yield agent_yaml


@pytest.fixture
def agent_file_with_tools() -> Generator[Path, Any, Any]:
    """Create an agent configuration file with tools and exclude_tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("You are a test agent")

        # Create agent.yaml
        agent_yaml = tmpdir / "agent.yaml"
        agent_yaml.write_text("""
version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think", "kimi_cli.tools.shell:Shell"]
  exclude_tools: ["kimi_cli.tools.shell:Shell"]
""")

        yield agent_yaml


@pytest.fixture
def agent_file_extending() -> Generator[Path, Any, Any]:
    """Create an agent configuration file that extends another."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create base agent
        base_agent = tmpdir / "base.yaml"
        base_agent.write_text("""
version: 1
agent:
  name: "Base Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
""")

        # Create system.md
        system_md = tmpdir / "system.md"
        system_md.write_text("Base system prompt")

        # Create extending agent
        extending_agent = tmpdir / "extending.yaml"
        extending_agent.write_text("""
version: 1
agent:
  extend: ./base.yaml
  name: "Extended Agent"
  system_prompt_args:
    CUSTOM_ARG: "custom_value"
""")

        yield extending_agent
