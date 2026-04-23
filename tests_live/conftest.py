"""pytest plugin for live integration tests.

Live tests are OPT-IN: they are skipped unless ``--live-endpoint`` is passed,
and they require a reachable OpenAI-compatible endpoint (typically a local
sglang/vLLM serving Qwen3.5 or similar). Nothing here runs on upstream CI;
this suite is amdpilot's own harness.

Usage::

    cd kimi-cli
    uv run pytest tests_live/ \\
        --live-endpoint=http://localhost:30000/v1 \\
        --live-model=Qwen3.5-397B-A17B-SFT-v5.1-64k
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests_live.helpers.config import write_kimi_config
from tests_live.helpers.runner import KimiRunner


# --------------------------------------------------------------------------
# CLI options
# --------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("live")
    group.addoption(
        "--live-endpoint",
        action="store",
        default=None,
        help="OpenAI-compatible endpoint URL (e.g. http://localhost:30000/v1). "
        "If omitted, all tests_live/ tests are skipped.",
    )
    group.addoption(
        "--live-model",
        action="store",
        default="Qwen3.5-397B-A17B-SFT-v5.1-64k",
        help="Model name (served-model-name on sglang, or model alias).",
    )
    group.addoption(
        "--live-timeout",
        action="store",
        type=int,
        default=180,
        help="Default per-test timeout in seconds.",
    )
    group.addoption(
        "--live-api-key",
        action="store",
        default="sk-dummy",
        help="API key to send to the endpoint (sglang ignores, but required by openai_legacy).",
    )


# --------------------------------------------------------------------------
# Skip all tests in this suite unless --live-endpoint is given
# --------------------------------------------------------------------------

def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live-endpoint"):
        return
    skip = pytest.mark.skip(reason="tests_live/ requires --live-endpoint")
    for item in items:
        if "tests_live" in str(item.fspath):
            item.add_marker(skip)


# --------------------------------------------------------------------------
# Session-scoped fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def live_endpoint(request: pytest.FixtureRequest) -> str:
    """Verify the endpoint is reachable and return its URL."""
    url = request.config.getoption("--live-endpoint")
    if not url:
        pytest.skip("no --live-endpoint")

    import httpx

    try:
        r = httpx.get(f"{url.rstrip('/')}/models", timeout=5.0)
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover
        pytest.fail(
            f"live-endpoint {url} is not reachable: {exc}\n"
            f"Start sglang first, e.g.:\n"
            f"  python3 -m sglang.launch_server --model-path <...> --port 30000"
        )
    return url.rstrip("/")


@pytest.fixture(scope="session")
def live_model(request: pytest.FixtureRequest, live_endpoint: str) -> str:
    """Verify the requested model is served at the endpoint."""
    model = request.config.getoption("--live-model")
    import httpx

    r = httpx.get(f"{live_endpoint}/models", timeout=5.0)
    r.raise_for_status()
    ids = {m["id"] for m in r.json().get("data", [])}
    if model not in ids:
        pytest.fail(
            f"model {model!r} not served at {live_endpoint}; "
            f"served models: {sorted(ids)}"
        )
    return model


@pytest.fixture(scope="session")
def live_api_key(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--live-api-key")


@pytest.fixture(scope="session")
def live_timeout(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--live-timeout")


@pytest.fixture(scope="session")
def kimi_project_dir() -> Path:
    """The root of the kimi-cli repo (so tests can `uv run --project <this>`)."""
    # conftest.py is at tests_live/conftest.py; root is one level up.
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Per-test isolation fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def home_dir(tmp_path: Path) -> Path:
    d = tmp_path / "home"
    d.mkdir()
    return d


@pytest.fixture
def kimi_config(
    home_dir: Path,
    live_endpoint: str,
    live_model: str,
    live_api_key: str,
) -> Path:
    """Default config: openai_legacy, thinking on, yolo on."""
    return write_kimi_config(
        home_dir=home_dir,
        endpoint=live_endpoint,
        model=live_model,
        api_key=live_api_key,
    )


@pytest.fixture
def agent_file(tmp_path: Path) -> Path:
    """A minimal agent.yaml for live tests.

    Exposes core tools (Shell, ReadFile, WriteFile) plus a short system prompt.
    Individual tests can override by writing their own agent file.
    """
    p = tmp_path / "agent.yaml"
    system_md = tmp_path / "system.md"
    system_md.write_text(
        "You are a helpful test agent. Be concise. "
        "Use tools when asked. Reply in one short sentence when the task is done.\n",
        encoding="utf-8",
    )
    p.write_text(
        f"version: 1\n"
        f"agent:\n"
        f"  name: live-test-agent\n"
        f"  system_prompt_path: {system_md}\n"
        f"  tools:\n"
        f'    - "kimi_cli.tools.shell:Shell"\n'
        f'    - "kimi_cli.tools.file:ReadFile"\n'
        f'    - "kimi_cli.tools.file:WriteFile"\n'
        f'    - "kimi_cli.tools.think:Think"\n',
        encoding="utf-8",
    )
    return p


@pytest.fixture
def runner(
    work_dir: Path,
    home_dir: Path,
    agent_file: Path,
    kimi_config: Path,  # noqa: ARG001  (side effect: write config.toml)
    kimi_project_dir: Path,
) -> KimiRunner:
    return KimiRunner(
        work_dir=work_dir,
        home_dir=home_dir,
        agent_file=agent_file,
        project_dir=kimi_project_dir,
    )


# --------------------------------------------------------------------------
# Debug helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _dump_on_failure(
    request: pytest.FixtureRequest, work_dir: Path, home_dir: Path
) -> None:
    """If a test fails, dump .agent_status.jsonl + config.toml for debugging."""
    yield
    # Only runs if we got here after the test finished.
    rep = getattr(request.node, "rep_call", None)
    if rep is None or not rep.failed:
        return
    status = work_dir / ".agent_status.jsonl"
    if status.exists():
        print(f"\n--- {status} (tail 2000 chars) ---")
        print(status.read_text()[-2000:])
    cfg = home_dir / ".kimi" / "config.toml"
    if cfg.exists():
        print(f"\n--- {cfg} ---")
        print(cfg.read_text())


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# --------------------------------------------------------------------------
# Module-level sanity check: bail out early if `kimi` can't even parse args.
# This catches broken pyproject / broken submodule state before we chew
# through an expensive real-LLM test.
# --------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--live-endpoint"):
        return
    import subprocess

    project = Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(
            ["uv", "run", "--project", str(project), "--", "kimi", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )
    except Exception as exc:  # pragma: no cover
        pytest.exit(f"`uv run kimi --version` failed to start: {exc}", returncode=3)
    if r.returncode != 0:
        pytest.exit(
            f"`uv run kimi --version` exited {r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}",
            returncode=3,
        )
