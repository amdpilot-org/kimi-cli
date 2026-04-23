"""Generate kimi-cli config.toml for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_kimi_config(
    home_dir: Path,
    endpoint: str,
    model: str,
    *,
    provider: str = "live",
    provider_type: str = "openai_legacy",
    api_key: str = "sk-dummy",
    thinking: bool = True,
    yolo: bool = True,
    max_context_size: int = 200000,
    max_steps_per_turn: int = 50,
    max_retries_per_step: int = 3,
    max_ralph_iterations: int = 0,
    extra_provider_lines: str = "",
    extra_model_lines: str = "",
) -> Path:
    """Write ~/.kimi/config.toml in *home_dir* pointing at *endpoint*.

    Mirrors ``amdpilot.orchestrator.runtime.kimi_runtime._build_kimi_config_toml``
    so live tests exercise the same config shape amdpilot produces.
    """
    kimi_dir = home_dir / ".kimi"
    kimi_dir.mkdir(parents=True, exist_ok=True)
    alias = f"{provider}-{model}"
    content = (
        f'default_model = "{alias}"\n'
        f"default_thinking = {'true' if thinking else 'false'}\n"
        f"default_yolo = {'true' if yolo else 'false'}\n"
        f"\n"
        f"[providers.{provider}]\n"
        f'type = "{provider_type}"\n'
        f'base_url = "{endpoint}"\n'
        f'api_key = "{api_key}"\n'
        f"{extra_provider_lines}"
        f"\n"
        f'[models."{alias}"]\n'
        f'provider = "{provider}"\n'
        f'model = "{model}"\n'
        f"max_context_size = {max_context_size}\n"
        f"{extra_model_lines}"
        f"\n"
        f"[loop_control]\n"
        f"max_steps_per_turn = {max_steps_per_turn}\n"
        f"max_retries_per_step = {max_retries_per_step}\n"
        f"max_ralph_iterations = {max_ralph_iterations}\n"
        f"reserved_context_size = 50000\n"
    )
    (kimi_dir / "config.toml").write_text(content, encoding="utf-8")
    return kimi_dir / "config.toml"


def amd_gateway_config_extras(
    *, gateway_key: str, user: str = "test", verify_ssl: bool = False
) -> dict[str, Any]:
    """Return extra config lines exercising `verify_ssl` + `custom_headers` patches.

    Used by `test_provider_sglang.py::test_live_custom_headers_and_verify_ssl`
    to assert our AMD fork's provider-patch wiring.
    """
    provider_lines = (
        f"verify_ssl = {'true' if verify_ssl else 'false'}\n"
        f"\n"
        f"[providers.live.custom_headers]\n"
        f'"X-AMD-Test-Header" = "live-test-marker"\n'
        f'"Ocp-Apim-Subscription-Key" = "{gateway_key}"\n'
        f'"user" = "{user}"\n'
    )
    return {"extra_provider_lines": provider_lines}
