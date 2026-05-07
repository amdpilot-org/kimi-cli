"""Verify KIMI_SHELL_MAX_TIMEOUT env var controls Shell MAX_TIMEOUT."""

from __future__ import annotations

import importlib


class TestKimiShellMaxTimeoutEnv:
    def test_default_max_timeout_is_300(self, monkeypatch):
        monkeypatch.delenv("KIMI_SHELL_MAX_TIMEOUT", raising=False)
        import kimi_cli.tools.shell as shell_mod

        importlib.reload(shell_mod)
        assert shell_mod.MAX_TIMEOUT == 300

    def test_max_timeout_from_env_1800(self, monkeypatch):
        monkeypatch.setenv("KIMI_SHELL_MAX_TIMEOUT", "1800")
        import kimi_cli.tools.shell as shell_mod

        importlib.reload(shell_mod)
        assert shell_mod.MAX_TIMEOUT == 1800

    def test_max_timeout_from_env_600(self, monkeypatch):
        monkeypatch.setenv("KIMI_SHELL_MAX_TIMEOUT", "600")
        import kimi_cli.tools.shell as shell_mod

        importlib.reload(shell_mod)
        assert shell_mod.MAX_TIMEOUT == 600
