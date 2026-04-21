from kimi_cli.utils.subprocess_env import get_clean_env


def test_get_clean_env_applies_amdpilot_shell_contract():
    env = get_clean_env(
        {
            "PATH": "/tmp/uv-venv/bin:/usr/bin",
            "PYTHONPATH": "/tmp/uv-venv/site-packages",
            "VIRTUAL_ENV": "/tmp/uv-venv",
            "UV_ACTIVE": "1",
            "AMDPILOT_SHELL_PATH_PREFIX": "/opt/venv/bin",
            "AMDPILOT_SHELL_PYTHONPATH": "",
            "AMDPILOT_SHELL_UNSET_VARS": "VIRTUAL_ENV,UV_ACTIVE",
        }
    )

    assert env["PATH"].startswith("/opt/venv/bin:")
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    assert "UV_ACTIVE" not in env
    assert "AMDPILOT_SHELL_PATH_PREFIX" not in env
    assert "AMDPILOT_SHELL_PYTHONPATH" not in env
    assert "AMDPILOT_SHELL_UNSET_VARS" not in env


def test_get_clean_env_can_restore_vllm_pythonpath():
    env = get_clean_env(
        {
            "PATH": "/tmp/uv-venv/bin:/usr/bin",
            "PYTHONPATH": "/tmp/uv-venv/site-packages",
            "AMDPILOT_SHELL_PATH_PREFIX": "/usr/bin",
            "AMDPILOT_SHELL_PYTHONPATH": "/workspace/vllm",
        }
    )

    assert env["PATH"].startswith("/usr/bin:")
    assert env["PYTHONPATH"] == "/workspace/vllm"
