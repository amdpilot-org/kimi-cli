"""Tests for the sensitive file detection module."""

from __future__ import annotations

import pytest

from kimi_cli.utils.sensitive import is_sensitive_file, sensitive_file_warning


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "/app/.env",
        "project/.env",
    ],
)
def test_is_sensitive_env_files(path: str):
    assert is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.local",
        ".env.production",
        "/app/.env.staging",
    ],
)
def test_is_sensitive_env_variants(path: str):
    assert is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "/home/user/.ssh/id_rsa",
        "/home/user/.ssh/id_ed25519",
    ],
)
def test_is_sensitive_ssh_keys(path: str):
    assert is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "/home/user/.aws/credentials",
        "/home/user/.gcp/credentials",
        ".aws/credentials",
        ".gcp/credentials",
        "credentials",
    ],
)
def test_is_sensitive_cloud_credentials(path: str):
    assert is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "app.py",
        "config.yml",
        "README.md",
        "package.json",
        "server.key.example",
        "id_rsa.pub",
        "credentials.json",
        ".envrc",
        "environment.py",
        ".env_example",
        ".env.example",
        ".env.sample",
        ".env.template",
        "/app/.env.example",
    ],
)
def test_not_sensitive_normal_files(path: str):
    assert not is_sensitive_file(path)


def test_sensitive_file_warning_single():
    warning = sensitive_file_warning([".env"])
    assert "1 sensitive file(s)" in warning
    assert ".env" in warning
    assert "protect secrets" in warning


def test_sensitive_file_warning_multiple():
    warning = sensitive_file_warning([".env", ".env.local", "id_rsa"])
    assert "3 sensitive file(s)" in warning
    assert ".env" in warning
    assert "id_rsa" in warning


# ---------------------------------------------------------------------------
# Expanded coverage (second-pass review follow-up)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # PEM keys / certs
        "server.pem",
        "/etc/ssl/private/server.pem",
        "tls.key",
        "/etc/letsencrypt/live/example.com/privkey.pem",
        # Private key variants
        "id_dsa",
        "/home/user/.ssh/id_dsa",
        # Auth dotfiles
        ".netrc",
        "/home/user/.netrc",
        ".pgpass",
        "/home/user/.pgpass",
        ".npmrc",
        "/home/user/.npmrc",
        ".pypirc",
        "/home/user/.pypirc",
        # Kubernetes
        "/home/user/.kube/config",
        ".kube/config",
        # GCP service account JSON
        "service-account.json",
        "service-account-prod.json",
        "my-sa-service-account.json",
    ],
)
def test_is_sensitive_expanded_patterns(path: str):
    assert is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        # Skill assets MUST NOT trip the sensitive filter — they're
        # reference docs the agent legitimately needs to read.
        "/workspace/skills/amd-rocm-porting/SKILL.md",
        "/workspace/skills/amd-kernel-optimization/references/torch-compile-and-cudagraph.md",
        "/workspace/skills/aiter-repo/references/module-mapping.md",
        "skills/profiling-discipline/references/profile.md",
        # Template / example suffix variants are not sensitive
        "server.key.example",
        "server.pem.template",
        "config.pem.dist",
        "secrets.env.sample",
        "auth.key.tmpl",
        # Public keys / documentation about credentials
        "id_rsa.pub",
        "credentials.md",
        "README-credentials.md",
        # Generic `credentials.json` is common app config, NOT flagged
        # (narrower than `service-account*.json`)
        "credentials.json",
    ],
)
def test_not_sensitive_expanded_not_sensitive(path: str):
    assert not is_sensitive_file(path)


def test_skill_asset_not_flagged_as_sensitive():
    """Regression guard (second-pass review): AMD skills under
    ``/workspace/skills/**/references/*.md`` must not match any sensitive
    pattern, or the agent can't read its own docs."""
    skill_assets = [
        "/workspace/skills/amd-rocm-porting/SKILL.md",
        "/workspace/skills/amd-rocm-porting/references/torch-compile-and-cudagraph.md",
        "/workspace/skills/amd-kernel-optimization/references/rocprofv3.md",
        "/workspace/skills/aiter-repo/references/module-mapping.md",
    ]
    for path in skill_assets:
        assert not is_sensitive_file(path), f"Skill asset falsely flagged: {path}"
