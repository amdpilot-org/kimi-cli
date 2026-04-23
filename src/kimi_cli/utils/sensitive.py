from __future__ import annotations

import fnmatch
from pathlib import PurePath

# High-confidence sensitive file patterns.
# Only patterns with very low false-positive risk are included.
SENSITIVE_PATTERNS: list[str] = [
    # Environment variable / secrets
    ".env",
    ".env.*",
    # SSH private keys (and common key-file extensions)
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "*.pem",
    "*.key",
    # Auth / credential dotfiles
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    # Cloud credentials (path-based, also bare name for stripped-path scenarios)
    ".aws/credentials",
    ".gcp/credentials",
    ".kube/config",
    # GCP service-account JSON (common filename conventions). Intentionally
    # narrow: `credentials.json` alone is NOT flagged (common app config).
    "service-account*.json",
    "*-service-account.json",
    # Bare-name fallback for stripped paths (e.g. rg output after
    # prefix-stripping .aws/credentials → credentials). Broad by design;
    # exemptions below handle obvious non-sensitive cases.
    "credentials",
]

# Suffix-based exemptions for example/template variants.
# A filename whose final segment (after the last dot) matches one of these
# is treated as non-sensitive regardless of the base pattern it hits.
# Example: `.env.example`, `credentials.json.sample`, `config.pem.dist`.
SENSITIVE_SUFFIX_EXEMPTIONS: frozenset[str] = frozenset(
    {
        "example",
        "sample",
        "template",
        "dist",
        "tmpl",
    }
)

# Exact-filename exemptions (kept for backward compat with prior explicit list).
SENSITIVE_EXEMPTIONS: set[str] = {
    ".env.example",
    ".env.sample",
    ".env.template",
    # Documentation about credentials, not actual credentials.
    "credentials.md",
}


def _is_exempted(name: str) -> bool:
    """Filename-level exemption check.

    Trips on (a) exact match in ``SENSITIVE_EXEMPTIONS`` or (b) a trailing
    suffix token in ``SENSITIVE_SUFFIX_EXEMPTIONS`` (e.g. ``.env.example``
    → suffix ``example``).
    """
    if name in SENSITIVE_EXEMPTIONS:
        return True
    # Suffix check: split on last dot, compare final segment.
    if "." in name:
        tail = name.rsplit(".", 1)[1].lower()
        if tail in SENSITIVE_SUFFIX_EXEMPTIONS:
            return True
    return False


def is_sensitive_file(path: str) -> bool:
    """Check if a file path matches any sensitive file pattern.

    Patterns are matched two ways:
      - Path-like patterns (those containing ``/``) must appear as a suffix
        of ``path`` or as a ``/``-prefixed substring (e.g. ``.aws/credentials``
        matches both ``/home/u/.aws/credentials`` and bare ``.aws/credentials``).
      - Filename patterns (no ``/``) are fnmatched against the basename.

    Filename-level exemptions (``.env.example``, any ``*.sample`` /
    ``*.template`` / ``*.dist`` / ``*.tmpl`` / ``*.example`` suffix) short-
    circuit to False even when the base pattern would otherwise match.
    """
    name = PurePath(path).name
    if _is_exempted(name):
        return False
    for pattern in SENSITIVE_PATTERNS:
        if "/" in pattern:
            if path.endswith(pattern) or ("/" + pattern) in path:
                return True
        else:
            if fnmatch.fnmatch(name, pattern):
                return True
    return False


def sensitive_file_warning(paths: list[str]) -> str:
    """Generate a warning message for sensitive files that were skipped."""
    names = sorted({PurePath(p).name for p in paths})
    file_list = ", ".join(names[:5])
    if len(names) > 5:
        file_list += f", ... ({len(names)} files total)"
    return (
        f"Skipped {len(paths)} sensitive file(s) ({file_list}) "
        f"to protect secrets. These files may contain credentials or private keys."
    )
