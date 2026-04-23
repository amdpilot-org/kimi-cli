# tests_live — Live integration tests against a real OSS-model endpoint

This suite exercises the amd-dev fork patches that are **only observable in
real conditions** (sglang + Qwen3.5 tool-call quirks, reasoning-content
channels, multi-step status dumps, nudge file polling, ConsultAdvisor
round-trips). Upstream CI can't run these — they need a GPU-served LLM.

## When to run

- **Before** merging any upstream-absorption phase into `amd-dev` — catches
  regressions that `pytest tests/` and `pytest tests_e2e/` can't.
- **After** editing `kimi-cli/src/kimi_cli/soul/kimisoul.py`, any provider
  file in `packages/kosong/`, or any tool under `src/kimi_cli/tools/consult`.

## Running

From the `kimi-cli` submodule root:

```bash
# Verify sglang is up
curl -sS http://localhost:30000/v1/models

# Run the whole suite
uv run pytest tests_live/ -v \
  --live-endpoint=http://localhost:30000/v1 \
  --live-model=Qwen3.5-397B-A17B-SFT-v5.1-64k
```

Or from the amdpilot root:

```bash
make test-live
```

Without `--live-endpoint`, every test in `tests_live/` is skipped — so
invoking plain `pytest` on this repo is always safe.

## What each test covers (amd-dev patch coverage matrix)

| File | amd-dev SHAs covered |
|------|----------------------|
| `test_smoke.py` | baseline: subprocess + status dump wiring |
| `test_provider_sglang.py` | `f5bdde06` reasoning_key, `bf9bed8a` null args default, `47d51341`+`12441e5e` double-encode recovery, `392709e8`+`8c9aa449` custom_headers/verify_ssl |
| `test_contract_status_dump.py` | `725ee9bb` append-only, `33089173` event buffer, `fdfcdeed` content_tail/tool_args_summary, `2ba264f1` interval |
| `test_contract_nudge.py` | `43c4241b` message injection, `fae579d9` dual nudge path |
| `test_contract_consult.py` | `cd148265` ConsultAdvisor tool, `3a180b86` objective_contract, `b3bace7b` unsolicited consult detection |
| `test_shell_max_timeout.py` | `f76da46a` KIMI_MAX_TIMEOUT |
| `test_prompt_file.py` | `5a07c785` --prompt-file |
| `test_agent_system_prompt.py` | `7e1cf460`+system.md cluster — ROLE_ADDITIONAL substitution |

## Conventions

- Every test runs in an isolated `tmp_path/{work,home}` — no state leaks.
- `home/.kimi/config.toml` is regenerated per-test (override with
  `tests_live.helpers.config.write_kimi_config` for patch-specific configs).
- Tests using `KIMI_STATUS_INTERVAL=1` force per-step dumps; others let it
  default to 5 (matches amdpilot production).
- `max_ralph_iterations=0` is the default (disables ralph wrapping for
  single-turn tests). Tests that want to exercise ralph must override it.
- Timing-dependent tests (nudge, consult) use daemon threads and soft
  assertions; they can flake on a loaded system — re-run before filing a bug.

## Failure triage

When a test fails, `conftest.py::_dump_on_failure` prints:
- Tail of `.agent_status.jsonl`
- Full `config.toml`

Also check `{home_dir}/.kimi/logs/kimi.log` (DEBUG-level; the runner passes
`--debug`) for step timing, tool invocation, and retry events.

## What's intentionally out of scope here

- Full eval trials (20+ min per case) — live in the outer amdpilot repo at
  `amdpilot/evals/` and should not be invoked from this suite.
- Fault injection (disconnect sglang mid-request, etc.) for `45cb3437`
  retry coverage. A follow-up could wrap the endpoint in a toxiproxy-like
  shim; not yet implemented.
- Wire-mode protocol tests — kimi-cli upstream already has `tests_e2e/` with
  scripted wire tests; only provider-specific or status-contract behaviour
  lives here.
