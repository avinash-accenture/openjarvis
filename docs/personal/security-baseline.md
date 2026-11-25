# Personal deployment: security baseline (Phase 1)

This fork runs OpenJarvis as a private, local-only assistant on one Mac. The
controls below are enforced in code and are **on by default**; relaxing any of
them requires an explicit setting in `~/.openjarvis/config.toml` and a restart.

Run `scripts/verify_phase1_security.sh` (with `jarvis serve` running) to check
all of it against the live server.

## What is enforced

| Control | Default | Where it is enforced | Setting to relax it |
| --- | --- | --- | --- |
| No cloud AI inference | on | engine factory, cloud engine constructors, `jarvis serve` engine pool, `/v1/chat/completions`, direct cloud router, `/v1/cloud/reload` | `[security] local_only_inference = false` |
| No cloud speech (OpenAI, ElevenLabs, Cartesia, Deepgram) | on | backend constructors | same as above |
| No cloud/code-running agents (`claude_code`, hybrid, remote `opencode`) | on | agent constructors | same as above |
| No external usage analytics (PostHog) | on | analytics client | same as above |
| API key required, 32+ characters | on | `jarvis serve` start-up | `[server] require_api_key = false` (loopback only) |
| Listen on loopback only | on | `jarvis serve` start-up | `[server] allow_non_loopback_bind = true` |
| No public tunnel (`jarvis tunnel`) | on | `jarvis tunnel` | `[server] allow_public_tunnel = true` |
| Tool allowlist | on | every `ToolExecutor` call | `[security] allowed_tools = [...]` or `tool_allowlist_enabled = false` |

API keys for cloud providers in the shell environment are ignored while
local-only inference is on; `jarvis serve` prints that it detected and ignored
them.

### Local means

* the engine is an allowlisted local engine (`ollama`, `llamacpp`, `mlx`, ...),
* it is not flagged as a cloud engine, and
* its host is this machine (`localhost`, `127.0.0.0/8`, `::1`, `0.0.0.0`).

`OLLAMA_HOST` pointing at another machine, even on your LAN or tailnet, is
treated as remote and refused.

### Default tool allowlist

Read-only and local: `think`, `calculator`, `calendar_search`,
`calendar_upcoming`, `digest_collect`, `knowledge_search`, `knowledge_sql`,
`scan_chunks`, `retrieval`, `memory_search`, `memory_retrieve`, `kg_query`,
`kg_neighbors`, `llm`, `get_pending_actions`, `check_permission`,
`text_to_speech`.

Blocked: shell and code execution (and agents built on them), file read/write,
git, web search, HTTP, browser, messaging (`channel_send`), memory and profile
writes, skill authoring, and the action queue (`queue_action`,
`record_decision`, `execute_pending_actions`). Setting `allowed_tools` replaces
the whole list.

The check runs against the real tool behind any wrapper, so a managed agent
cannot run a blocked tool under an allowed name.

### Authentication

* REST: `Authorization: Bearer <key>` on `/v1/*`, `/api/*`, `/metrics`, and the
  API docs. `/health`, the dashboard shell and static UI files are unauthenticated
  by design; they contain no personal data.
* WebSocket: the key is checked before the connection is accepted.
* The key lives in `[server.auth] api_key` in `config.toml` (mode 600), or in
  `OPENJARVIS_API_KEY`. `jarvis auth create-key` writes only that entry.
* Browser clients store the key in `localStorage`. Treat a compromised browser
  or UI as a compromised key and rotate it with `jarvis auth create-key`.

### Remote access (Phase 3)

The server stays on `127.0.0.1`. Remote devices reach it through Tailscale
(e.g. `tailscale serve`), which forwards to loopback. Requests forwarded that way
arrive *from* `127.0.0.1`, which is why the API key is required even on
loopback: the network boundary and the application key are separate controls.

## Dependency audit (2026-11-25)

* **Python, server dependency set:** 11 vulnerable packages found; 10 upgraded in
  `uv.lock` (aiohttp, anyio, certifi, datasets, fsspec, lxml, multidict,
  pygments, requests, urllib3). **Remaining:** `filelock` 3.32.7 (fixes need
  4.x). It is capped below 4 by `litellm==1.104.2`, which is a cloud-only extra
  that local-only mode blocks. The advisories concern local lock-file handling.
  Revisit when `litellm` relaxes its pin.
* **Python, all optional extras (1,760 packages):** 34 vulnerable packages,
  mostly in extras this deployment does not install (vLLM, virtualenv tooling,
  Werkzeug via evaluation extras). Not changed.
* **Frontend production dependencies (`npm audit --omit=dev`):** 0.
* **Frontend dev tooling:** 4 moderate (`vite-plugin-pwa` → `minimatch`); build-time
  only, fix requires a breaking major upgrade. Not changed.

Re-run before enabling remote access:

```bash
uv export --frozen --no-hashes --no-dev --no-emit-project --extra server -o /tmp/req.txt
uvx pip-audit --no-deps --disable-pip -r /tmp/req.txt
(cd frontend && npm audit --omit=dev)
```

## Known gaps carried into later phases

* No mail search tools yet (`search_email`, `get_email`, ...): Phase 4.
* The action-queue tools let the model record an approval itself
  (`record_decision`). They stay blocked until Phase 5 provides an approval step
  only the user can trigger.
* Telemetry (local SQLite metrics) and traces are stored locally and include
  prompts; they never leave the machine but are personal data at rest.
* The rate limiter needs the native `openjarvis_rust` extension; without it,
  start-up logs a warning and API rate limiting is off.
