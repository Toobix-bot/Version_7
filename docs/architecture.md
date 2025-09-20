# Architecture Overview

This document captures a concise, developer-facing overview of the system. It complements existing docs (OVERVIEW.md, STORY.md, IDLE.md) and focuses on runtime flows and key design choices.

## Core components
- FastAPI backend (`api/app.py`): routes, SSE stream, metrics, in-process rate limiting, SQLite persistence.
- UI (inline HTML/JS served from `/story/ui`): dashboard, suggestions, plan→PR, policies; EventSource for live events.
- SSE queue: bounded per-connection queue with drop-on-overflow and Prometheus counter `sse_queue_dropped_total`.
- VS Code extension (`vscode-extension/`): starts the backend (via watchdog), embeds the UI in a webview, stores API key in SecretStorage.
- Watchdog (`scripts/watchdog.py`): keeps the server running with exponential backoff and a stop-file mechanism.

## Authentication model
- API requests: header `X-API-Key`.
- SSE EventSource: query parameter `?key=...` (headers are not supported by EventSource in browsers/Webviews).
- Multiple keys supported via `API_TOKENS` (comma-separated). Tests also accept `test`/`test-token`.

## Observability
- Prometheus metrics at `/metrics` (text format) including rates, gauges and histograms.
- SSE heartbeat comments (`:-hb`) every `SSE_HEARTBEAT_INTERVAL` seconds.
- Structured JSON logs with `X-Request-Id` correlation.

## Important environment settings
- SSE queue size: `SSE_QUEUE_MAX` (default 1000)
- SSE heartbeat interval: `SSE_HEARTBEAT_INTERVAL` (default 15)
- In-memory rate limit: `RATE_LIMIT_RPS` (default 5 req/s per agent)
- API tokens: `API_TOKENS` (comma separated)
- Database path: `DB_PATH` (SQLite file)
- Log file and level: `LOG_FILE`, `LOG_LEVEL`

## High-level flow
1. User opens UI (`/story/ui`) and sets API key (stored in localStorage in the UI, SecretStorage in the extension).
2. The UI opens an EventSource to `/events?key=...` and fetches data using `fetch` with `X-API-Key` header.
3. The backend processes actions (plan/suggest/policy/etc.), emits SSE events, and updates metrics.
4. Artifacts are written under `plans/` and `suggestions/`; no direct code writes occur.
5. Optionally, the plan→PR route can generate a branch or a PR (dry-run by default in UI script).

## Sequence examples
- Suggestion (happy path)
  - POST `/suggest/generate` → `suggest.generated` (SSE)
  - Optional: POST `/suggest/llm` to refine
  - POST `/suggest/review` with action `approve|revise` → metrics and SSE reflect the state change

- Plan → PR (dry-run)
  - POST `/plan` → returns variants → UI selection
  - POST `/dev/pr-from-plan` with `dry_run=true` → returns proposed branch name

## Edge cases and resilience
- SSE queue overflow: oldest events are dropped; the metric `sse_queue_dropped_total` increments.
- Idle/heartbeats: EventSource stays alive thanks to `:-hb` keepalives.
- Watchdog restarts the server on failures and backs off progressively; the backoff resets after stable uptime.
- Port conflicts: choose a free port (e.g., 8003) if 8001 is occupied.

## Security considerations
- API key is required for mutating endpoints; UI uses a query param only for EventSource due to browser constraints.
- Risk-regex filters and allowlist are enforced server-side; optional OPA hook can be integrated.
- CORS is limited to configured origins.

## Testing outline
- Unit/async tests (pytest + httpx) cover endpoints, SSE heartbeats, and filtering.
- Add E2E browser tests with Playwright for the key UI flows (see `docs/testing.md`).
