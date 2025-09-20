# Testing Guide

This guide outlines a pragmatic test approach for the project covering unit, integration, and E2E layers.

## Unit and integration tests (pytest)
- Location: `tests/`
- Transport: `httpx.ASGITransport` for fast in-process FastAPI testing
- Auth setup: set `API_TOKENS` or rely on test fixtures adding `test` and `test-token`
- Key checks:
  - SSE stream lifecycle (`test_sse_stream.py`, `test_sse_heartbeat.py`)
  - SSE filtering by kind and overflow metric (`test_sse_filtering.py`, `test_sse_queue_metric.py`)
  - Policy apply/dry-run, plan variants, PR dry-run
  - Metrics content-type and presence

Run locally:
- `pytest -q`

## E2E browser tests (Playwright) – proposal
Add minimal Playwright setup to validate critical UI flows end-to-end.

Suggested scenarios:
1. Open `/story/ui`, set API key, verify dashboard badges turn green after health/ready.
2. Generate auto suggestions, expect `suggest.generated` toast and list to populate.
3. Policy: load → validate (dry-run) → apply (if desired in a temp workspace).
4. SSE: verify heartbeats and at least one domain event arriving.

Implementation sketch:
- Add `playwright.config.ts` with baseURL (from env `E2E_BASE_URL`)
- One spec `e2e/ui.spec.ts` with the above scenarios
- Document running via `npx playwright test` and CI hints

## Load/robustness tests – proposal
- Python async test creating many events quickly; assert that system remains responsive and that `sse_queue_dropped_total` increments when exceeding `SSE_QUEUE_MAX`.

## CI recommendations
- Keep `pytest -q` mandatory.
- Optionally gate PRs with Playwright smoke on a hosted ephemeral server or local headless run.
- Include `pip-audit`, `ruff`, `mypy`.
