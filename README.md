# Version_7

Self-Observing Life-Agent Development Environment (MVP).

## Goals (MVP)
- FastAPI service with endpoints: `POST /act`, `POST /turn`, `GET /state`, `GET /meta`, `GET /events` (SSE), `GET /metrics`, `POST /policy/reload`, `POST /plan`.
- Observer decorator capturing timing, inputs/outputs, scores -> SQLite (events, proposals tables).
- Policies-as-Code via YAML with hot reload & strict schema `policy/policy.yaml` (see `policy/model.py`).
- Deterministic execution (seed) and path whitelist (no self-mod outside allowed dirs).
- Change plans produce patch artifacts; no direct writes.
- Metrics (Prometheus) & SSE event stream for real-time oversight.
- Groq LLM integration (deterministic defaults temperature=0).

## Security & Risk Gates
- Auth: `X-API-Key` (API key) or Bearer token; write endpoints require credentials.
- Risk Gate Regex filters prompt/context for dangerous patterns (command exec, secrets, URLs, path traversal).
- Whitelist enforcement for target paths; policy-based allowed_dirs override.
- Optional OPA gate (`OPA_ENABLE=true`) evaluating `data.agent.allow`.
- Denials increment `policy_denied_total{reason=...}`.
- Status codes: 401/403 auth/authorization, 422 validation (policy invalid / risky prompt), 503 kill switch.

## Metrics
Exported at `/metrics`:
- `actions_total{kind}`
- `policy_denied_total{reason}`
- `plan_seconds` (Histogram)
- `act_seconds{kind}` (Histogram)

## SSE
`GET /events` emits events + `retry:` hint and `:keepalive` comment every 15s.

## Groq Example
```
python examples/groq_basic.py  # requires GROQ_API_KEY env
```
Default model & generation limits come from policy schema (llm.model, temperature=0 for determinism).

## Policy Schema
See `policy/model.py` for strict fields: version, allowed_dirs, deny_globs, max_diff_lines, llm.*
Invalid policy reload returns 422 with structured errors.

## Development
1. Install deps: `pip install -r requirements.txt`
2. Copy `.env.example` -> `.env` and set `API_TOKEN`, `GROQ_API_KEY` etc.
3. Run: `uvicorn api.app:app --reload`
4. Visit `/docs` for OpenAPI with security schemes.

## CI Security
- pip-audit (fails on HIGH+)
- Bandit (level high -lll)
- Ruff, mypy, pytest.

## Change Management
Plans produce JSON artifacts in `plans/` only (PR-only workflow), snapshot script for git tag backups.

## Testing
- Async tests use `httpx.ASGITransport` (see `tests/conftest.py`).
- Run: `pytest -q`
- Write endpoints require `X-API-Key`; tests set `API_TOKENS=test`/`X-API-Key: test`.

