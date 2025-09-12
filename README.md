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
2. Copy `.env.example` -> `.env` and set `API_TOKENS`, `GROQ_API_KEY` etc.
3. Run: `uvicorn api.app:app --reload`
4. Visit `/docs` for OpenAPI with security schemes.

### Quick Start Script (Windows PowerShell)
Use the helper script to start with correct env:

```
pwsh -ExecutionPolicy Bypass -File scripts/run_dev.ps1 -BindHost 127.0.0.1 -Port 8000 -Reload
```

Parameters:
- `-BindHost` host interface (default 127.0.0.1)
- `-Port` port (default 8000)
- `-Reload` include flag to enable auto-reload
- `-ApiKey` override API token for this session

Then in a second terminal:
```
Invoke-WebRequest -Uri http://127.0.0.1:8000/meta -Headers @{ 'X-API-Key'='test' } -UseBasicParsing
```

### Windows Batch Alternative
```
scripts\run_dev.cmd -Port 8000 -BindHost 127.0.0.1 -ApiKey test -Reload
```
Omit `-Reload` for production-like run.

## CI Security
- pip-audit (fails on HIGH+)
- Bandit (level high -lll)
- Ruff, mypy, pytest.

## Change Management
Plans produce JSON artifacts in `plans/` only (PR-only workflow), snapshot script for git tag backups.

## Testing
**Setup**
- Async tests use `httpx.ASGITransport` (see `tests/conftest.py`).
- Run: `pytest -q`
- Provide a token in env: `API_TOKENS=test-token` (tests also auto-add `test` & `test-token`).

**Auth Expectations**
- Missing or invalid key: 401
- Valid key in header `X-API-Key`: success

**Policy Reload Examples**
Reload with valid file:
```
curl -X POST http://127.0.0.1:8000/policy/reload \
	-H "X-API-Key: test" \
	-H "Content-Type: application/json" \
	-d '{"path":"policies/valid.yaml"}'
```
Reload with invalid file (expect 422):
```
curl -X POST http://127.0.0.1:8000/policy/reload \
	-H "X-API-Key: test" \
	-H "Content-Type: application/json" \
	-d '{"path":"policies/invalid.yaml"}'
```
The error payload lists schema/structure issues.

## Logging
- Set `LOG_FILE` (default `logs/app.log`) and optional `LOG_LEVEL` (INFO, DEBUG, ...).
- Both helper scripts accept a log destination:
	- PowerShell: `-LogFile logs/app.log`
	- Batch: `-LogFile logs\app.log`
- Rotating handler keeps up to 3 backups of ~1MB each.

## .env Auto Load
Both `run_dev.ps1` and `run_dev.cmd` parse a root `.env` (simple KEY=VALUE) before starting.

## LLM Test
Ensure you set `GROQ_API_KEY` in `.env`, then:
```
python examples/llm_test.py
```
It first hits `/llm/status`, then `/llm/chat` with a minimal prompt if configured.

## Deployment (Docker + Render)

1. Build locally:
```
docker build -t version7:local .
docker run -e API_KEY=your_key -p 10000:10000 version7:local
```
2. Health check: `curl -s -H "X-API-Key: your_key" http://localhost:10000/healthz`
3. Deploy to Render:
	- New Web Service → Use repo → Docker detected.
	- Set Environment Variable `API_KEY` (or allow generated in `render.yaml`).
	- Confirm Health Check Path `/healthz`.
4. After deploy, note public base URL (e.g. `https://echo-realm.onrender.com`).

## Static OpenAPI for GitHub Pages

The live interactive docs remain at `/docs`. A static spec lives at `docs/openapi.yaml` for GitHub Pages & GPT Actions.

Regenerate (ensures routes/security up to date):
```
python ops/export_openapi.py  # uses PUBLIC_SERVER_URL env if set
```
Set `PUBLIC_SERVER_URL` before exporting to embed the final server URL.

Enable Pages (GitHub → Settings → Pages): Branch `main`, folder `/docs`.

Resulting URLs (example):
- OpenAPI YAML: `https://toobix-bot.github.io/Version_7/openapi.yaml`
- Landing: `https://toobix-bot.github.io/Version_7/`

## Custom GPT (Actions) Integration

When the API is public:
1. Ensure `openapi.yaml` has correct `servers:` pointing to the deployed domain.
2. In ChatGPT → Create GPT → Actions → Import from URL (Pages URL).
3. Auth: API key in header → Name `X-API-Key`.
4. Suggested instructions:
	- "Wenn der Nutzer eine Spielaktion beschreibt → POST /turn"
	- "Bei Statusfragen → GET /state"
	- "Antworten sind knapp, immersiv; JSON-Felder in erzählenden Text wandeln"

## Rate Limiting & Request IDs

- Simple in-memory limit (5 req/s per agent) disabled under tests (`PYTEST_CURRENT_TEST`).
- Each response carries `X-Request-Id` for correlation; logs are JSON lines with that ID.

## Error Format
All handled errors unify as:
```
{"error": {"code": "<machine_code>", "message": "<human text>"}}
```

## Security Hardening Notes
- CORS restricted to explicit origins (extend via `ALLOWED_ORIGINS`).
- API key required on all mutating endpoints (`/act`, `/turn`, `/plan`, `/policy/reload`, `/llm/*`).
- Risk regex & whitelist gating mitigate prompt/file abuse.
- Optional OPA gate (`opa_allow`) for external policy evaluation.

## Maintenance Scripts
- `ops/export_openapi.py` regenerates `docs/openapi.yaml` with merged security scheme.

## Next Enhancements (Ideas)
- Persisted rate limit (Redis) for multi-instance deploy.
- JWT-based auth alternative.
- Structured metrics label reduction for cardinality control.
- Background task queue for heavier `/turn` logic.


