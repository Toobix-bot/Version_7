# Version_7

Self-Observing Life-Agent Development Environment (MVP).

## Goals (MVP)
- FastAPI service with endpoints: `POST /act`, `POST /turn`, `GET /state`, `GET /meta`, `GET /events` (SSE), `GET /metrics`, `POST /policy/reload`, `POST /plan`.
- Observer decorator capturing timing, inputs/outputs, scores -> SQLite (events, proposals tables).
- Policies-as-Code via YAML with hot reload; enforce PR-only change gating.
- Deterministic execution (seed) and path whitelist (no self-mod outside allowed dirs).
- Change plans produce patch artifacts; no direct writes.
- Metrics (Prometheus) & SSE event stream for real-time oversight.

## Directory Structure
- `api/` FastAPI application code
- `plans/` Generated plan/patch artifacts (to be reviewed via PR)
- `policies/` YAML policies
- `compliance/` Governance & DPIA docs
- `tests/` Automated tests
- `scripts/` Utility scripts (backup, snapshot)

## Security & Compliance
Token auth, kill switch, policy enforcement, audit logs, metrics, SSE monitoring. See `compliance/README.md`.

## Getting Started
1. Create virtualenv & install requirements: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and adjust.
3. Run dev server: `uvicorn api.app:app --reload`.

## Roadmap
Further integration with OPA/Rego, advanced risk scoring, and automated PR creation workflow.

