# Compliance & Governance Artifacts

This directory holds documentation supporting DPIA / EU AI Act / NIST AI RMF / ISO 42001 alignment.

## Artifacts
- Logging: Structured events in SQLite (events table) with timestamps, actor, action, inputs, outputs, scores.
- Policies-as-Code: YAML in `policies/` hot-reloaded via `POST /policy/reload`.
- Risk Controls: Kill switch env `KILL_SWITCH`, deterministic seed, path whitelisting.
- Change Management: Planner produces patch bundles in `plans/` requiring PR-based merge.
- Metrics: Prometheus at `/metrics` (latency, counts, policy decisions).
- SSE Audit Feed: `/events` for real-time oversight.

## DPIA Pointers
Describe data categories (operational logs only, no personal data by design), retention (configurable), access controls (token-based), and rights (data export possible via SQLite dump).

## AI Act Alignment
Risk mitigation, transparency (meta endpoint), traceability (events + proposals), human oversight (PR review), robustness (tests + policies).
