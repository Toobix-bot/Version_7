# Beispiele & Nutzungsideen

Policy Wizard
- POST /policy/wizard
- Beispiel JSON: { "template":"solo_dev", "risk_profile":"low", "annotate":true }

Story
- GET /story/state
- POST /story/choose { "id":"opt_1" }

Idle
- GET /game/idle/state
- POST /game/idle/tick

Plan
- POST /plan { "objective":"Verbessere Logging" }
- GET  /plan/ideas

SSE
- GET /events?kinds=plan.created

Tipps
- API‑Key via Header X-API-Key setzen.
- Für erste Schritte /docs im Browser öffnen (Swagger).
