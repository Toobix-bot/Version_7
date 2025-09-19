# Version_7

Self-Observing Life-Agent Development Environment (MVP).

> Dokumentationsstruktur nach Divio: Tutorial (Start), How-To (Aufgaben), Reference (API/Felder), Explanation (Hintergründe). Dieser README bündelt die wichtigsten Einstiegspfade für schnelle Wirksamkeit.

Hinweis/Quick Links:
- PR-Roadmap & Leitplanken: siehe [copilot-instructions.md](./copilot-instructions.md)

## Kurz erklärt
Ein kleiner Server mit UI, der Pläne baut, sich selbst beobachtet und sich per Regeln steuern lässt. In der Web-UI siehst du Meta/LLM-Infos, eine Plan-Demo, einen LLM-Chat (sparsam) sowie Live-Bereiche für Events & Metrics.

Neu: Tab-basierte deutsche UI, Impact-Score für Vorschläge, Multi-Agent Memory (`agent_id`), Quest-Fortschritt, kategorisierte Gedanken (risk/opportunity/action/system/neutral), verbesserter Diff-Viewer (Zeilennummern, Suche, Hunk-Navigation), automatische statische Verbesserungsvorschläge (`/suggest/auto`), Onboarding-Panel ("Erste Schritte"), Endpoint & Script für PR aus Plan (`POST /dev/pr-from-plan`, `scripts/new_pr_from_plan.ps1`), Idle Auto-Ticks bei Inaktivität (`idle.tick.auto` Events) sowie Policy Dry-Run Shortcut (`/policy/dry-run`).

Frisch dazu:
- Dashboard-Toast bei `suggest.generated`
- Idle Tipps im Dashboard per `idle.suggest`
- "Aktiv seit"-Timer (umschaltet bei Useraktionen vs. `idle.tick.auto`)
- Begrenzte SSE-Queue mit Metrik `sse_queue_dropped_total` (Dropping bei Überlauf)

Wichtiges auf einen Blick:
- OpenAPI-Doku: `/docs` (Header `X-API-Key` erforderlich)
- Live-JSON: `/openapi.json` (enthält `servers` und Security-Schema)
- Statische Spec: `docs/openapi.yaml` (für GitHub Pages / GPT Actions)
- Dashboard: `/ui`

SSE & API-Key (Wichtig):
- EventSource kann keine eigenen Header senden. Der Server akzeptiert daher den API-Key auch als Query-Parameter: `/events?key=<API_KEY>`.
- Für `fetch()`-Aufrufe weiterhin `X-API-Key` im Header setzen.
 - Optional: Queuegröße via `SSE_QUEUE_MAX` (Default 1000). Bei Overflow werden älteste Events verworfen; Metrik `sse_queue_dropped_total` zählt Drops.

## Tutorial – In 5 Minuten zum ersten Plan
1. API-Key setzen: `.env` anlegen `API_TOKENS=test` oder Env Var exportieren. UI öffnen (`/ui`), rechts oben Key eingeben – Badge wird grün.
2. Policy laden: Button „Load“ im Policy-Panel (lädt Default oder lege `policy/policy.yaml` an). Bei Fehlern zeigt das Panel strukturierte Meldungen.
3. Einfachen Plan bauen: Panel „Plan Demo“ – Intent + Ziel-Datei (`api/app.py`) lassen – Button klicken. Varianten erscheinen (vorsichtig / ausgewogen / mutig) mit Parametern.
4. Live überwachen: Events-Panel öffnen (SSE) – du siehst `plan.created`, Metriken abrufen (`Metrics`).
5. Vorschlag generieren: Panel „Suggestions“ – Ziel eingeben, Generate → ID merken → optional refine (LLM) oder approve.
6. Auto-Vorschläge: Button "Auto" → statische Heuristiken erzeugen mehrere Vorschläge (keine LLM-Kosten).
7. Artefakte prüfen: JSON unter `plans/` oder `suggestions/` (Dateiname = ID). Keine direkten Code-Schreiboperationen – nur Patches.

Fertig: Du hast jetzt einen auditierten Änderungsplan + erste Verbesserungsideen im Artefakt-Verzeichnis.

## Automatische Statische Vorschläge (`/suggest/auto`)
Erzeugt mehrere gruppierte Verbesserungs-Vorschläge ohne LLM durch schnelles Scannen des Repos (Limit auf eine definierte Anzahl Dateien, ignoriert z.B. `venv/`, `node_modules/`). Jeder Vorschlag erhält Tags `auto`, `static` und durchläuft denselben Lebenszyklus wie manuelle (`/suggest/generate`).

Erkannte Heuristiken:
- TODO / FIXME / HACK Aggregation → Reduktion technischer Schulden
- Große Dateien (>= Schwellwert Zeilen) → Aufteilung / Modularisierung
- Lange Funktionen (>= Schwellwert Zeilen) → Refactoring in Hilfsfunktionen
- Breite Exception Handler (`except Exception:`) → Präzisere Fehlerbehandlung
- Duplizierte Literale (Strings >=4 Wiederholungen) → Konstante / Config extrahieren

Returned Structure: Liste von vollständigen `Suggestion` Objekten.

Vorteile:
- Zero-Kosten Basisanalyse vor LLM Nutzung
- Schneller Überblick über strukturelle / hygienische Baustellen
- Event-Integration (`suggest.generated`) & Metriken (`suggestions_generated_total`, Gauge Update)

Nächste mögliche Erweiterungen:
- Automatische Impact-Schätzung pro Auto-Vorschlag
- Optionaler Risk-Score pro Heuristik
- Diff-Vorschau für triviale Text-Konstanten Konsolidierung

## How-To (Aufgabenorientiert)
- Plan-Varianten vergleichen: Nach `/plan` → Buttons der Varianten anklicken → Patch Preview / Knobs vergleichen.
- Vorschlag verfeinern: Suggestion ID laden → Instruktion im Textfeld → „Refine (LLM)“. Fallback-Heuristik greift ohne konfiguriertes LLM.
- Auto-Vorschläge generieren: Im Suggestion Tab "Auto" klicken → Liste aktualisiert sich; relevante prüfen und ggf. approven.
- Policy validieren ohne Persistenz: Inhalt editieren → „Validate“ (dry_run) – bei Erfolg „Apply“.
- Offene Vorschläge reduzieren: Liste refresh → Relevante prüfen → Approve oder Revise. Gauge `suggestions_open_total` sinkt bei Approval.
- Export statische OpenAPI: `python ops/export_openapi.py` (optional `PUBLIC_SERVER_URL` setzen).
- PR aus Plan: Button im Plan-Panel ("PR aus Plan") oder Script `scripts/new_pr_from_plan.ps1` (nutzt `/dev/pr-from-plan` Dry-Run oder tatsächlichen Branch + optional GitHub PR falls `gh` CLI verfügbar).

## Reference (Auswahl)
Sicherheitsrelevante Header:
- `X-API-Key`: Auth für alle schreibenden Routen.

Wichtige Endpoints (GET/POST):
- `/meta`, `/state`, `/events`, `/metrics`
- `/plan`, `/act`, `/turn`
- `/policy/current`, `/policy/apply`
- `/suggest/auto`, `/suggest/generate`, `/suggest/review`, `/suggest/list`, `/suggest/llm`
- `/suggest/impact` (Impact Score & Rationale eines genehmigten Vorschlags)
- `/quest/list` (abgeleiteter Status quest:* Tags)
- `/memory/list?agent_id=` (gefilterte Memory-Einträge pro Agent)
- `/thought/stream` (kategorisierter Gedankenverlauf)
- `/env/info`
- Plugin: `/game/idle/state`, `/game/idle/tick` + Hintergrund Auto-Ticks (`idle.tick.auto`)
- Plan → PR: `/dev/pr-from-plan` (POST)
- Policy schnelles Validieren: `/policy/dry-run`
- Story Meta: `/story/meta/companions`, `/story/meta/buffs`, `/story/meta/skills` (GET & POST)

Kern-Metriken:
- `plan_seconds` Histogram → Plan-Latenzen
- `suggestions_open_total` Gauge → aktuell offene (draft / revised)
- `suggestions_review_total{action}` → approve vs. revise Rate
- `policy_denied_total{reason}` → Policy Gate Wirksamkeit
- `quest_completed_total` Counter → abgeschlossene Quests (approve quest:* Vorschlag)
- `thought_category_total{category}` Counter → Verteilung Gedanken-Kategorien

SSE Events (Beispiele): `plan.created`, `suggest.generated`, `suggest.revised`, `suggest.approved`, `suggest.open`, `suggest.refined`, `policy.reload`, `idle.tick`.

Artefakt-Verzeichnisse:
- `plans/` (Plan JSON + Patches Preview)
- `suggestions/` (Suggestion JSON inkl. weaknesses, rationale, steps, potential_patches)

## Explanation (Warum diese Architektur?)
- PR-only Change Flow: Minimiert Risiko „Silent Writes“ – jede Änderung transparent über Patches.
- Gauge für offene Vorschläge: Ermöglicht klare Alert-Schwellen (Work-in-Progress-Kontrolle / Flow).
- Varianten statt Parameter-Explosion: Drei kuratierte Risikoprofile => geringere kognitive Last.
- Schwachstellen (weaknesses) im Vorschlag: Erzwingt explizite Risiko- und Qualitätsreflexion vor Umsetzung.
- SSE Heartbeats (`:keepalive`): Verhindern Idle Timeouts im Browser, stabile UX.
- Heuristisches LLM-Fallback: System bleibt funktionsfähig ohne API-Key – Developer Experience zuerst.

## Aktueller Stand (Funktionaler Umfang)
- Security Gates aktiv (API Key, Regex Risk Filter, Whitelist, optional OPA Hook)
- Plan-Varianten (safe / balanced / bold) + UI Auswahl
- Suggestions Workflow: generate → (revise/refine) → approve – inklusive Gauge & Events
- Automatische statische Vorschläge: `/suggest/auto` gruppiert Hygienethemen & erzeugt mehrere Vorschläge auf einmal
- Thought Stream: Hintergrund-"Gedanken" alle 5–10s (konfigurierbar) mit Kontext (open_suggestions, idle_tick) → Events `thought.stream`, abrufbar via `/thought/stream`, manuell triggerbar `/thought/generate`.
- Gedanken Kategorien: risk / opportunity / action / system / neutral (Heuristik) + Metrik
- LLM Integration optional (Groq) für Chat & Suggestion Refinement
- Multi-Agent Memory (`agent_id` Chat & /memory/list)
- Quests (Tags `quest:*` + Status /quest/list + Event `quest.completed` + Metrik `quest_completed_total`)
- Impact Scoring bei Approval, abrufbar `/suggest/impact?id=`
- Policy Editor (Load, Validate, Apply) mit Inline-Fehlern
- Idle Game Plugin (State + Tick + Events + Auto-Ticks bei Inaktivität)
- Plan → PR Workflow (Endpoint + PowerShell Script)
- Policy Dry-Run Endpoint
- Observability: Prometheus + strukturierte JSON Logs + SSE Stream
- Exportierbare OpenAPI (statisch + live) mit Security Schemes & Servers
- UI Tabs (Deutsch) + verbesserter Diff Viewer (Zeilen, Farben, Navigation, Regex Suche)

## UX / A11y Notes
- Eingaben besitzen semantische Labels (Browser DevTools prüfen) – zukünftige Erweiterung: ARIA-Live Bereich für Events.
- Lange Tokens werden via CSS (wrap/break) gekappt (geplant: separate Utility-Klasse – TODO).
- Lokaler Storage nur für API Key – kein Persist persönlicher Daten.

## Addendum – Schneller Überblick (4 Schritte)
1. API-Key eingeben (oben rechts) → Badge wird grün.
2. Policy laden & Validate (Dry-Run) → Status-Badge „policy: valid“.
3. Plan anlegen (Intent + Zielpfade) → Varianten erscheinen + Slider nutzen.
4. Variante prüfen → optional „PR aus Plan“ (Dry-Run zuerst) → Branch / (PR URL) im Output.

### PlanResponse Beispiel
```json
{
	"status": "created",
	"artifact": "plans/plan_1736531111.json",
	"variants": [
		{"id":"v-safe","label":"vorsichtig","risk_level":"low","knobs":{"depth_limit":2,"risk_budget":"low"},"summary":"Minimaler Scope","patch_preview":"diff --git ..."},
		{"id":"v-balanced","label":"ausgewogen","risk_level":"medium","knobs":{"depth_limit":3,"risk_budget":"medium"},"summary":"Ausbalanciert"},
		{"id":"v-bold","label":"mutig","risk_level":"high","knobs":{"depth_limit":5,"risk_budget":"high"},"summary":"Aggressive Erweiterung"}
	]
}
```

### PR aus Plan (Request)
```json
{
	"intent": "demo feature",
	"variant_id": "v-safe",
	"dry_run": true
}
```
Antwort (Dry-Run): `{ "status": "dry-run", "branch": "plan/demo-feature-20250110-101500" }`

### Architekturfluss
Intent → Plan (Varianten) → Auswahl (Slider) → Optionaler Review & Patch-Vorschau → PR Erstellung → Review → Merge.

### Variante wählen (UI)
- Slider löst internes Event `variant.selected` aus (lokale Badge Aktualisierung)
- Buttons der Liste bleiben für direkten Vergleich nutzbar.

### Tipps für Einsteiger
- Erst Dry-Run (Policy, PR) nutzen.
- Events-Tab früh öffnen: `plan.created`, `idle.tick.auto` zeigen Aktivität.
- `metrics` im UI nur gekürzt – vollständige Ausgabe direkt via GET /metrics.

### Smoke Tests (Empfohlen minimal)
| Ziel | Route | Erwartung |
|------|-------|-----------|
| Auth Gate | POST /plan ohne Key | 401/403 |
| Plan Flow | POST /plan mit Key | 200 + variants[] |
| OpenAPI | GET /openapi.json | securitySchemes vorhanden |
| SSE | GET /events | event: ready + Heartbeats |
| Metrics | GET /metrics | Prometheus Text (# HELP) |

Diese Tabelle dient als Referenz für CI-Fehlschläge.

## Offene Vorhaben / Roadmap (Kurz)
- PR-Erstellung aus Plan (Branch, Commit, GitHub API Integration)
- Side-by-Side Diff + Inline Kommentaranker
- Idle Game Progression / zusätzliche Quests
- Policy Templates & Wizard
- Rate Limit Persistenz (Redis)
- Benchmark / Turn Execution Queue
- Thought Stream Feintuning (LLM, Sentiment, Trends)

---

Die folgenden Abschnitte spiegeln weiterhin die ursprüngliche Zielsetzung wider und bleiben zur Referenz erhalten.

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
- `suggestions_generated_total`
- `suggestions_review_total{action}` (approve|revise)
- `suggestions_open_total` (Gauge – aktuell offene Vorschläge)
- `quest_completed_total`
- `thought_category_total{category}`

## SSE
`GET /events` emits events + `retry:` hint and `:keepalive` comment every 15s.

Neue Events:
- `quest.completed`
- `thought.stream` (mit category Feld)
- `suggest.open` (Open Count Broadcast)

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

## Copilot Instructions
See `copilot-instructions.md` for a PR roadmap (Plan-Varianten, PR-only flow, Idle-Game plugin, Telemetry, and tests/acceptance).

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

## Sprint Addendum (Aktuelle Erweiterungen)
1. Impact Scoring: Score (0–100) + Rationale & approved_at beim ersten Approve erzeugt.
2. Multi-Agent Memory: Isolation je `agent_id` über Chat & /memory/list Filter.
3. Quests: Vorschläge mit Tag `quest:*` → Status pending/done; Metrik & Event.
4. Gedanken Kategorien: Einfache Keyword-Heuristik → Metrik Aggregation.
5. Deutsche Tab-UI: Klarere Navigation, reduzierte kognitive Last.
6. Diff Viewer: Zeilennummern, Farben, Hunk-Navigation, Regex Highlight.
7. Automatische statische Vorschläge: `/suggest/auto` heuristische Hygiene-Checks.

Snippets:
```
curl -H "X-API-Key: test" http://localhost:8000/suggest/impact?id=<ID>
curl -H "X-API-Key: test" http://localhost:8000/memory/list?agent_id=alpha
curl -H "X-API-Key: test" http://localhost:8000/quest/list
curl -H "X-API-Key: test" http://localhost:8000/thought/stream?limit=10
curl -H "X-API-Key: test" -X POST http://localhost:8000/suggest/auto
```

## Operational Alerts (Prometheus Beispiele)

Beispiel-Regeln (Prometheus `rules.yml` Ausschnitt) zur Überwachung des Suggestion-Lebenszyklus & Policy Gate:

```
groups:
	- name: life_agent_core
		rules:
			- alert: VieleOffeneVorschlaege
				expr: suggestions_open_total > 10
				for: 5m
				labels:
					severity: warning
				annotations:
					summary: Zu viele offene Vorschläge (>10)
					description: Bitte Vorschläge prüfen und genehmigen oder schließen.

			- alert: StauInReview
				expr: (suggestions_open_total > 0) and on() (rate(suggestions_review_total{action="approve"}[30m]) < 0.1)
				for: 30m
				labels:
					severity: warning
				annotations:
					summary: Kaum Freigaben
					description: Offene Vorschläge werden nicht genehmigt (Approve-Rate <0.1 / 30m).

			- alert: PolicyDenialsSpike
				expr: increase(policy_denied_total[10m]) > 5
				for: 2m
				labels:
					severity: critical
				annotations:
					summary: Policy Denials Spike
					description: Mehr als 5 Policy-Verweigerungen in 10 Minuten – mögliche Fehlkonfiguration oder Angriff.

			- alert: LLMFehlt
				expr: absent(llm_requests_total)
				for: 15m
				labels:
					severity: info
				annotations:
					summary: Keine LLM Nutzung
					description: Noch kein LLM aktiv (oder Metrik fehlt). Optional – nur zur Sichtbarkeit.

	- name: performance
		rules:
			- alert: LangsamePlanerstellung
				expr: histogram_quantile(0.95, sum(rate(plan_seconds_bucket[5m])) by (le)) > 3
				for: 10m
				labels:
					severity: warning
				annotations:
					summary: 95%-Latenz der Planerstellung hoch
					description: P95 von /plan > 3s (letzte 5m). Prüfe LLM / IO.
```

Hinweise:
- `suggestions_open_total` wird bei jeder Änderung (Generate, Revise, Approve) aktualisiert und via SSE Event `suggest.open` gespiegelt (`open_count=<n>`).
- Für Alertmanager kann ein Route basierend auf `severity` eingerichtet werden.
- Diff / Patch Inhalte bewusst nicht in Metrics – lieber über Artefakte & Repos prüfen.

Dashboards (Grafana) – empfohlene Panels:
- Current Open Suggestions (Gauge)
- Approvals vs Revisions (stacked rate)
- Plan Duration P95
- Policy Denials (rate)

## Story System (Lebendige Erzählung)
Ein persistenter narrativer Zustand, der sich über Aktionen und Zeit entwickelt. Ressourcen (deutsche Keys) steuern Heuristiken für Optionen und LLM-Erzähltexte.

### Ressourcen (Keys)
| Key | Bedeutung | Typischer Start |
|-----|-----------|----------------|
| energie | Kurzfristige Handlungs-/Aufmerksamkeitsenergie | 80 |
| wissen | Akkumuliertes strukturiertes Wissen | 0 |
| inspiration | Roh-Ideen / kreative Impulse | 0 |
| ruf | Außenwirkung / Reputation | 0 |
| stabilitaet | Innere Ordnung / System-Stabilität | 80 |
| erfahrung | Fortschritt / Erfahrungspunkte | 0 |
| level | Progressionsstufe (Skalierung XP-Schwelle = level * 100) | 1 |

### Endpoints
- `GET /story/state` → Gesamter Zustand inkl. aktueller Optionsliste.
- `GET /story/log?limit=50` → Chronologisch sortierte Events (tick / action / future: arc_shift ...).
- `GET /story/options` → Nur aktuelle Optionen (falls UI periodisch pollt oder SSE nutzt).
- `POST /story/action` `{ option_id? , free_text? }` → Wendet Option an ODER erzeugt Freitext-Aktion (leichter Inspiration+ Energie-Tradeoff).
- `POST /story/advance` → Zeit-Fortschritt (passiver Tick, Energie-Decay, Option-Refresh, LLM-Erzähl-Satz).
- `POST /story/options/regen` → Forciert Regeneration (Heuristik neu, z.B. nach externen State-Änderungen).
- `GET /story/meta/companions|buffs|skills` → Aktuelle Meta-Ressourcen (persistente Erweiterungen des Zustands).
- `POST /story/meta/companions` `{ name, archetype?, mood?, stats? }` → Neuen Begleiter einfügen.
- `POST /story/meta/buffs` `{ label, kind?, magnitude?, expires_at?, meta? }` → Temporären Buff einfügen.
- `POST /story/meta/skills` `{ name, category?, level?, xp? }` → Skill anlegen (Default level=1,xp=0).

Alle POST-Operationen benötigen `X-API-Key`.

### Eventtypen (StoryEvent.kind)
- `action` (aus gewählter Option oder Freitext)
- `tick` (Zeitverlauf)
- `arc_shift` (Arc-Wechsel durch Level-Schwelle)
- (geplant) `milestone`, `level_up`

### SSE Events
- `story.event` → Payload = StoryEvent
- `story.state` → z.Z. Minimal (epoch / option count). Erweiterbar für UI Instant-Refresh.
- `story.meta.*` → Bei CRUD Aktionen: `story.meta.companion.add`, `story.meta.buff.add`, `story.meta.skill.add` (liefert jeweiliges Objekt)

### Heuristische Optionen (MVP)
Regeln (vereinfachte Beispiele):
- Niedrige `energie` < 40 → Meditation (Energie +15, Inspiration +2)
- Hohe `inspiration` > 10 & `wissen` < 50 → Strukturieren (Inspiration -> Wissen + Erfahrung)
- `erfahrung` >= `level * 100` → Level-Up (XP Reset Teil, +Stabilität)
- Fallback: Exploration (Inspiration +5, Energie -5, Erfahrung +3)

### LLM Integration
Wenn `GROQ_API_KEY` gesetzt → Kurzer deutscher Erzähl-Satz pro Aktion/Tick (Model Default: `llama-3.3-70b-versatile` oder Policy Override). Fallback ohne Key: abgeschnittener Prompt-Inhalt.

### Metriken (Prometheus)
- `story_events_total{kind}` → Anzahl pro Eventtyp
- `story_options_open` (Gauge) → Aktuelle Anzahl generierter Optionen

### Beispiel Flow (curl)
```
# Zustand holen
curl -H "X-API-Key: test" http://localhost:8000/story/state
# Optionen anzeigen
curl -H "X-API-Key: test" http://localhost:8000/story/options
# Aktion aus erster Option
OPT=$(curl -s -H "X-API-Key: test" http://localhost:8000/story/options | python -c "import sys,json;d=json.load(sys.stdin);print(d[0]['id'])")
curl -X POST -H "X-API-Key: test" -H "Content-Type: application/json" \
  -d "{\"option_id\":\"$OPT\"}" http://localhost:8000/story/action
# Freitext Aktion
curl -X POST -H "X-API-Key: test" -H "Content-Type: application/json" \
  -d '{"free_text":"kurzes Notat bündeln"}' http://localhost:8000/story/action
# Zeit voranschreiten
curl -X POST -H "X-API-Key: test" http://localhost:8000/story/advance
```

### Persistenz
SQLite Tabellen:
- `story_state(id=1)` – Singleton (epoch, mood, arc, resources JSON)
- `story_events` – Verlauf
- `story_options` – Kurzlebige aktuelle Auswahl
- `story_companions` – Persistente Begleiter (name, archetype, mood, stats JSON)
- `story_buffs` – Temporäre Modifikatoren (label, kind, magnitude, expires_at, meta JSON)
- `story_skills` – Skills mit Fortschritt (name, level, xp, category)

### Geplante Erweiterungen
- Erweiterte Arc-Narration (adaptive Texte pro Arc)
- Buff-Expiry-Verarbeitung & automatische Entfernung abgelaufener Buffs
- Skill-XP Zuwachs über Aktionen (automatische Progression)
- Option-Scoring & Risikoexplizierung
- LLM Prompt Feintuning (Kontextkompression, Mood-Einflüsse)
- Option TTL / automatische Veralterung
- Tests: Edge-Cases (Level-Up, mehrfacher Tick ohne Aktionen, Migration alter englischer Keys)

### Meta Ressourcen Details

Die Meta-Ressourcen erweitern den narrativen Zustand um längerfristige Progressionselemente, die unabhängig von kurzfristigen Ressourcen wirken.

| Typ | Endpoint (GET/POST) | Primäre Felder | Beschreibung |
|-----|---------------------|----------------|--------------|
| Companion | `/story/meta/companions` | name, archetype, mood, stats{} | Dauerhafte Gefährten mit frei strukturierbaren Stats (JSON) |
| Buff | `/story/meta/buffs` | label, kind, magnitude, expires_at, meta{} | Zeitlich begrenzte Effekte (Client kann nach Ablauf filtern; zukünftige automatische Purge geplant) |
| Skill | `/story/meta/skills` | name, level, xp, category | Fortschrittsfähigkeiten; XP/Level Logik kann später automatisiert wachsen |

Beispiel POST Companion:
```
curl -X POST -H "X-API-Key: test" -H "Content-Type: application/json" \
	-d '{"name":"Archivarin","archetype":"wissend","mood":"ruhig","stats":{"analyse":5}}' \
	http://localhost:8000/story/meta/companions
```

Antwort (vereinfachtes Schema):
```
{
	"id": 5,
	"name": "Archivarin",
	"archetype": "wissend",
	"mood": "ruhig",
	"stats": {"analyse":5},
	"acquired_at": 1757772000.12
}
```

UI-Hinweis: Aktuell werden Meta-Listen nur via `/story/state` geliefert (companions, buffs, skills Arrays). Eine direkte Darstellung im Inline-HTML folgt optional.

### Aktueller Implementierungsstand (Story-Core)
- Arc-Wechsel aktiv (level >=3 → exploration, level >=10 → mastery) erzeugt `arc_shift` Event.
- Option TTL Mechanik vorhanden (Expiration Filter beim Laden – automatische Generierung bei Regeneration / Tick).
- Meta CRUD Endpoints senden SSE Events (`story.meta.*`) + `story.state` Trigger für UI Refresh.
- Tests decken State, Optionen, Actions, Arc-Shifts und Meta-CRUD (Seeds + Create) ab.


## Policy Templates & Wizard (Spezifikation – in Arbeit)

Dieser Abschnitt dokumentiert das geplante Policy-Template-System bevor die Implementierung erfolgt. So kann die eigentliche Backend-Integration konsistent und „schema-first" erfolgen.

### Ziele
* Wiederverwendbare, kuratierte Start-Policies (Solo, Team, Sandbox)
* Guided Wizard: Benutzer-Ziele + optionaler Risikoprozentsatz → abgeleitete finale Policy
* LLM (optional) darf nur Ergänzungen (Kommentare, rationale) liefern – keine unkontrollierten Felder

### Template Speicherort
`policies/templates/*.yaml`

Aktuell vorhanden:
* `solo_dev.yaml` – Minimaler Sicherheitsrahmen für Einzelentwickler
* `team.yaml` – Kollaborationsregeln (Branch Prefixes, Reviewer Mindestanzahl)
* `sandbox.yaml` – Schnelles Experimentieren (lockerer, Fokus auf Credential-Schutz)

### Gemeinsame Template Felder
```yaml
version: 1
name: <string>
allowed_dirs: [list von Pfaden]
rules:                # polymorph, engine-spezifisch
	- id: <string>
		description: <string>
		match|condition: <regex oder Ausdruck>
		action: allow|deny|review|escalate
llm:                  # optional
	model: <string>
	temperature: <float>
	max_tokens: <int>
branching:            # optional (team)
	require_prefixes: [feat/, fix/, chore/]
reviews:              # optional (team)
	min_reviewers: 2
	required_labels: [reviewed]
```

### Wizard Endpoint (geplant)
`POST /policy/wizard`

Request Body (JSON):
```json
{
	"template": "solo_dev",        "goals": ["schneller review", "klare sicherheitsgrenzen"],
	"risk_profile": "low|medium|high", "overrides": {"allowed_dirs": ["api", "ui", "policies"], "llm": {"model": "llama-3.3-70b-versatile", "temperature": 0.0}},
	"annotate": true
}
```

Response (200):
```json
{
	"source_template": "solo_dev",
	"policy": {
		"version": 1,
		"name": "solo_dev_baseline",
		"allowed_dirs": ["api", "policies", "ui"],
		"rules": [ {"id": "deny_secrets", "action": "deny", "description": "..."} ],
		"llm": {"model": "llama-3.3-70b-versatile", "temperature": 0.0, "max_tokens": 800},
		"_meta": {"goals": ["schneller review", "klare sicherheitsgrenzen"], "risk_profile": "low", "generated_at": "2025-09-14T10:15:00Z"}
	},
	"diff": {"added_rules": [], "changed": {"allowed_dirs": {"from": ["api","policies","ui"], "to": ["api","policies","ui"]}}, "removed_rules": []},
	"notes": ["Risk Profile low → erzwingt niedrige LLM Temperatur"]
}
```

Error Cases:
* 404 Template nicht gefunden
* 422 Ungültige Overrides (Schema-Verstoß)
* 429 Zu viele Wizard-Aufrufe (Rate Limit – geplant)

### Verarbeitungsschritte (geplant)
1. Template laden (Datei → Parse YAML)
2. Validierung gegen internes Policy-Schema
3. Goals Normalisierung (lowercase, trim)
4. Heuristische Anpassungen anhand `risk_profile` (z.B. `max_diff_lines`, zusätzliche review-Regeln)
5. Overrides anwenden (whitelist-basierte Feld-Merge)
6. Optional: Annotation (LLM → nur erläuternde Strings, keine Strukturänderung)
7. Diff berechnen (Vorher/Nachher) → `diff` Objekt
8. Response signieren (X-Request-Id Header bleibt Quelle der Korrelation)

### Sicherheitsleitplanken
* Keine dynamische Code-Auswertung in `condition`
* Regex Sandbox (Timeout, Kompilierung vor Nutzung)
* LLM darf keine neuen Root-Felder erzeugen (Strict Merge)
* Overrides nur in expliziter Allowlist: `allowed_dirs`, `rules`, `llm`, `branching`, `reviews`

### Entscheidungspunkte (offen)
| Thema | Option A | Option B | Status |
|-------|----------|----------|--------|
| Diff Format | Einfach (added/removed/changed) | Voll AST mit Kontext | A (vorerst) |
| Rules Engine | Regex + einfache Conditions | OPA / Rego Integration | A (MVP) |
| LLM Nutzung | Optional Annotation | Vollständige Rule Synthese | A (konservativ) |
| Persistenz Ergebnis | Nein (nur Return) | Speichern unter `policy/generated/*.yaml` | Offen |
| Rate Limit | Global Counter | Per API Key Window | Offen |

### Beispiel cURL (geplant)
```bash
curl -X POST http://localhost:8000/policy/wizard \
	-H "X-API-Key: test" \
	-H "Content-Type: application/json" \
	-d '{
				"template": "solo_dev",
				"goals": ["klarheit", "schnelle iteration"],
				"risk_profile": "low",
				"annotate": true
			}'
```

### Nächste Implementierungsschritte
1. Endpunkt + Schema Klassen (Pydantic) anlegen
2. Template Loader + Cache
3. Risk Profile Heuristik Mapping
4. Diff Utility
5. (Optional) Annotation Adapter (call_groq_json)
6. Tests (Template existiert / nicht gefunden, Overrides, Risk Profile)

---

Hinweis: Diese Spezifikation dient als Vorlauf. Implementierung folgt, sobald Backend-Kontext bestätigt wurde.


