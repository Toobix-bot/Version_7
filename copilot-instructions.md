# copilot-instructions.md

## Kontext
Repo: `Version_7` – FastAPI-Server mit OpenAPI, SSE-Events, Prometheus `/metrics`, YAML-Policies (Hot-Reload), Risk/Whitelist-Gates, optionalem OPA-Gate, PR-only-Plänen, Groq-LLM.  
Guardrails: **PR-only** (keine Direkt-Writes), **API-Key/Bearer** geschützt, deterministische Defaults (temp=0), Policy-Whitelist.

---

## Zielbild
Eine **self-observing Entwicklungsumgebung**, die
1) ihre Umgebung liest (Telemetry, Lint/Type/Deps),  
2) daraus **Plan-Varianten** (A/B/C) generiert (LLM=Groq),  
3) **nur** PRs erstellt (GitHub CLI),  
4) wahlweise ein **Idle-Game** als Plugin ausführt und dessen Regeln schrittweise (PR-only) anpasst,  
5) alles live über **SSE** & **Prometheus** sichtbar macht.

---

## PR-1 — Plan-Varianten & Regler-„Knobs“
**Aufgabe**
- `/plan` so erweitern, dass **mehrere Varianten** zurückkommen, z. B. `variants=[{id, knobs, summary, patch_preview}]`.  
- Query-/Body-Parameter für **Knobs** (z. B. `temperature`, `risk_budget`, `depth_limit`).  
- In OpenAPI: `response_model` + Beispiel-Responses je Variante.

**Akzeptanz**
- `/openapi.json` zeigt `PlanResponse` mit `variants[]`.  
- `/plan?risk_budget=low` liefert andere Variante als `risk_budget=high`.  
- LLM-Aufruf über **Groq-SDK** (kein OpenAI). Defaults: `model="llama-3.3-70b-versatile"`, `temperature=0`.  
Quellen: Groq Chat Completions & Modelle.

---

## PR-2 — GitOps-Ausführung: **PR-only**
**Aufgabe**
- Script `scripts/create_pr_from_plan.py`: ruft `/plan`, schreibt Artefakte in einen Branch, erstellt **Pull Request** mit `gh pr create` (Titel, Body, Labels).  
- Endpoint `/plan/apply` (optional) triggert nur das Script – **keine** direkten File-Writes.

**Akzeptanz**
- `gh pr create` wird mit `--title/--body` korrekt aufgerufen; URL im Log.  
- PR enthält Plan-Zusammenfassung (What/Why/How) + Checkliste.  
Quellen: GitHub CLI `gh pr create` & PR-Workflow.

---

## PR-3 — Idle-Game als Plugin (schrittweise Selbst-anpassung)
**Aufgabe**
- Ordner `plugins/games/idle/`.  
- Endpunkte: `GET /game/idle/state`, `POST /game/idle/tick`.  
- Jeder Tick darf **kleine Regel-Mutationen** vorschlagen → wird als **Plan-Variante** zurückgegeben; Policy-Whitelist nur `plugins/games/**`.  
- SSE-Kanal (Topic `"idle"`) sendet `event: idle-tick` + State.

**Akzeptanz**
- `/game/idle/tick` erzeugt SSE-Event; `/events` läuft stabil mit Heartbeats.  
- Policy verhindert Änderungen außerhalb der Whitelist.  
Quellen: SSE per `StreamingResponse` + EventSource.

---

## PR-4 — Umwelt lesen: Telemetry & Befunde
**Aufgabe**
- Sammle einfache **Befunde**: Lint/Typing/Dep-Audit (z. B. ruff/mypy/pip-audit) per Lightweight-Runner; aggregiere in `/meta` + erhöhe **Prometheus-Metriken** (`lint_issues_total`, `type_errors_total`, `dep_vulns_total`).  
- `/metrics` muss **CONTENT_TYPE_LATEST** ausliefern.

**Akzeptanz**
- `/metrics` zeigt neue Counter/Histogramme (z. B. `plan_seconds`).  
- `/meta` liefert letzte Befunde (Zeitstempel + Counts).  
Quellen: Prometheus Python-Client & CONTENT_TYPE_LATEST.

---

## PR-5 — Copilot-Handshake (PR-Text & Hints)
**Aufgabe**
- PR-Body so strukturieren, dass **Copilot Chat** ihn gut verwerten kann:  
  - **Kontext** (Dateien, Ziele),  
  - **Constraints** (PR-only, Policy, Whitelist),  
  - **Akzeptanzkriterien** (Tests, Lint, Typen),  
  - **Diff-Skizzen** / Pseudocode.  
- Optional: Labels „ready-for-copilot“.

**Akzeptanz**
- Manuelle Prüfung: PR-Beschreibung ist für Copilot/Reviewer klar; `gh pr view --web` zeigt strukturierten Text.  
Quellen: GitHub CLI PR-Kommandos.

---

## PR-6 — Groq-Integration (sauber & deterministisch)
**Aufgabe**
- `llm/groq_client.py` (sync/async): Wrapper für Chat Completions, **temperature=0** Default, optionale Streaming-Rückgabe.  
- System-Prompt: „cautious planner“, **JSON-Antwort** (Schema für Plan-Variante).

**Akzeptanz**
- `GROQ_API_KEY` aus ENV; Beispielscript `examples/groq_basic.py` läuft.  
- Bei Fehlern klare Logs; Timeouts begrenzen.  
Quellen: Groq SDK Quickstart & Chat Doku.

---

## PR-7 — SSE robuster machen
**Aufgabe**
- `/events` sendet beim Connect `event: ready` und regelmäßig `:keepalive\n\n`; `Content-Type: text/event-stream`.  
- Test: mind. 1 ready + 2 Heartbeats.

**Akzeptanz**
- EventSource im Browser hält >5 min; Reconnect-Verhalten ok.  
Quellen: SSE-Praxis in FastAPI/HTTP.

---

## PR-8 — Policies & OPA (optional)
**Aufgabe**
- YAML-Schema strenger validieren (saubere 422-Fehler).  
- Optionales **OPA-Gate** hinter Flag `OPA_ENABLE=true` – `opa eval` gegen `data.agent.allow`.

**Akzeptanz**
- Ungültige Policy → 422 mit Fehlerpfaden; OPA-deny → 403/422 mit Grund.  
Quellen: FastAPI Security/OpenAPI-Einbindung; OPA-Basics.

---

## Tests (Erwartungen)
- Auth-Pflicht: `/plan` ohne Key → 401/403; mit Key → OK.  
- OpenAPI: `components.securitySchemes` enthält API-Key/Bearer; `paths./plan.post.security` gesetzt.  
- Risk-Gate: verbotene Muster (z. B. `subprocess`, `../`) → 422/403.  
- Whitelist: Pfade außerhalb `allowed_dirs` → 403.  
- Policy-Reload: `invalid.yaml` → 422 mit Fehlerliste.  
- OPA-Flag: `OPA_ENABLE=true` + `opa_allow=False` → 403/422.  
- SSE: mindestens `ready` + 2 Heartbeats in kurzer Zeit.  
- Metrics: `/metrics` liefert Prometheus-Textformat (inkl. neue Zähler/Histogramme).  
Quellen: httpx **ASGITransport** für FastAPI-Tests; Prometheus-Exposition.

---

## Hinweise & Constraints
- **PR-only** bleibt in allen Pfaden bestehen (keine Direkt-Writes).  
- **Security**: Schreibende Endpunkte mit `Security(...)` absichern; FastAPI integriert Security-Schemes automatisch in OpenAPI.  
- **SSE**: `StreamingResponse(..., media_type="text/event-stream")`, Heartbeats halten Verbindungen stabil.  
- **Groq**: Nur Groq-SDK verwenden (`pip install groq`), `GROQ_API_KEY` aus ENV; Modell z. B. `llama-3.3-70b-versatile`.
