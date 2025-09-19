# Copilot System Instructions — Life-Agent v7

## Mission
Baue ein **selbst-erweiterndes System**, das sich über **geplante Vorschläge → Patches → Pull Requests → Reviews → Deploy** iterativ verbessert.
- **Keine** direkten Commits auf `main` durch Automationen. Immer Branch + PR.
- Fokus: klare Architektur, Tests, Telemetrie, UI-Signale (Plan, Suggestions, Policies, Events, Memory, Quests).

## Leitplanken (unverhandelbar)
- **Sicherheit:** Nur schreibende Aktionen via PR. Gatekeeping durch Policies/Checks.
- **Nachvollziehbarkeit:** Jede Änderung enthält: Problem, Hypothese, Risiko, Metrik, Rollback-Plan.
- **Idempotenz & Dry-Run:** Tools müssen Vorschau (Diff/Plan) liefern, bevor sie echte Wirkung haben.
- **Konfiguration per Code:** `./config/*.yaml` steuert Modelle, Limits, Policies, Feature-Flags.

## Architektur-Ziele
1. **Core Service (Node/TS)**  
  - Endpunkte: `/plan`, `/suggest`, `/patch/preview`, `/patch/apply (PR)`, `/events`, `/metrics`, `/memory`, `/quests`.  
  - Adapter: `ai/groqAdapter.ts` (OpenAI-kompatibel), `git/githubAdapter.ts`, `deploy/renderAdapter.ts`.
  - Job-Queue (z. B. BullMQ) für langlaufende Tasks.
  - Testbar, mit `vitest` + `supertest`.

2. **VS Code Extension (optional, aber erwünscht)**
  - Feature: Commands „Life-Agent: Plan“, „Preview Patch“, „Create PR“, Panel mit Events/Metriken.
  - Webview für Dashboards; Kommunikation per `vscode.postMessage`.
  - Extension redet **nur** mit dem Core-Service (lokal/remote), keine Secrets im Client.

3. **GitHub Integration**
  - PR-Erstellung über REST API.
  - Labels: `agent:proposal`, `risk:low|med|high`, `type:refactor|feature|fix`.
  - Review-Kommentare generieren (strukturierte Checkliste).

4. **Deploy**
  - Render Web Service (Node). Autodeploy auf `main` oder `release/*`. Healthcheck.
  - Env: `GROQ_API_KEY`, `GITHUB_TOKEN`, `RENDER_SERVICE_ID` etc. via `.env` (nie commiten).

5. **Observability**
  - Events/Metrics: Request-Latenzen, PR-Zykluszeiten, Akzeptanzrate, Revert-Rate, Modellkosten, Token.

## Definition of Done (PR)
- CI: Lint, Typecheck, Tests grün.
- Änderungs-Spec im PR-Body (Template): Problem → Ansatz → Risiko → Messung → Rollback.
- Changelog-Eintrag + Docs aktualisiert.
- Manual QA Schritte dokumentiert.

---

## Tasks — bitte in dieser Reihenfolge vorschlagen & umsetzen

### 0) Repo-Härtung
- `/.github/PULL_REQUEST_TEMPLATE.md` mit obiger Checkliste.
- CI (GitHub Actions): `lint.yml`, `test.yml`, `typecheck.yml`.
- `commitlint` + Conventional Commits.

### 1) Core-Scaffold
- Packages: `typescript`, `tsx`, `express`, `zod`, `vitest`, `supertest`, `dotenv`, `pino`, `bullmq`, `ioredis`.
- Ordnerstruktur:


/src
/adapters/ai/groqAdapter.ts
/adapters/git/githubAdapter.ts
/adapters/deploy/renderAdapter.ts
/domains/{plan,suggest,patch,policy,memory,quests}/...
/infra/{http,queue,logger,config}.ts
/telemetry/{events,metrics}.ts
/config/{app.yaml,policies.yaml,models.yaml}

- Express-Server + `/healthz`.

### 2) Groq-Adapter
- OpenAI-kompatible Chat-Completions; `GROQ_API_KEY` aus Env; Timeout/Retry/Rate-Limit.
- Utility: `structuredLlmCall<T>(schema: zod.Schema<T>)` für **JSON-sichere** Antworten.

### 3) GitHub-Adapter (PR-only)
- Funktionen:
- `createBranchFromDefault()`
- `openPullRequest({title, body, head, base})`
- `commentOnPR(pr, markdown)`
- `getRepoTree() / applyPatchAsBranch(diff)`
- Dry-Run: Patch nur als Diff zurückgeben.

### 4) Render-Adapter
- `triggerDeployIfMerged()` (Webhook-kompatibel) + Status-Polling.
- Healthcheck-Verifier.

### 5) Policy Engine
- Lese `config/policies.yaml`:  
- Risk-Gate: max Lines Changed, verbotene Pfade (z. B. Secrets), erlaubte Domains.
- Review-Escalation: `risk:high` → Draft-PR + Pflichtreview.
- `evaluatePatch(diff): {risk, violations[], recommendation}`.

### 6) Plan → Suggest → Patch → PR Flow
- `/plan`: nimmt Ziel + Kontext; liefert Meilensteine, Tasks, Risiken, Metriken.
- `/suggest`: kleinteilige Vorschläge mit Files/Reasoning.
- `/patch/preview`: generiert unified diff.
- `/patch/apply`: legt Branch an, commit diff, erstellt PR, kommentiert Spec.

### 7) VS Code Extension
- Kommandos:  
- `lifeAgent.plan` → öffnet Panel mit Plan.  
- `lifeAgent.previewPatch` → zeigt Diff, Buttons: „PR erstellen“, „Verwerfen“.  
- `lifeAgent.openDashboard` → Webview mit Events/Metrics.
- Settings: Service-URL, Token.

### 8) UI-Dashboard (bestehend erweitern)
- Tabs: Plan, Suggestions, Policies, Events, Metrics, Memory, Quests.
- SSE/WebSocket für Live-Events.
- „Risk Heatmap“ + „PR Funnel“ Charts.

### 9) Telemetrie & QA
- Metriken definieren:  
- `pr_time_to_merge`, `pr_rejection_rate`, `revert_rate`, `avg_patch_size`, `policy_violation_rate`.
- E2E-Happy Path Test (lokal): Plan→Suggest→Preview→PR (Draft)→Close.

---

## Prompt-Formate (für LLM-Aufrufe)

### System Prompt (Planung)


Du bist ein Software-Planner. Ziel: minimal-risk, testbare Schritte.
Gib strukturierte JSON-Antwort:
{
"milestones": [{ "name": "", "why": "", "acceptance": [] }],
"tasks": [{ "id": "", "title": "", "steps": [], "risk": "low|med|high", "metrics": [] }],
"constraints": ["PR-only", "policy:<...>"]
}
Halte dich strikt an JSON; keine freien Texte außerhalb der Felder.


### System Prompt (Patch-Vorschlag)


Du bist ein Refactor-/Feature-Generator. Liefere NUR:
{
"changes": [
{"file": "path", "operation": "replace|insert|delete", "at": "marker|range", "content": "..." }
],
"rationale": "...",
"risk": "low|med|high"
}


---

## Konfiguration (Beispiel)
```yaml
# config/models.yaml
llm:
  provider: groq
  model: llama-3.1-70b-instruct
  max_tokens: 2000
  temperature: 0.3

# config/policies.yaml
risk:
  maxChangedLines: 400
  forbiddenPaths: ["**/*.env", "secrets/**"]
  requireDraftForHighRisk: true
  reviewers:
   high: ["owner1","owner2"]
   default: ["owner1"]

CI/CD

Actions:

ci/lint.yml: pnpm eslint .

ci/test.yml: pnpm vitest run

ci/typecheck.yml: tsc -p tsconfig.json

Deploy: Autodeploy in Render nach Merge auf main oder Release-Branch.
```

Nächste sinnvolle PRs (bitte erzeugen)

feat(core): scaffold express + config + logger + healthz

feat(ai): groqAdapter with structured JSON helper

feat(git): PR-only adapter + dry-run

feat(policy): YAML policies + evaluator

feat(flow): /plan /suggest /patch/preview /patch/apply

feat(vscode): minimal extension commands + webview shell

feat(ui): events + metrics panels

chore(ci): actions + commitlint + templates

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
