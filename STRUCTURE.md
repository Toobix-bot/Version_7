# Geplante Code-Struktur (Roadmap)

Ziel: `api/app.py` entlasten, Verantwortlichkeiten klar trennen, Testbarkeit & Erweiterbarkeit erhöhen.

## Leitprinzipien
- Single Responsibility pro Modul (Routing, Services, Modelle, Infra)
- Keine zirkulären Importe
- Services enthalten reine Logik (keine FastAPI-Objekte)
- Router minimal (Parsing, Aufruf Service, Response-Form)
- Events + Metrics zentralisiert
- Variant-Engine wiederverwendbar (Plan, Chat, Suggestions)

## Ziel-Verzeichnisbaum
```
api/
  app.py                # Bootstrap, FastAPI Instanz, Includes
  core/
    infra.py            # DB Connection, init_db, emit_event, background tasks hooks
    events.py           # Event Queue & SSE Handling
    metrics.py          # Prometheus Counter/Histogram Definition
    security.py         # Auth, rate_limit, risk_gate, whitelist enforcement
  models/
    plan.py             # PlanRequest/Response, PlanVariant
    suggestion.py       # Suggestion Models
    policy.py           # PolicyWizardRequest etc (oder reuse from policy.model)
    idle.py             # IdleQuest Modelle
    story.py            # Story State / Event Modelle
    multiio.py          # MultiIORequest + helper
  services/
    variants.py         # build_variants(kind, base, profiles)
    planner.py          # plan generation logic (artifact write, variant invocation)
    suggestions.py      # generation, refine, impact scoring
    policy_wizard.py    # load_template, apply_risk_profile, compute_diff
    idle.py             # quest seed, progress update
    story.py            # story progression, option refresh
  routes/
    plan.py             # /plan
    suggestions.py      # /suggest/*
    policy.py           # /policy/* (reload, current, dry-run, apply)
    wizard.py           # /policy/wizard
    idle.py             # /game/idle/*
    story.py            # /story/*
    glossary.py         # /glossary
    meta.py             # /meta, /state

policy/
  model.py
  loader.py

policies/
  templates/*.yaml

```

## Migrationsphasen
1. Phase 1 (Minimal Extraction)
   - `core/infra.py`: `init_db`, globale `db_conn`, `emit_event` (aktuelles SSE-Emitter nutzt Queue – verschieben oder Wrapper in events.py)
   - `core/security.py`: `require_auth`, `rate_limit`, `risk_gate`, `enforce_whitelist`
   - `services/policy_wizard.py`: Extrahiere Wizard-Hilfsfunktionen
   - Router: `routes/wizard.py` + Include in `app.py`

2. Phase 2 (Plan & Variants)
   - Auslagerung Plan-Logik in `services/planner.py`
   - `models/plan.py` + Nutzung in Route
   - Entferne Plan-spezifische Inline-Funktionen aus `app.py`

3. Phase 3 (Suggestions & Impact)
   - Move suggestion endpoints + logic (services + routes)
   - Impact Scoring in eigenem Service-Modul

4. Phase 4 (Story & Idle)
   - Story Kernlogik in `services/story.py`
   - Idle Quests in `services/idle.py`
   - Tabellen-Init in infra statt app

5. Phase 5 (Events & SSE)
   - `core/events.py`: Kapselt Queue, SSE Filter, Heartbeats
   - `app.py` nur noch: `from api.core.events import sse_router` oder Inline Factory

6. Phase 6 (Cleanup / MultiIO / Variant Reuse)
   - MultiIO in `models/multiio.py`
   - `services/variants.py` wird von Planner, Suggestions, Chat aufgerufen

## Konkrete Schnittstellen (Contracts)
- `emit_event(kind: str, payload: dict) -> None`
- `build_variants(kind: str, base: dict, profiles: list[dict]) -> list[dict]`
- `plan_service.generate(intent: str, context: str, targets: list[str]) -> PlanResult`
- `wizard.generate(req: PolicyWizardRequest) -> PolicyWizardResponse`
- `idle.advance(delta: int=1) -> list[QuestProgressChange]`
- `story.advance_tick(mode: str='normal') -> StoryEvent`

## Events Normalisierung
Empfehlung künftige Namenskonvention:
- plan.created
- suggestion.generated / suggestion.refined / suggestion.approved
- policy.reload / policy.apply
- wizard.generated
- idle.quest.progress / idle.quest.completed
- story.event / story.state

## Teststrategie nach Refactor
- Unit: services.* (keine FastAPI Abhängigkeit) → direkter Aufruf
- Integration: routes/* via TestClient
- Migration Safety: Alte Tests unverändert lassen; Pfade & Response-Struktur kompatibel halten

## Technische Schulden Geplant
- Großes `app.py`: schrittweise schrumpfen, kein Big Bang
- Mixed Deutsch/Englisch Keys: später vereinheitlichen (Konfiguration Flag?)
- Error Format uneinheitlich (detail vs error wrapper): zentralisieren in `core/security.py`

## Nächste Sofortaktionen
1. Anlegen Verzeichnisgerüst (`core`, `services`, `routes`, `models`)
2. Verschiebe Wizard Hilfsfunktionen → `services/policy_wizard.py`
3. Erstelle `routes/wizard.py` → importiert Service, registriert Router
4. Entferne Wizard-Code aus `app.py` (Adapter-Aufruf)

Danach: Plan-Service.

## Naming & Style
- Dateien: snake_case
- Pydantic Modelle: PascalCase + `model_config = ConfigDict(...)` (optional später)
- Services: einfache Funktionen, side-effects (DB) klar dokumentiert

## Offene Punkte
- Kontext für DB Connection weiterreichen (global vs. dependency) → später Dependency Injection
- Hintergrund-Tasks (Idle Ticks) in separaten Startup Hook verlagern
- Metrics Label Cardinality prüfen (risk_level Strings reduzieren?)

---
Letzte Aktualisierung: 2025-09-14
