# Implementierungsplan (schrittweise)

Stand: 2025-09-19

Ziel: Risikoarm, iterativ, mit klaren Prüf- und Messpunkten.

## Phase 1 – Grundlagen & Hardening (fertig/teils erledigt)
- SSE-Verbesserungen, /metrics Content-Type korrekt
- Auth/Sessions/RBAC, UI-Anbindung
- Story-Verbesserungen (Stil, Stimmen, Zufallsereignisse)
- Policy-Templates/Wizard (Validate, Deep-Merge)
- Plan→PR End-to-End (Draft/Labels, Fallback)
- Suggestions-Endpoints + interner Probe

Messung: Tests grün, /metrics erreichbar, SSE stabil.

## Phase 2 – UI Konsolidierung & Theme/Palette (fertig)
- Design-Tokens, Theme (Hell/Dunkel), Hues + Rotation
- Inline-UI an Tokens anbinden

Messung: UI-Bedienelemente greifen Tokens, Palette persistiert.

## Phase 3 – Struktur & Navigation
- Dashboard-Ansicht (Kacheln für Health, Vorschläge, Story, PR)
- Router/Pane-Umschaltung (falls geplant)
- Öffentliche Leseansichten (Story, Health)

Messung: Navigierbarkeit, klare Einsteigeroberfläche, RBAC-basierte Sichtbarkeit.

## Phase 4 – Suggestions UI
- Liste, Filter, Detail/Review, Impact
- Metrik-Badge (Anzahl offen) aus /metrics

Messung: End-to-End Review-Fluss, Nutzungsereignisse, Metriken sichtbar.

## Phase 5 – Policies UX
- Wizard UI (Params → Dry-Run → Diff → Apply)
- Rollengestaffelte Freigaben

Messung: Erfolgreiche Apply-Vorgänge, verständliche Dry-Run-Diffs.

## Phase 6 – Plan→PR UX-Polish
- Stepper UI, Voraussetzungen-Panel (Token/CLI), Fehlermeldungen

Messung: Erfolgsquote PR-Erstellung, Fallback-Transparenz.

## Phase 7 – A11y & i18n
- Tastaturbedienung, Kontraste, ARIA-Labels
- Sprachschlüssel + de/en-Grundlage

Messung: Manuelle A11y-Checks, Lighthouse-Basics.

## Phase 8 – Dokumentation & QA
- README-Abschnitte, kurze Nutzer-HowTos
- E2E Smoke-Szenarien, Monitoring der Kernpfade

## Risiken & Gegenmaßnahmen
- RBAC-Inkonsistenzen → zentrale Helper erzwingen, Tests ergänzen
- Typwarnungen/SDK-Antworten → leichte Wrapper/Typing-Casts
- UI-Divergenz → Tokens/CSS zentralisieren, Komponenten-Patterns definieren

---

Dieser Plan dient als Leitfaden für den schrittweisen Ausbau. Wir können Phasen 3–6 parallelisiert angehen, achten aber auf kleine, reviewbare Inkremente.