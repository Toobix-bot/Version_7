# Informationsarchitektur (IA-Map)

Stand: 2025-09-19

Ziel: Klare Struktur der Anwendung (Seiten, Routen, Zustände) und Rollenkonzepte (RBAC), als Grundlage für Wireframes, Designs und Implementierungsplan.

## 1. Top-Level Navigation

- Dashboard
- Story
- Plan → PR
- Policies
- Suggestions
- Einstellungen
- Anmelden/Registrieren

Optional/Administrativ:
- Admin/Monitoring (SSE-/Metrics-Status, Health)

## 2. Rollen & Zugriffsmodell (RBAC)

Rollen: Anfänger | Fortgeschritten | Profi

- Anfänger: Basisfunktionen, Lesen/Interaktion, keine PR-Erstellung, eingeschränkte Policy-/Vorschlagsaktionen
- Fortgeschritten: Erweiterte Funktionen (Plan→PR freigeschaltet), Policies anwenden (ggf. im Dry-Run), Vorschläge prüfen/bewerten
- Profi: Vollzugriff auf Plan→PR (inkl. Draft/Labels), Policy-Apply, erweiterte Admin/Monitoring-Ansichten

Spezialfälle:
- API-Key/Service: darf PR/automatisierte Endpunkte nutzen gemäß Serverkonfiguration
- Öffentlich (Einstellung): Lesender Zugriff ohne Login auf ausgewählte Bereiche (Story-Ansicht, Metrik-Badges)

## 3. Seitenstruktur & Zustände

### 3.1 Dashboard
- Inhalte: Kurzübersicht Status (SSE verbunden, /metrics erreichbar), offene Vorschläge, letzte Story-Ereignisse, PR-Status
- Zustände:
  - Unangemeldet: Zeigt nur public-Infos, CTA zum Anmelden
  - Anfänger: Übersicht + Lesezugriff auf Vorschläge
  - Fortgeschritten/Profi: Zusatzpanels (Plan→PR Quick Actions, Policy-Dry-Run Shortcuts)

### 3.2 Story
- Panels: Zustand/Log, Stil (Ton, Temperatur, Stimme), Zufallsereignisse-Konfiguration, Aktionen (Advance, Events), Badges/Metriken
- Zustände:
  - Generierung: „busy“ mit Fortschrittsbalken, abbrechbar
  - Idle: Aktionen verfügbar
  - Fehler: Fehlerhinweis, Retry möglich
  - SSE: verbunden/nicht verbunden (Badge)
- Sichtbarkeit:
  - Öffentlich: Leseansicht ohne Aktionen (wenn Einstellung „Öffentlich“ aktiv)

### 3.3 Plan → PR
- Schritte: Plan erfassen/auswählen → Branch/PR-Parameter → Erstellen (Draft/Labels) → Ergebnis/Links
- Zustände:
  - Voraussetzungen: Git/GitHub verfügbar? Token vorhanden?
  - Erfolgreich: PR-URL, Branchname, Labels
  - Fehler: Fallback (lokaler Branch), Hinweise
- Rollen: Ab Fortgeschritten sichtbar, volle Ausführung für Fortgeschritten/Profi

### 3.4 Policies
- Funktionen: Templates anzeigen, Dry-Run, Apply (mit Validierung/Deep-Merge), Wizard
- Zustände:
  - Dry-Run Ergebnis (Diff/Validierung)
  - Apply Bestätigung (rollenabhängig)
- Rollen: Lesen für Anfänger, Dry-Run/Apply für Fortgeschritten/Profi (gestaffelt)

### 3.5 Suggestions (Vorschläge)
- Bereiche: Liste (offen/geschlossen), Details/Review, Impact, Auto-Generate
- Zustände:
  - Ladezustand, Leere Liste, Gefiltert
  - Review-Status (Entwurf, geprüft, abgelehnt, übernommen)
- Rollen: 
  - Anfänger: Liste lesen, Feedback abgeben
  - Fortgeschritten: Generieren, Review, Status ändern
  - Profi: Vollzugriff inkl. Impact-Aktionen

### 3.6 Einstellungen
- Bereiche: Account (Login/Logout/Registrierung), Rolle (Anzeige), Ansicht (Theme, Palette, Dichte), Sichtbarkeit (Öffentlich)
- Zustände: Angemeldet/Abgemeldet, Server-/Session-Fehler

### 3.7 Admin/Monitoring (optional)
- Inhalte: Event-Stream-Badges, /metrics-Health, Anzahl offener Vorschläge (Gauge), Logs
- Rollen: Fortgeschritten/Profi

## 4. Routen (UI & API)

UI-Routen (Beispiele, Single-Page/Pane-konzept):
- / (Dashboard)
- /story
- /plan-pr
- /policies
- /suggestions
- /settings

API-Routen (Auszug, gemäß Implementierung in `api/app.py`):
- Events & Metrics
  - GET /events (SSE, ready/heartbeats)
  - GET /metrics (Prometheus Text)
- Auth & User
  - POST /auth/register, /auth/login, POST /auth/logout
  - GET /me, GET/POST /me/settings
- Story
  - GET /story/state, GET /story/log, POST /story/advance
  - GET/POST /story/style
  - GET /story/voices
  - GET/POST /story/events/config, POST /story/events/emit
- Plan → PR
  - POST /pr/plan, POST /pr/create (RBAC: Fortgeschritten+)
- Policies
  - POST /policy/dry-run, POST /policy/apply, GET /policy/templates
- Suggestions
  - POST /suggest/auto, POST /suggest/generate, GET /suggest/list
  - POST /suggest/review, POST /suggest/impact, POST /suggest/llm

RBAC-Hinweis: PR- und Apply-Routen durch Rollen/API-Key geschützt; UI blendet nicht erlaubte Aktionen aus.

## 5. Navigationsflüsse

- Erstbesuch (öffentlich): Dashboard → Story (read-only) → Anmelden-CTA → nach Login zurück zur letzten Ansicht
- Authentifizierter Flow (Anfänger): Story arbeiten → Vorschläge ansehen → Einstellungen (Theme/Palette) → Optional Upgrade
- Fortgeschritten/Profi: Policies Dry-Run → Plan→PR → PR-Link; Suggestions-Review → Impact erfassen

## 6. Zustandsmodell (querliegend)

- Session: none | gültig | abgelaufen → UI passt Sichtbarkeit an
- SSE: connected | reconnecting | disconnected → Badge/Retry
- Netzwerk: idle | loading | error (pro Panel) → einheitliche Statusanzeige

## 7. Berechtigungs-Matrix (Kurzform)

| Seite/Funktion       | Anfänger | Fortgeschritten | Profi |
|----------------------|---------:|----------------:|------:|
| Dashboard (lesen)    |    ✔️    |         ✔️      |  ✔️   |
| Story Aktionen       |    ✔️    |         ✔️      |  ✔️   |
| Plan→PR              |          |         ✔️      |  ✔️   |
| Policies Dry-Run     |          |         ✔️      |  ✔️   |
| Policies Apply       |          |         (⚠️ ggf.)|  ✔️   |
| Suggestions Gen/Review|   (lesen)|        ✔️      |  ✔️   |
| Admin/Monitoring     |          |         ✔️      |  ✔️   |

Hinweis: Feinsteuerung serverseitig via RBAC-Helpern; UI blendet gesperrte Features aus.

## 8. A11y & Internationalisierung (Grundlagen)

- Tastaturfokus klar sichtbar, Kontraste ausreichend (WCAG AA)
- ARIA-Labels für interaktive Elemente
- Sprachumschaltung (de/en) perspektivisch vorbereiten

## 9. Offene Punkte für Wireframes

- Detail-Layouts pro Panel (Tabs vs. Karten)
- Fehlermeldungs-Patterns und leere Zustände
- Platzierung von Metrik-Badges und Health-Anzeigen
- Suggestions-Detail mit Review/Impact-Workflow

---

Diese IA dient als Referenz für die nächsten Schritte (Wireframes, Designsystem, Implementierungsplan).