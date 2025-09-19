# Wireframes (Low/Mid Fidelity)

Stand: 2025-09-19

Ziel: Skizzen für die Kernbereiche als Referenz für Umsetzung und Feinspezifikation.

Hinweis: Textuelle Wireframes mit Struktur, Zuständen und Hauptinteraktionen. Visuelle Umsetzung folgt im Designsystem.

## Dashboard

- Header: Titel, Login/Account, Rolle-Badge, Theme/Palette-Schnellzugriff
- Karten (Grid):
  - SSE-Status (verbunden/neu verbinden) + /metrics Health
  - Offene Vorschläge (Count, Liste top 5, Link zur vollen Liste)
  - Letzte Story-Ereignisse (chronologisch)
  - PR-Status (zuletzt erstellt, Links)
- CTA-Bereich: „Plan→PR“ (wenn Rolle ≥ Fortgeschritten), „Policies Dry-Run“ Quick Links

## Story

- Linke Spalte: Zustand + Aktionen
  - State: Titel, aktueller Abschnitt, Badges (Risk, Level, Inspiration), Fortschrittsbalken
  - Aktionen: Advance, Zufallsereignis erzwingen, Clear Log
  - Log: Scrollbare Liste mit Zeitstempeln, Event-Typ (Icons/Farbe)
- Rechte Spalte: Einstellungen
  - Stil: Ton (Dropdown), Temperatur (Slider), Stimme (Dropdown)
  - Zufallsereignisse: Konfig (Wahrscheinlichkeiten, Filter), Test-Hooks
  - Ansicht: Theme, Palette (Primär/Sekundär, Drehen), Dichte
- Zustände: Busy/Idle/Error, SSE-Status sichtbar
- Öffentlich-Modus: Aktionen ausgeblendet, nur Lesen

## Plan → PR

- Stepper (3 Schritte):
  1) Plan-Details: Titel, Beschreibung, optional „Draft“, Labels (Tags)
  2) Branch/PR-Parameter: Branch-Alias, Basis-Branch, Reviewer (optional)
  3) Ergebnis: Preview, Erstellen, Fallback-Hinweise, PR-Link
- Seitenleiste: Voraussetzungen (GitHub-Token/CLI), RBAC-Hinweise
- Fehlerzustände: Token fehlt → Fallback auf lokalen Branch, klare Hinweise

## Policies

- Liste der Templates (Suche/Filter), Detail-Preview
- Wizard: Parameter erfassen → Dry-Run ausführen → Diff/Validierung anzeigen → Apply (rollenabhängig)
- Statusbereich: Letzte Applies, Fehlermeldungen, Erfolge

## Suggestions

- Liste: Filter (offen/geschlossen), Sortierung (Impact, Datum), Suchfeld
- Eintrag: Titel, Status (Entwurf/geprüft/übernommen), Impact-Badge, Quick-Aktionen (Review)
- Detail: Beschreibung, Schritte, vorgeschlagene Patches, Diskussion/Kommentare
- Aktionen (rollenbasiert): Generieren, Review, Status ändern, Impact erfassen

## Einstellungen

- Account: Anmelden/Abmelden/Registrieren, Session-Info
- Rolle: Anzeige, Hinweise zu Berechtigungen
- Ansicht: Theme, Primär/Sekundär-Farben, Rotation, Dichte
- Sichtbarkeit: Öffentlich-Schalter (Header-Badge spiegelt Status)

## Admin/Monitoring (optional)

- SSE: Eventfluss (letzte Events), Verbindungsstatus, Reconnects
- /metrics: Live-Checks, offene Vorschläge (Gauge), Latenzen (falls vorhanden)

---

Diese Wireframes sind Grundlage für das Designsystem und die spätere Implementierung. Sie bilden Zustände und Interaktionen ab, ohne visuelle Feinheiten festzulegen.