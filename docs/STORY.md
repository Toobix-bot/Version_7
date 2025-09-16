# Story-Modul

Ziel: Narrative Spielschleife mit Zuständen, Ereignissen und Entscheidungen.

Kernbegriffe
- State: epoch, mood, arc, resources
- Events: story_events (mit text, tags, mood, deltas)
- Options: story_options (id, label, rationale, risk, expected)
- Skills/Companions/Buffs: sekundäre Progressionssysteme

API (Auszug)
- GET /story/state – aktuellen Zustand & letzte Events laden
- POST /story/choose – Option wählen (id)
- POST /story/reset – Story zurücksetzen
- GET /story/export – JSON Export
- GET /story/meta/skills|companions|buffs – Metadaten

Beispielablauf
1) /story/state laden
2) Option wählen via /story/choose
3) Ressourcen/XP wachsen; neue Optionen erscheinen

Hinweise
- Entscheidungen sind zentral. Keine Auto‑Ticks. Idle‑Mechaniken gehören nicht hierher.
- Events können im UI geloggt werden (Timeline).
