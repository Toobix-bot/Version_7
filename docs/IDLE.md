# Idle-Modul

Ziel: Fortschritt im Hintergrund mit minimaler Interaktion.

Kernbegriffe
- Quests: Ziel (required), Fortschritt (progress), Status
- Auto‑Ticks: regelmäßige, kleine Fortschritte
- Belohnungen: Ressourcen, XP, einfache Effekte

API (Auszug)
- GET /game/idle/state – aktueller Fortschritt
- POST /game/idle/tick – einen Tick anstoßen
- GET /game/idle/quests – Liste aktiver Quests

Hinweise
- Kein Entscheidungen‑Baum, keine Story‑Events.
- Kann parallel zur Story laufen, aber Logik und Daten getrennt halten.
