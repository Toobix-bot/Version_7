# Änderungs-Spec

- Problem: <!-- Was war das Problem / Ziel? -->
- Ansatz: <!-- Wie wird es gelöst? Architektur / Design / Alternativen -->
- Risiko: <!-- low | med | high; warum? -->
- Messung: <!-- Metriken / Tests / Telemetrie zur Wirkung -->
- Rollback: <!-- Wie rückgängig machen / Feature-Flag / Plan B -->

## Checkliste

- [ ] PR-only: Keine direkten Writes auf `main` (Branch + PR)
- [ ] Tests grün (Lint/Type/Unit)
- [ ] Docs/README/Changelog aktualisiert
- [ ] Telemetrie (Events/Metrics) berücksichtigt
- [ ] Policy-Gates/Whitelist eingehalten
- [ ] Dry-Run/Preview vorhanden (wo sinnvoll)

## QA Hinweise

- Schritte zur manuellen Prüfung:  
  1.  
  2.  
  3.  

## Kontext

- Relevante Issues/Links:
- Feature-Flags/Configs: `./config/*.yaml`
