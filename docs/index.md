# Life-Agent

Ein kleiner Server mit UI, der Pläne schmiedet, sein Verhalten beobachtet und sich über Regeln steuern lässt. In der Web-UI siehst du Meta/LLM-Infos, eine Plan-Demo, einen schlanken LLM-Chat sowie Live-Bereiche für Events & Metrics.

- Live-Docs: `/docs` (Header `X-API-Key` nötig)
- OpenAPI JSON: `/openapi.json`
- Statische OpenAPI: `openapi.yaml` (dieser Ordner)
- Dashboard: `/ui`

## Was es kann (einfach)
- Pläne bauen mit Safety: `POST /plan` prüft Risk-Regex & Pfad-Whitelist, erzeugt dann ein Plan-Artefakt (keine Direkt-Schreiberei).
- Regeln live laden: `POST /policy/reload` lädt YAML-Policy; Fehler kommen als 422 strukturiert zurück. Optional OPA.
- Live mitlesen: `GET /events` (SSE) sendet sofort `ready` + Heartbeats.
- Gesundheitswerte: `GET /metrics` (Prometheus) liefert Zähler/Zeiten.
- LLM (Groq): Minimal-Chat bei gesetztem `GROQ_API_KEY`.

## Schnellstart UI
1. API-Key in der Kopfzeile eintragen (z.B. `test`).
2. In „Plan Demo“ Intent/Targets setzen, Button klicken → Antwort erscheint.
3. Events öffnen, um Heartbeats/Events zu sehen.

## Policy-Beispiele
- Gültig: `policies/examples/valid.yaml`
- Ungültig: `policies/examples/invalid.yaml`

Reload ausführen (Beispiel):

```bash
curl -X POST "$BASE/policy/reload" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"path":"policies/examples/valid.yaml"}'
```

Bei ungültiger Policy kommt ein `422` mit Details.
