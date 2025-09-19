# Designsystem (Richtlinien & Tokens)

Stand: 2025-09-19

Ziel: Konsistente visuelle und Interaktionssprache. Tokenisieren, benennen, anwenden.

## 1. Design Tokens

- Farben (Semantik):
  - --color-bg, --color-fg, --color-panel, --color-border, --color-muted
  - --color-chip, --color-chip-fg, --color-accent, --color-accent-2
  - --color-ok, --color-warn, --color-danger (optional, abgeleitet)
- Hues:
  - --primary-h (Standard: 210), --secondary-h (Standard: 330)
- Typografie:
  - Basis: system-ui, Segoe UI, Arial, sans-serif
  - Größen: 12 / 13 / 14 / 16 / 18 / 24 (Skalierung in rem)
  - Zeilenhöhe 1.4–1.6, Gewicht: 400/600
- Spacing & Radius:
  - 2 / 4 / 6 / 8 / 12 / 16 px; Radius 4 / 6 / 8
- Fokus & Interaktion:
  - Fokusrahmen mit ausreichendem Kontrast (AA), Outline 2px, Offset 2px
  - Hover/Active-Stati, Disabled-Klarheit

Hinweis: Tokens in `api/app.py`-Inline-UI bereits grob umgesetzt; spätere Auslagerung in eigene CSS-Datei vorgesehen.

## 2. Komponenten (Basispalette)

- Button
  - Primär: Hintergrund hsl(--primary-h, 80%, ~52%), Weißer Text
  - Sekundär: --color-accent-2 + Text in --color-fg
  - Zustände: hover (aufhellen), active (abdunkeln), disabled (reduzierte Opazität)
- Badge
  - Standard: --color-chip / --color-chip-fg
  - Level/Warnung: --color-warn, Text dunkel
  - Inspiration: hsl(--secondary-h, 80%, ~60%)
- Card/Panel
  - Hintergrund --color-panel, Border --color-border, Radius 8px
- Progressbar
  - Verlauf: hsl(--primary-h, 80%, 45%) → 60%
- Input/Select
  - Hintergrund dunkel, Border --color-border, Radius 4–6px, klarer Fokus

## 3. Zustände & Feedback

- Ladezustand: Spinner/Balken + „Lädt…“
- Fehler: rote/kontrastreiche Meldung, klare Aktion (Retry)
- Leere Zustände: erklärender Text + CTA (optional)
- Erfolg: Bestätigungs-Toast oder Inline-Badge

## 4. A11y

- Kontraste prüfen (AA mind. 4.5:1 für Fließtext)
- Tastaturbedienung vollständig
- ARIA-Attribute bei komplexen Controls (Slider, Toggles)

## 5. Internationalisierung

- Texte zentralisieren, Schlüssel definieren; de/en als Start

## 6. Umsetzung & Migration

- Kurzfristig: Inline-Styles schrittweise in zentrale CSS-Datei überführen
- Komponenten-Patterns vereinheitlichen (Buttons, Karten, Badges, Inputs)
- Tokens als Quelle der Wahrheit (Single Source)

---

Dieses Designsystem ist bewusst pragmatisch gehalten und deckt die aktuellen UI-Bereiche ab. Es kann iterativ erweitert werden (z. B. Tabellen, Tabs, Modals, Toaster).