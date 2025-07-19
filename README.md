# TFT Meet-in-the-Middle Optimizer

Dieses Projekt implementiert einen Meet‑in‑the‑Middle Optimizer für Teamfight Tactics Set 14. Der Optimizer durchsucht alle möglichen Teams einer gegebenen Größe exhaustiv und nutzt eine Kompression beider Hälften um die Suche drastisch zu beschleunigen.

## Projektstruktur

```
.
├── src/
│   └── optimizer.py    # MiM Optimizer und Hilfsfunktionen
├── tests/
│   └── test_optimizer.py   # Pytest Suite
├── pyproject.toml      # Tool-Konfiguration für Formatierung, Linting und Tests
├── README.md           # Diese Anleitung
└── AGENTS.md           # Projektregeln für Codex
```

## Installation

Das Projekt benötigt Python ≥3.10. Abhängigkeiten werden über `pip` installiert.

```bash
pip install -r requirements.txt  # falls vorhanden
```

Für lokale Checks kann `pre-commit` genutzt werden:

```bash
pre-commit run --all-files
```

## Nutzung

Der Optimizer wird über das Skript `optimizer.py` gestartet:

```bash
python -m src.optimizer
```

Das Skript erzeugt während der Ausführung Fortschrittsmeldungen und gibt am Ende die besten gefundenen Teams aus.

### Anpassungen

* **Championdaten**: Der Block `CHAMPION_DATA` in `src/optimizer.py` muss mit den korrekten Champions, Kosten und Traits des aktuellen Sets gefüllt werden.
* **Scoring**: Die Funktion `compute_score` sowie die Parameter `LEVEL_WEIGHT`, `GOLD_UTIL_LINEAR` usw. können an individuelle Bewertungsmodelle angepasst werden.
* **Parameter**: Teamgröße (`TEAM_SIZE`) und weitere Konstanten lassen sich im Kopf der Datei konfigurieren.

## Tests

Tests werden mit `pytest` ausgeführt:

```bash
pytest
```

Die Konfiguration in `pyproject.toml` erzwingt eine Testabdeckung von mindestens 90 Prozent für das `src`-Verzeichnis.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
