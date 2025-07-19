# AGENTS.md

Diese Datei leitet AI-Agenten (z. B. OpenAI Codex) beim Arbeiten mit diesem Python-Projekt an und beschreibt Projektstruktur, Codierungs­konventionen, Testanforderungen, PR-Richtlinien und programmatische Prüfungen.

---

## 1. Project Structure

```
/ (Projekt-Root)
├─ src/
│   └─ optimizer.py      # Meet-in-the-Middle Optimizer für TFT Set 14
├─ tests/
│   └─ test_optimizer.py # Unit- und Integrationstests
├─ pyproject.toml        # Projekt-Konfiguration (Build, Format, Linter, Test)
├─ README.md             # Kurzbeschreibung und Benutzer­anleitung
└─ AGENTS.md             # Diese Datei
```

* **`src/optimizer.py`**: Hauptskript mit der MiM-Logik, Konfigurations­parametern, Datenblöcken und der `main()`-Routine.
* **`tests/`**: Testfälle für einzelne Funktionen (`compute_score`, `enumerate_half`, `combine_and_search`, etc.).
* **`pyproject.toml`**: Definiert Abhängigkeiten, Tool-Konfigurationen für pytest, black, flake8, mypy.
* **`README.md`**: Überblick, Beispiel­aufrufe und Hinweise zur Anpassung (z. B. `CHAMPION_DATA`, `LEVEL_WEIGHT`).

---

## 2. Coding Conventions

* **Sprache**: Python 3.10+
* **Formatierung**: Verwende `black` (88 Zeichen Breite) und formatiere vor jedem Commit.
* **Typhinweise**: Vollständige Type-Hinweise überall; `mypy` sollte ohne Fehler laufen.
* **Linting**: `flake8` mit Plugins für Typhinweise (`flake8‑annotations`) und Import­optimierung.
* **Docstrings**: PEP 257-konforme Docstrings für alle Module, Klassen und public-Funktionen.
* **Imports**: Sortiert mit `isort` (stdlib, Drittanbieter, lokale Module).
* **Logging**: Nutze das `logging`-Modul statt `print` für alle Ausgaben; konfiguriere Logger im `main()`.

Beispiel in `src/optimizer.py`:

```python
import logging
from typing import List, Tuple
# ...

def compute_score(...):
    """
    Berechnet den Score anhand trait_counts, cost und high_cost_units.
    """
    ...
```

---

## 3. Testing Requirements

* **Test-Framework**: `pytest` (Version ≥7.0)
* **Test-Ordner**: Alle Tests in `tests/` mit `test_`-Präfix.
* **Coverage**: Mindestens 90 % Abdeckung für Kernelemente (`compute_score`, `enumerate_half`, `combine_and_search`, `bitmask_to_team`, `summarize_traits`).
* **Fixtures**: Verwende `pytest.fixture` zum Aufsetzen wiederverwendbarer Szenarien.
* **Parametrisierte Tests**: Für unterschiedliche Konfigurations­kombinationen (z. B. `TEAM_SIZE`, `HIGH_COST_THRESHOLD`).

Beispiel in `tests/test_optimizer.py`:

```python
import pytest
from src.optimizer import compute_score

@ pytest.mark.parametrize(
    "trait_counts,total_cost,high_cost_units,expected",
    [([0,2,0], 10, 1, 2.5), ...]
)
def test_compute_score_linear_use(...):
    assert compute_score(trait_counts, total_cost, high_cost_units) == pytest.approx(expected)
```

---

## 4. PR Guidelines

* **Branch-Namen**: `feature/<kurz-beschreibung>`, `fix/<kurz-beschreibung>`.
* **Commit-Messages**:

  * Format: `<Typ>(Scope): Kurze Beschreibung`
  * Typen: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
* **PR-Beschreibung**:

  1. **Motivation** – Warum diese Änderung?
  2. **Lösung** – Kurze Zusammenfassung der Implementierung.
  3. **Test** – Wie wurde getestet?
* **Review-Checks**:

  * Alle Tests müssen grün sein.
  * `black`, `flake8` und `mypy` müssen fehlerfrei durchlaufen.
  * Coverage darf nicht sinken.

**Beispiel-PR**:

```
feat(optimizer): add progress ETA logging

- Implemented logging of ETA in enumerate_half()
- Added tests for progress formatting
```

---

## 5. Programmatic Checks

In CI (z. B. GitHub Actions) automatisch ausführen:

1. **Format**: `black --check src tests`
2. **Imports**: `isort --check src tests`
3. **Linting**: `flake8 src tests`
4. **Types**: `mypy src`
5. **Tests & Coverage**:

   ```bash
   pytest --maxfail=1 --disable-warnings --cov=src --cov-fail-under=90
   ```

**Optional**: Integriere `pre-commit` mit den Hooks für Black, isort, flake8 und mypy. Beispiel `pyproject.toml`:

```toml
[tool.pre-commit]
repos = [
  { repo = "https://github.com/psf/black", rev = "22.3.0", hooks = [{ id = "black" }] },
  { repo = "https://github.com/PyCQA/isort", rev = "5.10.1", hooks = [{ id = "isort" }] },
  { repo = "https://github.com/pycqa/flake8", rev = "4.0.1", hooks = [{ id = "flake8" }] },
  { repo = "https://github.com/pre-commit/mirrors-mypy", rev = "v0.982", hooks = [{ id = "mypy" }] },
]
```

---

*Hinweis*: Passe `CHAMPION_DATA`, `LEVEL_WEIGHT` und `compute_score()` genau an deine Spiel­daten und bisherigen Score-Formeln an, um exakte Reproduzierbarkeit sicherzustellen.
