# AGENTS.md

This file guides AI agents (e.g. OpenAI Codex) when working with this repository. The project currently consists of a single Python script containing the full optimizer logic.

---

## 1. Project Structure

```
/ (repository root)
├─ Rech.py   # meet-in-the-middle optimizer
├─ README.md # short description
└─ AGENTS.md # guidelines for contributors
```

* **`Rech.py`**: standalone script implementing the optimizer including configuration, data blocks and the `main` section.
* **`README.md`**: minimal top‑level description.
* **`AGENTS.md`**: the document you are currently reading.

---

## 2. Coding Conventions

* Target **Python 3.10+**.
* Run `black` with default settings before committing.
* Keep imports grouped: standard library, third‑party, local modules.
* Add type hints where practical.
* Prefer the `logging` module over `print` for future extensions.

---

## 3. Testing Requirements

At the moment there are no automated tests. If you add features, also create tests under a new `tests/` directory using `pytest`.

---

## 4. PR Guidelines

* Commit messages may follow `feat:`, `fix:`, `docs:` etc.
* PR descriptions should briefly mention the motivation, the solution and how it was tested.

---

## 5. Programmatic Checks

Before submitting a PR run:

```bash
black Rech.py --check
flake8 Rech.py
```

If you add tests, also run `pytest`.

---

*Note*: update the `CHAMPION_DATA` section in `Rech.py` and adjust `compute_score()` to match your desired scoring rules.
