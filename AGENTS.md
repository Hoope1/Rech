# AGENTS.md

This file provides guidance for AI agents (for example OpenAI Codex) when working on this repository. The project consists of a single Python script that performs a meet-in-the-middle search on TFT Set 14 data fetched from Riot's Data Dragon.

---

## 1. Project Structure

```
/ (repository root)
├─ Rech.py    # optimizer logic and entry point
├─ README.md  # usage and development notes
├─ AGENTS.md  # this guide for contributors
└─ .flake8    # flake8 configuration
```

`Rech.py` downloads champion and trait data, defines two parameter sets (`P1`, `P2`) and contains helper functions for rescoring and progress reporting. Parameters can now be overridden via CLI flags.

---

## 2. Coding Conventions

- Target **Python 3.10+**.
- Format code using `black`.
- Lint with `flake8` (max line length 100).
- Group imports: standard library, third‑party, local.
- Use type hints where practical.
- Prefer the `logging` module over `print` for any new output.

---

## 3. Testing Requirements

No automated tests exist yet. If you add new functionality, create tests under `tests/` using `pytest`.

---

## 4. Pull Request Guidelines

- Commit messages may start with `feat:`, `fix:`, `docs:` and similar prefixes.
- PR descriptions should explain the motivation, the changes and how they were tested.

---

## 5. Programmatic Checks

Run the following before submitting a PR:

```bash
black Rech.py --check
flake8 Rech.py
```

If a `tests/` directory is present also execute `pytest`.

---

## 6. Customisation Tips

- Switch scoring presets using the `--param-set` CLI flag (or change `ACTIVE_PARAM_SET`).
- Adjust scoring behaviour via the `ScoreParams` dataclass or by modifying `compute_team_components()`.
- Champion and trait data is downloaded from Data Dragon and cached in `dd_version_cache.json` and `tft_set14_cache.json.gz`.
- The script automatically rescans `tft_full_bruteforce_results.json` or `.csv` files and prints score decompositions.
- CLI flags like `--team-size`, `--top-k` and `--seed` allow quick parameter experiments.

---
