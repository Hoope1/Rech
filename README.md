# Rech

This repository contains a standalone Python 3.10+ script implementing a meet-in-the-middle optimizer for Teamfight Tactics (TFT) Set 14. The tool enumerates all possible teams of a configured size and scores them using customizable rules.

## Features

- **Exact search**: Splits the champion pool into two halves and compresses each side to drastically reduce combinations compared to a brute-force search.
- **Trait-based scoring**: Flexible weighting for trait breakpoints with optional bonuses for high‑cost units or gold utilisation.
- **Progress reporting**: Logs progress and improvements while running.
- **Easily customizable**: Adjust the champion list, trait breakpoints and the scoring function directly in `Rech.py`.

## Requirements

- Python **3.10** or newer.
- Only standard library modules are used. `black` and `flake8` are recommended for formatting and linting.

Install the optional tools with:

```bash
pip install black flake8
```

## Usage

1. Edit the `CHAMPION_DATA` block in `Rech.py` so it lists all Set 14 champions with correct traits and costs.
2. Adjust `TRAIT_BREAKPOINTS` and `compute_score()` to match your desired scoring rules.
3. Run the optimizer:

```bash
python Rech.py
```

The script prints progress information and finally displays the top teams according to your scoring function.

## Development

Before committing changes, run the formatting and lint checks:

```bash
black Rech.py --check
flake8 Rech.py
```

If you extend the project with additional features, please also add tests under `tests/` using `pytest`.

## License

This project is provided as-is without warranty. Use and modify it at your own risk.
