# Rech

This repository contains `Rech.py`, a Python 3.10+ script implementing a meet-in-the-middle optimizer for Teamfight Tactics (TFT) Set 14.

## Features

- **Dynamic data**: champion and trait information is fetched from Riot's Data Dragon and cached locally.
- **Exact search**: the champion pool is split into two halves and enumerated with basic dominance pruning.
- **Trait-based scoring**: configurable weights for trait breakpoints plus bonuses for high-cost units and gold utilisation.
- **Progress reporting**: uses `tqdm` if available, otherwise a simple fallback progress bar.
- **Result re-scoring**: existing JSON or CSV results named `tft_full_bruteforce_results.*` are rescored on start.
- **Parameter sets**: two predefined parameter sets (`P1` and `P2`) selectable via `--param-set`.

## Requirements

- Python **3.10** or newer.
- The script only requires the standard library and `requests`. Optional: `tqdm` and `numba` for faster execution.
- `black` and `flake8` are recommended for development.

Install the optional tools with:

```bash
pip install requests tqdm numba black flake8
```

## Usage

1. Run the optimizer:
   ```bash
   python Rech.py --param-set P1 --top-k 20 --team-size 8 --seed 123 --verbose 1
   ```
   The script downloads the latest Set 14 data and stores a cache in `tft_set14_cache.json.gz`.
2. Modify `--param-set` or edit `compute_team_components()` if you want to change scoring behaviour.
3. Existing result files are detected automatically and rescored.
4. Use `--force-refresh` to ignore cached data and fetch new snapshots.

The best teams are printed to the console and written to `demo_mim_top_combined.json`.
Metadata about each run is saved to `run_meta.json` (commit hash and timestamp).

## Development

Before committing changes, run the formatting and lint checks:

```bash
black Rech.py --check
flake8 Rech.py
```

If you extend the project with additional features, add tests under `tests/` using `pytest`.

## License

This project is provided as-is without warranty.
