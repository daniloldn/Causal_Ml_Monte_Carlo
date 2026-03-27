# Causal ML Monte Carlo

Monte Carlo simulation project for comparing **OLS** and **DML (Double Machine Learning)** treatment-effect estimation under controlled misspecification and overlap regimes.

The core question is:
When does flexible partialling-out (DML with random forests) outperform linear adjustment (OLS)?

## What This Repo Does

For each simulation scenario, the code:

1. Generates synthetic data with:
- Adjustable outcome nonlinearity (`alpha_y`)
- Adjustable treatment assignment complexity (`alpha_d`)
- Adjustable overlap strength (`kappa`)

2. Estimates ATE using:
- `OLS` (`src/estimator.py::estimate_ols`)
- `DML` (`src/estimator.py::estimate_dml_manual`; the EconML version is also included)

3. Repeats this across many replications and scenario grids.

4. Aggregates metrics (bias, SD, RMSE, coverage, CI length, overlap, residual treatment variance).

5. Produces frontier/diagnostic visualizations (mostly Plotly heatmaps and line charts).

## Repository Layout

- `src/dgp.py`: data generating process (covariates, treatment, outcome, overlap controls)
- `src/estimator.py`: OLS and DML estimators
- `src/monte_carlo.py`: replication/scenario/grid runners
- `src/metrics.py`: summarization and wide metric tables
- `src/figures.py`: plotting helpers for frontier and mechanism analysis
- `src/utils.py`: config loading
- `configs/baseline.yaml`: baseline simulation settings
- `notebooks/simulations.ipynb`: main analysis notebook (run pipeline + figures)
- `tests/*.ipynb`: module-level sanity-check notebooks

## Environment Setup

Use either Conda (`environment.yml`) or pip (`requirements.txt`).

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate causal-ml-monte-carlo
```

### Option B: venv + pip

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run

### Option A: Main notebook workflow (recommended)

From repo root:

```bash
jupyter notebook notebooks/simulations.ipynb
```

In the notebook:

1. Load config and optionally run simulation:
```python
from src.utils import load_config
from src.monte_carlo import run_simulation_grid

config = load_config("baseline")
# df = run_simulation_grid(config, save_each=True, n_jobs=8)
```

2. Or load previously saved parquet outputs from `results/raw/`.

3. Build summaries and plots:
```python
from src.metrics import summarize_df
from src.figures import frontier_heatmap

summary_df = summarize_df(df)
fig = frontier_heatmap(summary_df, kappa=0.5)
fig.show()
```

### Option B: Run full simulation from script

From repo root:

```bash
python src/monte_carlo.py
```

This uses `configs/baseline.yaml` and writes scenario parquet files to `results/raw/`.

## Configuration

Primary controls are in `configs/baseline.yaml`:

- `sample_size`, `num_replications`
- `alpha_y_grid`: outcome nonlinearity levels
- `alpha_d_grid`: treatment assignment complexity levels
- `kappa`: overlap regime values
- `treatment_effect`, `noise_std`, functional-form terms for DGP
- output folders (`output_dir`, `processed_dir`, `figure_dir`)

If you want a quick run, reduce `num_replications` and/or shrink grids.

## Outputs

- Raw scenario-level replication outputs: `results/raw/*.parquet`
- Figures (if saved from notebook): typically under `results/figs/`

## Notes

- `estimate_dml_manual` is the default DML estimator used in simulation runs.
- `estimate_dml` (EconML `LinearDML`) remains in the code for comparison/extension.
