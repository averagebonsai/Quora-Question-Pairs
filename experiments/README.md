# experiments/

Plug-and-play ML experiment harness for the Quora Question-Pairs project.

## Structure

```
experiments/
├── data.py              Loads zarr + CSV → list[PairRecord]  (shared, no model logic)
├── features.py          Primitive feature functions (embedding, lexical, matryoshka)
├── report.py            Metrics printer + results writer (metrics.txt, config.json, …)
├── run_experiment.py    ← ENTRY POINT — run this
│
├── models/
│   ├── catboost_model.py   CatBoost  — matryoshka + lexical features
│   ├── xgboost_model.py    XGBoost   — matryoshka + lexical features
│   ├── logreg_model.py     Logistic Regression — embedding + lexical features (scaled)
│   └── cosine_baseline.py  Cosine similarity threshold baseline
│
├── splits/
│   └── default_split.npz   (auto-created on first run, reused forever)
│
└── results/
    ├── all_experiments.csv          one row per completed run
    └── <experiment_name>/
        ├── metrics.txt
        ├── errors.csv
        ├── config.json              ← NEW: full reproducibility record (see below)
        └── feature_importance.txt
```

## Running an experiment

You might probably want to use the submit.sh script here, but if you don't want to use SLURM:

```bash
cd experiments
uv run python run_experiment.py --model catboost --name catboost_matryoshka_all_features

# Optional: auto-push tracked results to DVC remote after a successful run
uv run python run_experiment.py --model catboost --name catboost_matryoshka_all_features --dvc-push
```

Available `--model` values: `xgboost`, `catboost`, `logreg`, `cosine`

The first run saves `splits/default_split.npz`.  
Every subsequent run loads those exact indices so all models are
compared on the **same test rows**.

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` / `-m` | *(required)* | Model to use |
| `--name` / `-n` | *(required)* | Unique experiment name (output folder) |
| `--max-rows` | None | Subsample N rows (smoke-tests) |
| `--test-size` | 0.20 | Held-out fraction |
| `--threshold` | model default or 0.5 | Decision threshold |
| `--zarr` | `../embeddings.zarr` | Path to embeddings store |
| `--split-file` | `splits/default_split.npz` | Path to saved split indices |
| `--results-dir` | `results/` | Output root directory |
| `--dvc-push` | off | After report generation, run `uv run dvc push experiments/results` |
| `--dvc-push-target` | `experiments/results` | DVC target path to push when `--dvc-push` is enabled |

> `--dvc-push` is intentionally opt-in. This avoids failing local experiments for users
> who only have read-only DVC access or no write credentials to the remote.

## Feature sets

### Matryoshka embedding features (XGBoost & CatBoost)

Both `XGBoostModel` and `CatBoostModel` use **`matryoshka_all_features`**:
for each prefix dimension `d` in `matryoshka_dims`, the following statistics
are computed over the `d`-dimensional slice of both embeddings:

```
d{d}_cos_sim, d{d}_dot_raw, d{d}_euclidean, d{d}_manhattan,
d{d}_abs_diff_mean, d{d}_abs_diff_max, d{d}_abs_diff_std,
d{d}_prod_mean, d{d}_prod_std
```

Plus the shared **lexical features** (Jaccard, token counts, char lengths, etc.).

Default prefix dims (defined in `features.py`):
```python
DEFAULT_MATRYOSHKA_DIMS = (128, 256, 512, 1024, 1536, 2048, 2560)
```
With the full 2560-d vector always included, this gives **8 slices × 9 stats + 10 lexical = 82 features**.

Pass custom dims when instantiating a model:
```python
CatBoostModel(matryoshka_dims=(128, 256, 512, 1024, 2560))
XGBoostModel(matryoshka_dims=(128, 256, 512, 1024, 2560))
```

## Config tracking (`config.json`)

Every experiment run now writes a `config.json` into its results folder.
It records everything needed to reproduce or compare the run:

```json
{
  "experiment_name": "catboost_matryoshka_all_features",
  "run_at": "2026-03-29 19:30:00",
  "model": "CatBoost",
  "threshold": 0.5,
  "test_size": 80892,
  "cli_args": {
    "model": "catboost",
    "name": "catboost_matryoshka_all_features",
    "max_rows": null,
    "test_size": 0.2,
    "threshold": null,
    "zarr": "../embeddings.zarr",
    "split_file": "splits/default_split.npz",
    "results_dir": "results/"
  },
  "model_config": {
    "model_class": "CatBoostModel",
    "matryoshka_dims": [128, 256, 512, 1024, 1536, 2048, 2560],
    "hyperparams": { "iterations": 500, "depth": 8, "learning_rate": 0.05, "..." : "..." },
    "n_features": 82,
    "feature_names": ["d128_cos_sim", "d128_dot_raw", "...", "jaccard"]
  }
}
```

Models expose `get_config() → dict` to provide the `model_config` block.
Both `CatBoostModel` and `XGBoostModel` implement this.  For models that
don't, `report.py` falls back to recording just `n_features` and
`feature_names`.

## Adding a new model

1. Copy any existing file in `models/` as a starting point.
2. Implement:
   - `build_features(records) → (X, y, feature_names)`
   - `fit(X_train, y_train)`
   - `predict_proba(X_test) → 1-D float32 array`
3. Optionally add:
   - `feature_importances() → dict[str, float]` for automatic importance reporting
   - `get_config() → dict` for full config tracking in `config.json`
4. Register it in `experiments/models/__init__.py` and `MODEL_REGISTRY` in
   `run_experiment.py`.

## Adding a new feature set

Add a function to `features.py` that takes a `PairRecord` and returns
`dict[str, float]`.  Reference it from your model's `_feature_fn`.

Available primitives:

| Function | Description |
|----------|-------------|
| `embedding_features(r)` | 12 stats from the full raw + normalised embeddings |
| `lexical_features(r)` | 10 token/char overlap features |
| `all_features(r)` | `embedding_features` + `lexical_features` (22 total) |
| `matryoshka_embedding_features(r, dims)` | Per-prefix-slice embedding stats |
| `matryoshka_all_features(r, dims)` | Matryoshka embedding stats + lexical features |

## Results comparison

After multiple runs, open `results/all_experiments.csv` to compare all
experiments side-by-side (accuracy, precision, recall, F1, TP/FP/TN/FN).
Per-run `config.json` files let you trace exactly which features and
hyperparameters produced each result.
