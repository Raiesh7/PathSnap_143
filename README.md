# PathSnap

Distributed map-matching and route reconstruction for the ML Hackathon.

## Project layout

- `data/` - dataset location and data documentation
- `src/` - loading, preprocessing, features, candidates, models, metrics, and decoding
- `experiments/` - training and hyperparameter-search scripts
- `results/` - generated metrics, tables, and plots
- `artifacts/` - trained models and preprocessing assets
- `maps/` - generated Folium maps
- `api/` - FastAPI backend
- `ui/` - evaluator-facing frontend
- `notebooks/` - exploratory analysis notebooks
- `report/` - technical report source and final PDF

The supplied development dataset is versioned in `data/Student folder/`:
80 training trips and 10 student-testing trips. See [data/README.md](data/README.md).

## Dataset audit

Install `requirements.txt`, then run:

```powershell
python experiments/01_dataset_exploration.py
```

The script inspects only `train` and `student_test`, preserves the raw data, and
writes inventory/quality tables and figures to `results/audit/`, interactive
Folium maps to `maps/audit/`, and report source to
[`report/dataset_audit.md`](report/dataset_audit.md). Open map HTML files in a
browser; scripts and OpenStreetMap tiles require internet access. Static PNGs
can be viewed offline. This stage does not train models or create pointwise labels.

## Preprocessing

```powershell
python experiments/01b_preprocess_dataset.py --cache-dir data/processed
python -m unittest discover -s tests -v
```

Preprocessing preserves raw observations and IDs, projects coordinates to metres,
builds directed road graphs and marks gaps and uncertain motion. It does not read
reference routes. See [report/preprocessing.md](report/preprocessing.md) for the
policy and measured results. Summary tables are saved in `results/preprocessing/`;
optional reproducible caches in `data/processed/` are excluded from Git.

The reusable entry points are `src.data_loader.load_trip`,
`src.preprocessing.preprocess_trip` and `src.preprocessing.load_preprocessed_trip`.
For example, after installing the requirements:

```python
from pathlib import Path
from src.data_loader import load_trip
from src.preprocessing import preprocess_trip

trip = preprocess_trip(load_trip(Path("data/Student folder/train/00000001"), "train"))
print(trip.observations.head())
```

## LightGBM Bayesian hyperparameter tuning

After the candidate tables and shared five-fold assignment have been generated:

```powershell
python experiments/03_bayesian_optimization.py --trials 20 --n-jobs 4
```

The Optuna TPE search selects parameters using training-trip cross-validation
Macro F1 only, then compares the tuned model with the untuned configuration on
student testing. It preserves the five-model baseline outputs. Results and the
search history are saved in `results/tuning/lightgbm/`; the tuned model is saved
as `artifacts/models/lightgbm_tuned.joblib`. Use `--resume` to continue an existing
study with matching inputs and configuration. See
[report/hyperparameter_tuning.md](report/hyperparameter_tuning.md) for methodology.

