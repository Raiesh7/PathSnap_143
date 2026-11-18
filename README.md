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

