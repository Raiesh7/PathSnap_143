# PathSnap API (Laptop 1)

FastAPI backend that runs the full prediction pipeline for one trip and scores it on the server.

```
trip_id -> load track/nodes/arcs -> preprocess -> candidates (50 m, 8) -> 21 features
        -> tuned LightGBM -> HMM/Viterbi (beta from results/hmm/config.json) -> connected route
        -> server-side scoring against the reference route -> JSON (metrics + route geometry)
```

## 1. Install (once)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use the library versions in `requirements.txt` so the saved model loads exactly as trained.

## 2. Put the evaluation trips on this laptop only

```
data/Student folder/evaluator_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
```

This folder is in `.gitignore`. It is never used for training, tuning or model choice.

## 3. Start the server

From the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` lets Laptop 2 reach it over the network.
- Find this laptop's IP address: `ipconfig` (Windows) or `hostname -I` (Linux).
- Laptop 2 then uses `http://<laptop-1-ip>:8000`.
- Allow port 8000 through the firewall if Laptop 2 cannot connect.

Optional settings (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `PATHSNAP_TRIPS_DIR` | `data/Student folder/evaluator_test` | Folder of trips offered to the UI |
| `PATHSNAP_MODEL` | `artifacts/models/lightgbm_tuned.joblib` | Trained model |
| `PATHSNAP_HMM_CONFIG` | `results/hmm/config.json` | HMM settings chosen on training trips |
| `PATHSNAP_CORS` | `*` | Allowed UI origins, comma-separated |

To rehearse without touching the evaluator trips:

```bash
PATHSNAP_TRIPS_DIR="data/Student folder/student_test" uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell: `$env:PATHSNAP_TRIPS_DIR="data/Student folder/student_test"` before the `uvicorn` command.

## 4. Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | Model, features, candidate and HMM settings, number of trips |
| GET | `/trips` | `{"count": 10, "trips": [{"trip_id": "..."}]}` |
| POST | `/predict` | Body `{"trip_id": "..."}` -> result below |
| GET | `/docs` | Interactive API page (FastAPI) |

`/predict` response:

```json
{
  "trip_id": "00000056",
  "counts":   {"gps_points": 3958, "candidate_rows": 26619, "hmm_breaks": 1, "route_segments": 2, "...": "..."},
  "metrics":  {"top1_accuracy": 0.96, "classifier_only_top1_accuracy": 0.91, "route_precision": 0.94,
               "route_recall": 1.0, "route_f1": 0.97, "route_continuity": 0.97, "scored_points": 3931, "...": "..."},
  "timings_s": {"preprocess": 0.05, "candidates": 0.16, "features": 0.05, "model": 0.2, "hmm": 3.5, "scoring": 0.25, "...": "..."},
  "model": {"file": "lightgbm_tuned.joblib", "hmm_beta_m": 3.0},
  "gps_points": [[lat, lon, t], "..."],
  "predicted_route": [[[lat, lon], "..."], "..."]
}
```

`predicted_route` is a list of connected pieces (a new piece starts after a GPS gap or an HMM break).

## 5. Ground truth stays on the server

- `src/inference.py` (prediction) never opens `.route` files and never imports `src/labels.py`.
- `src/scoring.py` reads the reference route only after the prediction is finished.
- The response holds metric numbers only, never the reference route.
- `tests/test_api.py` checks all of this.

## 6. Logs for the evaluator

Every request is printed in this console, for example:

```
2026-11-18 10:59:49 | INFO | PREDICT trip=00000056: running preprocess -> candidates -> features -> lightgbm_tuned.joblib -> HMM
2026-11-18 10:59:54 | INFO | RESULT trip=00000056 | points=3958 | Top-1 96.29% (classifier only 91.22%) | route F1 0.967 | ...
2026-11-18 10:59:54 | INFO | POST /predict from 192.168.1.23 -> 200 in 4.29s
```

## 7. Test

```bash
python -m unittest tests.test_api -v      # uses student_test trips, never evaluator trips
```
