"""PathSnap FastAPI backend (Laptop 1).

Run from the repository root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables (all optional):
    PATHSNAP_TRIPS_DIR   folder holding one sub-folder per trip   (default: data/Student folder/evaluator_test)
    PATHSNAP_MODEL       trained model file                        (default: artifacts/models/lightgbm_tuned.joblib)
    PATHSNAP_HMM_CONFIG  HMM settings chosen on training trips     (default: results/hmm/config.json)
    PATHSNAP_CORS        comma-separated UI origins allowed        (default: * )

Endpoints:
    GET  /health           model + settings loaded, number of trips available
    GET  /trips            the trip IDs the UI may offer (no ground truth)
    POST /predict          {"trip_id": "..."} -> full pipeline, server-side scoring, route for the map

Ground truth: the prediction pipeline (src/inference.py) never reads .route files. After the route is
predicted, src/scoring.py reads the reference on this server to compute metrics. Only the metric
numbers are returned - never the reference route - so ground truth reaches neither the model nor the UI.
Every request is logged to this console (shown to the evaluator).
"""
import logging
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import PROJECT_ROOT
from src.inference import DEFAULT_HMM_CONFIG, DEFAULT_MODEL, MapMatcher, gps_points, route_polylines
from src.scoring import score_match

TRIPS_DIR = Path(os.environ.get('PATHSNAP_TRIPS_DIR', PROJECT_ROOT / 'data' / 'Student folder' / 'evaluator_test'))
MODEL_PATH = Path(os.environ.get('PATHSNAP_MODEL', DEFAULT_MODEL))
HMM_CONFIG = Path(os.environ.get('PATHSNAP_HMM_CONFIG', DEFAULT_HMM_CONFIG))
CORS = [o.strip() for o in os.environ.get('PATHSNAP_CORS', '*').split(',') if o.strip()]
TRIP_ID = re.compile(r'^[0-9A-Za-z_-]{1,64}$')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('pathsnap')

app = FastAPI(title='PathSnap map-matching API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=CORS, allow_methods=['GET', 'POST'], allow_headers=['*'])

matcher = MapMatcher(MODEL_PATH, HMM_CONFIG)
log.info('model loaded: %s | HMM beta=%.1f m | trips folder: %s', MODEL_PATH.name, matcher.hmm_config.beta_m, TRIPS_DIR)


def available_trips():
    """Trip folders that have the three prediction inputs (the .route file is not needed to list them)."""
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir()
                  if p.is_dir() and all((p / f'{p.name}.{e}').is_file() for e in ('track', 'nodes', 'arcs')))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware('http')
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else '?'
    log.info('%s %s from %s -> %d in %.2fs', request.method, request.url.path, client, response.status_code,
             time.perf_counter() - start)
    return response


@app.get('/health')
def health():
    return dict(status='ok', trips_available=len(available_trips()), **matcher.describe())


@app.get('/trips')
def trips():
    ids = available_trips()
    return dict(count=len(ids), trips=[dict(trip_id=t) for t in ids])


@app.post('/predict')
def predict(req: PredictRequest):
    trip_id = req.trip_id.strip()
    if not TRIP_ID.match(trip_id) or trip_id not in available_trips():
        log.warning('rejected trip_id %r (not in %s)', trip_id, TRIPS_DIR.name)
        raise HTTPException(status_code=404, detail=f'unknown trip_id {trip_id!r}')
    folder = TRIPS_DIR / trip_id
    log.info('PREDICT trip=%s: running preprocess -> candidates -> features -> %s -> HMM', trip_id, MODEL_PATH.name)

    result = matcher.match_folder(folder)                       # prediction: no ground truth involved
    t = time.perf_counter()
    if (folder / f'{trip_id}.route').is_file():
        metrics = score_match(result, folder)                   # server-side scoring, after prediction
    else:
        metrics = None
        log.warning('trip=%s has no reference route on this server: metrics not available', trip_id)
    scoring_s = time.perf_counter() - t

    c, tm = result.counts, result.timings_s
    if metrics:
        log.info('RESULT trip=%s | points=%d | Top-1 %.2f%% (classifier only %.2f%%) | route F1 %.3f | '
                 'continuity %.2f%% | segments=%d | predict %.2fs + scoring %.2fs', trip_id, c['gps_points'],
                 100 * metrics['top1_accuracy'], 100 * metrics['classifier_only_top1_accuracy'], metrics['route_f1'],
                 100 * metrics['route_continuity'], c['route_segments'], tm['total_prediction'], scoring_s)
    return dict(
        trip_id=trip_id,
        counts=c,
        metrics=metrics,
        timings_s={**{k: round(v, 3) for k, v in tm.items()}, 'scoring': round(scoring_s, 3)},
        model=dict(file=MODEL_PATH.name, hmm_beta_m=matcher.hmm_config.beta_m),
        gps_points=gps_points(result.trip),                    # [[lat, lon, t], ...] raw input
        predicted_route=route_polylines(result.trip, result.segments),   # [[[lat, lon], ...], ...] prediction
    )
