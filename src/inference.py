"""End-to-end trip prediction pipeline (reference-free).

    raw trip (track, nodes, arcs)  ->  preprocess  ->  candidates  ->  21 features
        ->  tuned LightGBM probabilities  ->  HMM/Viterbi  ->  connected route

This module NEVER opens a .route file and never imports src/labels.py: ground truth is
handled only by src/scoring.py on the server, after prediction has finished.
All settings come from saved artifacts / config files - nothing trip-specific is hardcoded.
"""
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .candidates import CandidateConfig, generate_candidates
from .config import PROJECT_ROOT
from .data_loader import RawTrip, load_trip
from .features import FEATURE_COLUMNS, build_features
from .hmm import HMMConfig, expand_route, viterbi_trip
from .preprocessing import preprocess_trip

DEFAULT_MODEL = PROJECT_ROOT / 'artifacts' / 'models' / 'lightgbm_tuned.joblib'
DEFAULT_HMM_CONFIG = PROJECT_ROOT / 'results' / 'hmm' / 'config.json'


@dataclass
class MatchResult:
    trip_id: str
    trip: object                      # PreprocessedTrip (kept server-side for scoring)
    matched: object                   # DataFrame: observation_id, arc_id, sequence_id, hmm_break
    classifier_top1: object           # DataFrame: observation_id, arc_id (best candidate per point, no HMM)
    segments: list                    # list of arc-id lists (connected route pieces)
    timings_s: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


class MapMatcher:
    """Loads the trained model and HMM settings once; matches any trip folder on demand."""

    def __init__(self, model_path=DEFAULT_MODEL, hmm_config_path=DEFAULT_HMM_CONFIG,
                 candidate_config: CandidateConfig = CandidateConfig()):
        self.model_path, self.hmm_config_path = Path(model_path), Path(hmm_config_path)
        self.model = joblib.load(self.model_path)
        n_features = getattr(self.model, 'n_features_in_', len(FEATURE_COLUMNS))
        if n_features != len(FEATURE_COLUMNS):
            raise ValueError(f'model expects {n_features} features, pipeline builds {len(FEATURE_COLUMNS)}')
        hmm_keys = HMMConfig.__dataclass_fields__.keys()
        saved = json.loads(self.hmm_config_path.read_text()) if self.hmm_config_path.exists() else {}
        self.hmm_config = HMMConfig(**{k: v for k, v in saved.items() if k in hmm_keys})
        self.candidate_config = candidate_config

    def describe(self):
        return dict(model_file=self.model_path.name, model_type=type(self.model).__name__,
                    features=len(FEATURE_COLUMNS), candidate_config=self.candidate_config.to_dict(),
                    hmm_config=self.hmm_config.to_dict())

    def match_folder(self, folder, partition=''):
        """Load track/nodes/arcs only (load_trip never opens .route) and match."""
        t0 = time.perf_counter()
        raw = load_trip(Path(folder), partition)
        return self.match_raw(raw, load_s=time.perf_counter() - t0)

    def match_raw(self, raw: RawTrip, load_s=0.0) -> MatchResult:
        timings = {'load': load_s}

        t = time.perf_counter()
        trip = preprocess_trip(raw)
        timings['preprocess'] = time.perf_counter() - t

        t = time.perf_counter()
        cand = generate_candidates(trip, self.candidate_config)
        timings['candidates'] = time.perf_counter() - t

        t = time.perf_counter()
        feats = build_features(trip, cand)
        timings['features'] = time.perf_counter() - t

        t = time.perf_counter()
        prob = self.model.predict_proba(feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1] \
            if len(feats) else np.array([])
        scored = feats.assign(prob=prob)
        timings['model'] = time.perf_counter() - t

        t = time.perf_counter()
        matched = viterbi_trip(trip, scored[['observation_id', 'arc_id', 'proj_frac', 'prob']], self.hmm_config) \
            if len(scored) else scored.iloc[:0][['observation_id', 'arc_id']].assign(sequence_id=0, hmm_break=False)
        segments = expand_route(trip, matched, self.hmm_config) if len(matched) else []
        timings['hmm'] = time.perf_counter() - t

        top1 = (scored.loc[scored.groupby('observation_id').prob.idxmax(), ['observation_id', 'arc_id', 'sequence_id']]
                if len(scored) else scored.iloc[:0][['observation_id', 'arc_id']])
        valid = int(trip.observations.valid_observation.sum())
        counts = dict(gps_points=int(len(trip.observations)), valid_points=valid,
                      points_with_candidates=int(cand.observation_id.nunique()) if len(cand) else 0,
                      candidate_rows=int(len(cand)), hmm_breaks=int(matched.hmm_break.sum()) if len(matched) else 0,
                      route_segments=len(segments), route_arcs=int(sum(len(s) for s in segments)))
        timings['total_prediction'] = sum(v for k, v in timings.items())
        return MatchResult(raw.trip_id, trip, matched, top1, segments, timings, counts)


def route_polylines(trip, segments):
    """Connected route pieces as [[lat, lon], ...] lists (node by node), for drawing on a map."""
    lat, lon = trip.nodes.latitude.to_numpy(), trip.nodes.longitude.to_numpy()
    u, v = trip.arcs.from_node_id.to_numpy(), trip.arcs.to_node_id.to_numpy()
    lines = []
    for seg in segments:
        if not seg:
            continue
        nodes = [int(u[seg[0]])] + [int(v[a]) for a in seg]
        lines.append([[round(float(lat[n]), 7), round(float(lon[n]), 7)] for n in nodes])
    return lines


def gps_points(trip):
    """Valid GPS fixes as [[lat, lon, t], ...]."""
    o = trip.observations[trip.observations.valid_observation]
    return [[round(float(a), 7), round(float(b), 7), float(t)]
            for a, b, t in o[['latitude', 'longitude', 'timestamp_s']].to_numpy()]
