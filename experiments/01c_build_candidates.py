"""Build the candidate/feature/label tables used for model training.

Run from the repository root after 01b_preprocess_dataset.py --cache-dir data/processed:

    python experiments/01c_build_candidates.py
    python experiments/01c_build_candidates.py --sweep      # candidate-miss-rate grid on TRAIN only

Writes data/processed/tables/{train,student_test}_candidates.parquet (+ _observations.parquet)
and results/candidates/summary.csv / summary.json. Reads only train and student_test.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidates import CandidateConfig, generate_candidates          # noqa: E402
from src.features import build_features                                  # noqa: E402
from src.labels import observation_labels, attach_labels                 # noqa: E402
from src.preprocessing import load_preprocessed_trip                     # noqa: E402
from src.data_loader import load_reference_route                         # noqa: E402
from src.config import DEFAULT_DATA_ROOT                                 # noqa: E402


def process(cache_dir, raw_dir, cfg, with_features=True):
    trip = load_preprocessed_trip(cache_dir)
    cand = generate_candidates(trip, cfg)
    route = load_reference_route(raw_dir, len(trip.arcs))
    labels = observation_labels(trip, route, valid_within_m=cfg.radius_m)
    cand, labels = attach_labels(cand, labels)
    if with_features:
        cand = build_features(trip, cand)
    cand.insert(0, 'trip_id', trip.trip_id)
    labels.insert(0, 'trip_id', trip.trip_id)
    valid = labels.label_valid
    stats = dict(trip_id=trip.trip_id, observations=len(labels), candidate_rows=len(cand),
                 obs_without_candidates=int(len(labels) - cand.observation_id.nunique()),
                 label_valid_obs=int(valid.sum()),
                 candidate_miss_rate_valid=float(1 - labels.true_in_candidates[valid].mean()) if valid.any() else np.nan,
                 median_align_dist_m=float(labels.align_dist_m.median()))
    return cand, labels, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root', default=ROOT / 'data' / 'processed', type=Path)
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT, type=Path)
    ap.add_argument('--radius', type=float, default=CandidateConfig.radius_m)
    ap.add_argument('--k', type=int, default=CandidateConfig.max_candidates)
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    out_res = ROOT / 'results' / 'candidates'; out_res.mkdir(parents=True, exist_ok=True)

    if a.sweep:   # model-free choice of radius/K using TRAIN trips only
        rows = []
        trips = sorted(p for p in (a.cache_root / 'train').iterdir() if p.is_dir())
        for radius in [30, 50, 75, 100]:
            for k in [3, 5, 8, 12]:
                cfg = CandidateConfig(radius_m=radius, max_candidates=k)
                st = pd.DataFrame([process(t, a.data_root / 'train' / t.name, cfg, False)[2] for t in trips])
                w = st.label_valid_obs
                rows.append(dict(radius_m=radius, k=k, candidate_miss_rate=float((st.candidate_miss_rate_valid * w).sum() / w.sum()),
                                 candidates_per_obs=float(st.candidate_rows.sum() / st.observations.sum())))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out_res / 'sweep_train.csv', index=False)
        return

    cfg = CandidateConfig(radius_m=a.radius, max_candidates=a.k)
    out_tab = a.cache_root / 'tables'; out_tab.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for part in ['train', 'student_test']:
        t0 = time.time(); C, L = [], []
        for t in sorted(p for p in (a.cache_root / part).iterdir() if p.is_dir()):
            cand, lab, st = process(t, a.data_root / part / t.name, cfg)
            C.append(cand); L.append(lab); st['partition'] = part; all_stats.append(st)
        pd.concat(C, ignore_index=True).to_parquet(out_tab / f'{part}_candidates.parquet', index=False)
        pd.concat(L, ignore_index=True).to_parquet(out_tab / f'{part}_observations.parquet', index=False)
        print(f'{part}: {len(C)} trips in {time.time() - t0:.1f}s', flush=True)

    st = pd.DataFrame(all_stats)
    st.to_csv(out_res / 'summary.csv', index=False)
    summary = {'config': cfg.to_dict()}
    for part, g in st.groupby('partition'):
        w = g.label_valid_obs
        summary[part] = dict(trips=len(g), observations=int(g.observations.sum()), candidate_rows=int(g.candidate_rows.sum()),
                             label_valid_obs=int(w.sum()), label_valid_share=float(w.sum() / g.observations.sum()),
                             candidate_miss_rate_valid=float((g.candidate_miss_rate_valid * w).sum() / w.sum()))
    (out_res / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
