"""Classifier-only vs classifier + HMM/Viterbi.

Run from the repository root after 03_bayesian_optimization.py:

    python experiments/04_evaluate_hmm.py                     # full run
    python experiments/04_evaluate_hmm.py --max-train-trips 20 # quicker beta selection (not for the report)

Protocol
--------
1. Out-of-fold (OOF) probabilities for the 80 TRAIN trips: the tuned LightGBM settings
   (results/tuning/lightgbm/best_params.json) are refitted on 4 folds and predict the 5th, using
   the shared trip-grouped folds. So every train trip gets probabilities from a model that never
   saw it - like real unseen trips. Cached in results/hmm/train_oof_predictions.parquet.
2. HMM setting (beta) is chosen on these TRAIN OOF predictions only (grid search, best Top-1).
3. Student test is evaluated ONCE with the chosen beta, using the tuned model's saved
   student-test probabilities (results/tuning/lightgbm/student_test_predictions.parquet).
   Reported: Top-1 accuracy, route continuity, HMM breaks, length-weighted route
   precision/recall/F1 against the reference route, and decoding time.

Outputs: results/hmm/  (beta_search.csv, comparison.csv, per_trip.csv, config.json, figures/, maps/)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_DATA_ROOT                                              # noqa: E402
from src.data_loader import load_reference_route                                      # noqa: E402
from src.features import FEATURE_COLUMNS                                              # noqa: E402
from src.hmm import HMMConfig, expand_route, viterbi_trip                             # noqa: E402
from src.labels import canonical_arc_ids                                              # noqa: E402
from src.metrics import route_continuity, route_overlap, top1_predictions             # noqa: E402
from src.preprocessing import load_preprocessed_trip                                  # noqa: E402
from src.training import FOLD_FILE, load_partition                                    # noqa: E402
from src.tuning import build_tuned_model, grouped_fold_indices                        # noqa: E402

OUT = ROOT / 'results' / 'hmm'
TABLES = ROOT / 'data' / 'processed' / 'tables'
TUNED = ROOT / 'results' / 'tuning' / 'lightgbm'


def train_oof(seed, n_jobs):
    path = OUT / 'train_oof_predictions.parquet'
    if path.exists():
        print(f'using cached {path.relative_to(ROOT)}')
        return pd.read_parquet(path)
    params = json.loads((TUNED / 'best_params.json').read_text())['params']
    train, _ = load_partition('train')
    folds = grouped_fold_indices(train, pd.read_csv(FOLD_FILE, dtype={'trip_id': str}))
    X, y = train[FEATURE_COLUMNS].to_numpy(np.float32), train.label.to_numpy()
    prob = np.full(len(train), np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        model = build_tuned_model(params, seed, n_jobs).fit(X[tr], y[tr])
        prob[va] = model.predict_proba(X[va])[:, 1]
        print(f'  OOF fold {k}: {time.time() - t0:.1f}s', flush=True)
    out = train[['trip_id', 'observation_id', 'arc_id', 'sequence_id', 'label']].assign(prob=prob)
    out.to_parquet(path, index=False)
    return out


def with_geometry(pred, part):
    cand = pd.read_parquet(TABLES / f'{part}_candidates.parquet', columns=['trip_id', 'observation_id', 'arc_id', 'proj_frac'])
    df = pred.merge(cand, on=['trip_id', 'observation_id', 'arc_id'], how='left')
    assert df.proj_frac.notna().all() and len(df) == len(pred)
    return df


def run_hmm(df, part, cfg, trips=None):
    """Decode every trip; returns matched observations (all trips) and decode seconds per trip."""
    rows, secs = [], {}
    for tid, g in df.groupby('trip_id', sort=True):
        if trips is not None and tid not in trips:
            continue
        trip = load_preprocessed_trip(ROOT / 'data' / 'processed' / part / tid)
        t0 = time.perf_counter()
        m = viterbi_trip(trip, g, cfg)
        secs[tid] = time.perf_counter() - t0
        rows.append(m.assign(trip_id=tid))
    return pd.concat(rows, ignore_index=True), secs


def top1_acc(choice, obs):
    o = obs[obs.label_valid]
    pred = choice.set_index(['trip_id', 'observation_id']).arc_id.reindex(
        pd.MultiIndex.from_frame(o[['trip_id', 'observation_id']])).to_numpy()
    return float((pred == o.true_arc.to_numpy()).mean())


def per_trip_top1(choice, obs):
    o = obs[obs.label_valid]
    m = o.merge(choice[['trip_id', 'observation_id', 'arc_id']], on=['trip_id', 'observation_id'], how='left')
    return m.assign(ok=m.arc_id == m.true_arc).groupby('trip_id').ok.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--betas', nargs='+', type=float, default=[1, 2, 3, 5, 10, 20, 40])
    ap.add_argument('--max-train-trips', type=int, default=None, help='speed only; not for the report')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--maps', type=int, default=3, help='Folium maps for this many student-test trips')
    a = ap.parse_args()
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)

    # ---- 1-2. choose beta on TRAIN out-of-fold predictions only
    oof = with_geometry(train_oof(a.seed, a.n_jobs), 'train')
    train_obs = pd.read_parquet(TABLES / 'train_observations.parquet')
    trips = sorted(oof.trip_id.unique())
    if a.max_train_trips:
        trips = list(np.random.default_rng(a.seed).choice(trips, a.max_train_trips, replace=False))
    oof_sub, obs_sub = oof[oof.trip_id.isin(trips)], train_obs[train_obs.trip_id.isin(trips)]
    search = [dict(beta_m='classifier_only', top1_accuracy=top1_acc(top1_predictions(oof_sub), obs_sub))]
    print(f"TRAIN OOF classifier-only Top-1: {search[0]['top1_accuracy']:.4f} ({len(trips)} trips)", flush=True)
    for beta in a.betas:
        t0 = time.time()
        m, _ = run_hmm(oof_sub, 'train', HMMConfig(beta_m=beta), set(trips))
        search.append(dict(beta_m=beta, top1_accuracy=top1_acc(m, obs_sub), hmm_breaks=int(m.hmm_break.sum()),
                           seconds=time.time() - t0))
        print(f"  beta={beta:>5}: Top-1 {search[-1]['top1_accuracy']:.4f} | breaks {search[-1]['hmm_breaks']} "
              f"| {search[-1]['seconds']:.0f}s", flush=True)
    search_df = pd.DataFrame(search)
    search_df.to_csv(OUT / 'beta_search.csv', index=False)
    hmm_rows = search_df[search_df.beta_m != 'classifier_only']
    best_beta = float(hmm_rows.loc[hmm_rows.top1_accuracy.astype(float).idxmax(), 'beta_m'])
    cfg = HMMConfig(beta_m=best_beta)
    (OUT / 'config.json').write_text(json.dumps({**cfg.to_dict(), 'selected_on': f'TRAIN out-of-fold predictions, {len(trips)} trips',
                                                 'betas_tried': a.betas}, indent=2))
    print(f'selected beta = {best_beta} (train only)', flush=True)

    # ---- 3. student test, once
    test = with_geometry(pd.read_parquet(TUNED / 'student_test_predictions.parquet'), 'student_test')
    test_obs = pd.read_parquet(TABLES / 'student_test_observations.parquet')
    clf = top1_predictions(test)
    hmm, secs = run_hmm(test, 'student_test', cfg)

    per_trip, endpoints = [], {}
    for tid in sorted(test.trip_id.unique()):
        trip = load_preprocessed_trip(ROOT / 'data' / 'processed' / 'student_test' / tid)
        endpoints[tid] = (trip.arcs.from_node_id.to_numpy(), trip.arcs.to_node_id.to_numpy())
        route = load_reference_route(DEFAULT_DATA_ROOT / 'student_test' / tid, len(trip.arcs))
        canon, L = canonical_arc_ids(trip.arcs), trip.arcs.length_m.to_numpy()
        h = hmm[hmm.trip_id == tid]
        segs = expand_route(trip, h, cfg)
        r_clf = route_overlap(clf[clf.trip_id == tid].arc_id, route, L, canon)
        r_hmm = route_overlap([x for s in segs for x in s], route, L, canon)
        per_trip.append(dict(trip_id=tid, observations=int(test_obs[test_obs.trip_id == tid].label_valid.sum()),
                             hmm_segments=len(segs), hmm_breaks=int(h.hmm_break.sum()), decode_s=secs[tid],
                             **{f'clf_{k}': v for k, v in r_clf.items()}, **{f'hmm_{k}': v for k, v in r_hmm.items()}))
        if a.maps and len(per_trip) <= a.maps:
            make_map(trip, route, clf[clf.trip_id == tid], segs, OUT / 'maps' / f'{tid}.html')
    pt = pd.DataFrame(per_trip)
    pt['clf_top1'] = per_trip_top1(clf, test_obs).reindex(pt.trip_id).to_numpy()
    pt['hmm_top1'] = per_trip_top1(hmm, test_obs).reindex(pt.trip_id).to_numpy()
    pt.to_csv(OUT / 'per_trip.csv', index=False)

    n_obs = len(hmm)
    comp = pd.DataFrame([
        dict(method='classifier only (tuned LightGBM)', top1_accuracy=top1_acc(clf, test_obs),
             route_continuity=route_continuity(clf, endpoints),
             route_precision=pt.clf_route_precision.mean(), route_recall=pt.clf_route_recall.mean(),
             route_f1=pt.clf_route_f1.mean(), extra_ms_per_1000_obs=0.0),
        dict(method=f'classifier + HMM/Viterbi (beta={best_beta:g} m)', top1_accuracy=top1_acc(hmm, test_obs),
             route_continuity=route_continuity(hmm, endpoints),
             route_precision=pt.hmm_route_precision.mean(), route_recall=pt.hmm_route_recall.mean(),
             route_f1=pt.hmm_route_f1.mean(), extra_ms_per_1000_obs=1e6 * sum(secs.values()) / n_obs)])
    comp.to_csv(OUT / 'comparison.csv', index=False)
    plots(search_df, pt)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print('\nSTUDENT TEST (10 trips)\n', comp.round(4).to_string(index=False))
        print('\nper trip Top-1\n', pt[['trip_id', 'observations', 'clf_top1', 'hmm_top1', 'hmm_breaks', 'decode_s']].round(4).to_string(index=False))
    print('\nRoute precision/recall/F1 are trip means, length-weighted against the reference route.')


def plots(search_df, pt):
    h = search_df[search_df.beta_m != 'classifier_only']
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(h.beta_m.astype(float), h.top1_accuracy.astype(float), 'o-', label='classifier + HMM')
    ax.axhline(float(search_df.iloc[0].top1_accuracy), color='grey', ls='--', label='classifier only')
    ax.set_xscale('log'); ax.set_xlabel('beta (m)'); ax.set_ylabel('Top-1 accuracy (train out-of-fold)')
    ax.set_title('HMM setting chosen on training trips only'); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'beta_search.png', dpi=150); plt.close(fig)

    x = np.arange(len(pt)); fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, pt.clf_top1, 0.4, label='classifier only'); ax.bar(x + 0.2, pt.hmm_top1, 0.4, label='classifier + HMM')
    ax.set_xticks(x, pt.trip_id, rotation=45); ax.set_ylim(max(0, min(pt.clf_top1.min(), pt.hmm_top1.min()) - 0.05), 1)
    ax.set_ylabel('Top-1 accuracy'); ax.set_title('Student-test trips: classifier only vs classifier + HMM'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'figures' / 'per_trip_top1.png', dpi=150); plt.close(fig)


def make_map(trip, route, clf_choice, segments, path):
    try:
        import folium
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = trip.nodes[['latitude', 'longitude']].to_numpy()
    u, v = trip.arcs.from_node_id.to_numpy(), trip.arcs.to_node_id.to_numpy()
    obs = trip.observations[trip.observations.valid_observation]
    fmap = folium.Map(location=[obs.latitude.mean(), obs.longitude.mean()], zoom_start=15, tiles='OpenStreetMap')

    def line(arc_ids, color, weight, name, opacity=0.8):
        g = folium.FeatureGroup(name=name)
        for a in arc_ids:
            folium.PolyLine([nodes[u[a]].tolist(), nodes[v[a]].tolist()], color=color, weight=weight, opacity=opacity).add_to(g)
        g.add_to(fmap)

    line(np.unique(route), '#2ca02c', 8, 'reference route', 0.45)
    line(np.unique(clf_choice.arc_id), '#d62728', 4, 'classifier only (top-1 arcs)')
    line([a for s in segments for a in s], '#1f77b4', 3, 'classifier + HMM route')
    g = folium.FeatureGroup(name='GPS fixes')
    for la, lo in obs[['latitude', 'longitude']].to_numpy()[::3]:
        folium.CircleMarker([la, lo], radius=1.5, color='black', fill=True, opacity=0.6).add_to(g)
    g.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(str(path))


if __name__ == '__main__':
    main()
