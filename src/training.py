"""Shared training/evaluation routine used by the per-model scripts experiments/02_train_*.py.

Each model writes ONLY to its own folder, so team members can train different models on
different laptops and push without overwriting each other:

    results/models/<model>/cv_folds.csv               per-fold validation + training-fold metrics
    results/models/<model>/student_test_metrics.json  refit on 80 train trips, scored on student_test
    results/models/<model>/student_test_predictions.parquet  candidate probabilities (HMM input)
    results/models/<model>/run_info.json              config, row counts, timings, library versions
    results/models/<model>/training_history.csv       MLP only (epoch-wise)
    results/models/<model>/feature_importance.csv     LightGBM / Random Forest
    artifacts/models/<model>.joblib                   refitted model

Cross-validation folds are defined ONCE in results/models/cv_fold_assignment.csv (trip -> fold)
and shared by all models. The file is created by whoever runs first; commit it so every
laptop uses identical folds. Fold assignment depends only on sorted trip IDs and the seed.
"""
import json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn

from .features import FEATURE_COLUMNS
from .metrics import candidate_metrics, observation_top1_accuracy, route_continuity, top1_predictions
from .models import MODEL_INFO, build_model, model_config
from .preprocessing import load_preprocessed_trip

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'data' / 'processed' / 'tables'
RESULTS = ROOT / 'results' / 'models'
ARTIFACTS = ROOT / 'artifacts' / 'models'
FOLD_FILE = RESULTS / 'cv_fold_assignment.csv'


def load_partition(part):
    cand = pd.read_parquet(TABLES / f'{part}_candidates.parquet')
    obs = pd.read_parquet(TABLES / f'{part}_observations.parquet')
    return cand[cand.label_valid].reset_index(drop=True), obs


def arc_endpoints(part, trip_ids):
    out = {}
    for t in trip_ids:
        trip = load_preprocessed_trip(ROOT / 'data' / 'processed' / part / t)
        out[t] = (trip.arcs.from_node_id.to_numpy(), trip.arcs.to_node_id.to_numpy())
    return out


def fold_assignment(trip_ids, n_folds=5, seed=42):
    """Trip-grouped folds. Reuse the committed file if present so all laptops agree."""
    trips = np.array(sorted(set(trip_ids)))
    if FOLD_FILE.exists():
        f = pd.read_csv(FOLD_FILE, dtype={'trip_id': str})
        if set(f.trip_id) != set(trips):
            raise ValueError(f'{FOLD_FILE} does not match the current training trips; delete it or rebuild data')
        if f.fold.nunique() != n_folds:
            raise ValueError(f'{FOLD_FILE} has {f.fold.nunique()} folds but --folds={n_folds}')
        print(f'using existing fold assignment {FOLD_FILE.relative_to(ROOT)}')
        return f
    perm = np.random.default_rng(seed).permutation(trips)
    f = pd.DataFrame({'trip_id': perm, 'fold': np.arange(len(perm)) % n_folds}).sort_values('trip_id')
    FOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(FOLD_FILE, index=False)
    print(f'created fold assignment {FOLD_FILE.relative_to(ROOT)} - commit it so other laptops use the same folds')
    return f


def _subsample(idx, cap, rng):
    return idx if cap is None or len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False))


def _X(df):
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def evaluate(model, cand, obs, endpoints):
    t0 = time.perf_counter()
    prob = model.predict_proba(_X(cand))[:, 1]
    infer_s = time.perf_counter() - t0
    scored = cand.assign(prob=prob)
    m = candidate_metrics(cand.label.to_numpy(), prob)
    m.update(observation_top1_accuracy(scored, obs))
    m['route_continuity'] = route_continuity(top1_predictions(scored), endpoints)
    n_obs = scored[['trip_id', 'observation_id']].drop_duplicates().shape[0]
    m.update(inference_s=infer_s, inference_ms_per_1000_obs=1e6 * infer_s / max(n_obs, 1))
    return m, scored


def train_model(name, n_folds=5, cv_row_cap=None, seed=42, n_jobs=-1):
    if name not in MODEL_INFO:
        raise ValueError(f'unknown model {name}; choose from {list(MODEL_INFO)}')
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train, train_obs = load_partition('train')
    test, test_obs = load_partition('student_test')
    ep_train = arc_endpoints('train', train.trip_id.unique())
    ep_test = arc_endpoints('student_test', test.trip_id.unique())
    X, y = _X(train), train.label.to_numpy()
    folds = fold_assignment(train.trip_id, n_folds, seed).set_index('trip_id').fold
    row_fold = folds.reindex(train.trip_id).to_numpy()
    print(f'[{name}] train rows {len(train):,} ({train.trip_id.nunique()} trips) | student_test rows {len(test):,} '
          f'| {len(FEATURE_COLUMNS)} features | {n_folds} folds', flush=True)

    rng = np.random.default_rng(seed)
    cap = MODEL_INFO[name]['max_train_rows']
    rows = []
    for k in range(n_folds):
        tr, va = np.where(row_fold != k)[0], np.where(row_fold == k)[0]
        va_trips = np.unique(train.trip_id.to_numpy()[va])
        tr_fit = _subsample(tr, min(c for c in [cap, cv_row_cap, len(tr)] if c), rng)
        model = build_model(name, seed, n_jobs)
        t0 = time.perf_counter(); model.fit(X[tr_fit], y[tr_fit]); fit_s = time.perf_counter() - t0
        m, _ = evaluate(model, train.iloc[va], train_obs[train_obs.trip_id.isin(va_trips)], ep_train)
        tr_eval = _subsample(tr_fit, 200_000, rng)
        m_tr = candidate_metrics(y[tr_eval], model.predict_proba(X[tr_eval])[:, 1])
        rows.append(dict(model=name, fold=k, train_rows=len(tr_fit), val_trips=len(va_trips), fit_s=fit_s,
                         train_f1_macro=m_tr['f1_macro'], train_log_loss=m_tr['log_loss'],
                         **{kk: vv for kk, vv in m.items() if kk != 'confusion'}))
        pd.DataFrame(rows).to_csv(out / 'cv_folds.csv', index=False)   # saved after every fold
        print(f"  fold {k}: F1 {m['f1_macro']:.4f} (train {m_tr['f1_macro']:.4f}) | logloss {m['log_loss']:.4f} "
              f"| top1 {m['top1_accuracy']:.4f} | fit {fit_s:.1f}s", flush=True)

    # refit on all 80 train trips -> student_test
    fit_idx = _subsample(np.arange(len(X)), cap, rng)
    model = build_model(name, seed, n_jobs)
    t0 = time.perf_counter(); model.fit(X[fit_idx], y[fit_idx]); fit_s = time.perf_counter() - t0
    m, scored = evaluate(model, test, test_obs, ep_test)
    m.update(fit_s_full=fit_s, train_rows_full=int(len(fit_idx)))
    (out / 'student_test_metrics.json').write_text(json.dumps(m, indent=2))
    scored[['trip_id', 'observation_id', 'arc_id', 'sequence_id', 'label', 'prob']] \
        .to_parquet(out / 'student_test_predictions.parquet', index=False)
    joblib.dump(model, ARTIFACTS / f'{name}.joblib', compress=3)

    final = model.steps[-1][1] if hasattr(model, 'steps') else model
    if name == 'mlp':
        pd.DataFrame({'epoch': np.arange(1, len(final.loss_curve_) + 1), 'train_loss': final.loss_curve_,
                      'val_accuracy': final.validation_scores_}).to_csv(out / 'training_history.csv', index=False)
    if name == 'lightgbm':
        imp = final.booster_.feature_importance('gain')
    elif name == 'random_forest':
        imp = final.feature_importances_[:len(FEATURE_COLUMNS)]   # first columns = original features
    else:
        imp = None
    if imp is not None:
        pd.DataFrame({'feature': FEATURE_COLUMNS, 'importance': imp}).sort_values('importance', ascending=False) \
            .to_csv(out / 'feature_importance.csv', index=False)

    import lightgbm
    info = dict(model=name, family=MODEL_INFO[name]['family'], config=model_config(name), n_folds=n_folds,
                cv_row_cap=cv_row_cap, seed=seed, features=FEATURE_COLUMNS, total_runtime_s=time.time() - t_start,
                machine=platform.node(), python=platform.python_version(), sklearn=sklearn.__version__,
                lightgbm=lightgbm.__version__, numpy=np.__version__, pandas=pd.__version__)
    (out / 'run_info.json').write_text(json.dumps(info, indent=2))
    print(f"  STUDENT_TEST: F1 {m['f1_macro']:.4f} | logloss {m['log_loss']:.4f} | top1 {m['top1_accuracy']:.4f} "
          f"| continuity {m['route_continuity']:.4f} | infer {m['inference_ms_per_1000_obs']:.1f} ms/1000 obs", flush=True)
    print(f'  saved to {out.relative_to(ROOT)} and {(ARTIFACTS / (name + ".joblib")).relative_to(ROOT)}')
    if cv_row_cap:
        print('  NOTE: --cv-row-cap was used; rerun without it for report numbers.')
    return m
