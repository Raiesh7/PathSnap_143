"""Training-only Bayesian search helpers; student testing is final evaluation only."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

SEARCH_SPACE = {
    'n_estimators': dict(kind='int', low=200, high=700, step=100),
    'learning_rate': dict(kind='float', low=0.02, high=0.15, log=True),
    'num_leaves': dict(kind='int', low=15, high=95),
    'min_child_samples': dict(kind='int', low=20, high=250, log=True),
    'subsample': dict(kind='float', low=0.6, high=1.0),
    'colsample_bytree': dict(kind='float', low=0.6, high=1.0),
    'reg_alpha': dict(kind='float', low=0.0, high=5.0),
    'reg_lambda': dict(kind='float', low=0.0, high=10.0),
}
BASELINE_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0)


def suggest_params(trial):
    params = {}
    for name, spec in SEARCH_SPACE.items():
        options = {k:v for k,v in spec.items() if k != 'kind'}
        method = trial.suggest_int if spec['kind'] == 'int' else trial.suggest_float
        params[name] = method(name, **options)
    return params


def build_tuned_model(params, seed=42, n_jobs=4):
    return LGBMClassifier(**params, subsample_freq=1, random_state=seed,
                          n_jobs=n_jobs, verbose=-1)


def grouped_fold_indices(candidates, assignment):
    """Validate and reuse the committed folds, never split by observation/sequence."""
    required = {'trip_id','fold'}
    if not required <= set(assignment.columns):
        raise ValueError('Fold assignment requires trip_id and fold')
    if assignment.trip_id.duplicated().any() or assignment.fold.isna().any():
        raise ValueError('Missing or duplicate trip assignments')
    if set(assignment.trip_id) != set(candidates.trip_id):
        raise ValueError('Fold assignments do not exactly match training trips')
    folds = assignment.fold.to_numpy()
    if not np.issubdtype(folds.dtype, np.integer) or set(folds) != set(range(5)):
        raise ValueError('Expected five integer folds numbered 0 through 4')
    row_fold = assignment.set_index('trip_id').fold.reindex(candidates.trip_id).to_numpy()
    result = []
    for k in range(5):
        tr, va = np.flatnonzero(row_fold != k), np.flatnonzero(row_fold == k)
        if not len(tr) or not len(va):
            raise ValueError('Empty training or validation fold')
        if len(np.unique(candidates.label.to_numpy()[tr])) != 2:
            raise ValueError('A training fold lacks a binary class')
        result.append((tr,va))
    return result


def flatten_trial_folds(study):
    return pd.DataFrame([dict(trial=t.number, **row) for t in study.trials
                         for row in t.user_attrs.get('fold_metrics', [])])
