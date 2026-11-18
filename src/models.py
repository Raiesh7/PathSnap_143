"""The five candidate classifiers compared in experiments/02_train_models.py.

Each returns a fresh, unfitted scikit-learn compatible estimator with predict_proba.
Models that cannot use NaN (missing heading/speed at sequence starts or standstill) get
median imputation + missing-value indicator columns + standard scaling. These transforms live
INSIDE the pipeline, so they are fitted on the training fold only (no leakage).
Initial hyperparameters are reasonable defaults, not tuned values; tuning is step 03.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_INFO = {
    'logistic_regression': dict(family='Linear / probabilistic', max_train_rows=None),
    'knn': dict(family='Instance-based', max_train_rows=100_000),   # prediction cost grows with training size
    'random_forest': dict(family='Bagging ensemble', max_train_rows=None),
    'lightgbm': dict(family='Gradient boosting', max_train_rows=None),
    'mlp': dict(family='Neural network', max_train_rows=None),
}


def _scaled(estimator):
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), estimator)


def build_model(name, seed=42, n_jobs=-1):
    if name == 'logistic_regression':
        return _scaled(LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    if name == 'knn':
        return _scaled(KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=n_jobs))
    if name == 'random_forest':
        # sklearn >= 1.4 random forests accept NaN directly; imputation kept for robustness
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                             RandomForestClassifier(n_estimators=150, min_samples_leaf=20, max_features='sqrt',
                                                    max_samples=0.3, n_jobs=n_jobs, random_state=seed))
    if name == 'lightgbm':
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=n_jobs, random_state=seed, verbose=-1)   # handles NaN natively
    if name == 'mlp':
        return _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                     learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=5, random_state=seed))
    raise ValueError(f'unknown model {name}')


def model_config(name):
    """Human-readable key configuration for the comparison table."""
    est = build_model(name)
    final = est.steps[-1][1] if hasattr(est, 'steps') else est
    keys = {'logistic_regression': ['C', 'max_iter'], 'knn': ['n_neighbors', 'weights'],
            'random_forest': ['n_estimators', 'min_samples_leaf', 'max_features', 'max_samples'],
            'lightgbm': ['n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples'],
            'mlp': ['hidden_layer_sizes', 'alpha', 'learning_rate_init', 'batch_size']}[name]
    p = final.get_params()
    cfg = ', '.join(f'{k}={p[k]}' for k in keys)
    if MODEL_INFO[name]['max_train_rows']:
        cfg += f", train rows<={MODEL_INFO[name]['max_train_rows']:,}"
    return cfg
