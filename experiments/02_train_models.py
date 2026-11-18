"""Convenience: train several (default all five) models one after another on THIS laptop,
then build the comparison. To split work across laptops use the per-model scripts
experiments/02_train_<model>.py instead.

    python experiments/02_train_models.py
    python experiments/02_train_models.py --models lightgbm mlp
"""
import argparse, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import MODEL_INFO        # noqa: E402
from src.training import train_model     # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODEL_INFO), choices=list(MODEL_INFO))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cv-row-cap', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    a = ap.parse_args()
    for name in a.models:
        train_model(name, a.folds, a.cv_row_cap, a.seed, a.n_jobs)
    sys.argv = [sys.argv[0]]
    runpy.run_path(str(ROOT / 'experiments' / '02_compare_models.py'), run_name='__main__')
