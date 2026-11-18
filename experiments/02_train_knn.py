"""Train + evaluate the 'knn' candidate classifier (one of the five compared models).

Run from the repository root, on any laptop, after 01b and 01c have built data/processed:

    python experiments/02_train_knn.py
    python experiments/02_train_knn.py --folds 2 --cv-row-cap 150000   # quick check only

Writes results/models/knn/ and artifacts/models/knn.joblib. Uses the shared fold file
results/models/cv_fold_assignment.csv (pull it from GitHub before running).
Compare all models afterwards with: python experiments/02_compare_models.py
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import train_model  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cv-row-cap', type=int, default=None, help='speed-only cap on training rows per CV fold')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    a = ap.parse_args()
    train_model('knn', a.folds, a.cv_row_cap, a.seed, a.n_jobs)
