"""Merge the five per-model results into the report comparison table and figures.

Run after every model has been trained (on any laptop) and all results/models/<model>/
folders have been pulled from GitHub:

    python experiments/02_compare_models.py

Writes results/models/comparison.csv and results/models/figures/*.png.
Warns if models were trained with different folds, settings or library versions.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import MODEL_INFO  # noqa: E402

RES = ROOT / 'results' / 'models'
FIG = RES / 'figures'


def load():
    comp, cv_all, test, info = [], [], {}, {}
    for name in MODEL_INFO:
        d = RES / name
        if not (d / 'student_test_metrics.json').exists():
            print(f'  missing: {name} (not trained / not pulled yet)')
            continue
        cv = pd.read_csv(d / 'cv_folds.csv'); t = json.loads((d / 'student_test_metrics.json').read_text())
        i = json.loads((d / 'run_info.json').read_text())
        cv_all.append(cv); test[name] = t; info[name] = i
        comp.append({'Model': name, 'Algorithm family': i['family'], 'Key configuration': i['config'],
                     'CV macro F1 (mean)': cv.f1_macro.mean(), 'CV macro F1 (std)': cv.f1_macro.std(),
                     'CV train-fold macro F1': cv.train_f1_macro.mean(), 'CV log loss': cv.log_loss.mean(),
                     'CV top-1 accuracy': cv.top1_accuracy.mean(),
                     'Student-test top-1 accuracy': t['top1_accuracy'], 'Student-test macro F1': t['f1_macro'],
                     'Student-test precision (macro)': t['precision_macro'], 'Student-test recall (macro)': t['recall_macro'],
                     'Student-test log loss': t['log_loss'], 'Route continuity (classifier only)': t['route_continuity'],
                     'Candidate miss rate': t['candidate_miss_rate'], 'Train time full (s)': t['fit_s_full'],
                     'Inference (ms / 1000 obs)': t['inference_ms_per_1000_obs'], 'Trained on': i['machine']})
    return pd.DataFrame(comp), (pd.concat(cv_all) if cv_all else pd.DataFrame()), test, info


def consistency_warnings(info):
    if not info:
        return
    for key in ['n_folds', 'cv_row_cap', 'seed', 'sklearn', 'lightgbm', 'features']:
        vals = {n: str(i.get(key)) for n, i in info.items()}
        if len(set(vals.values())) > 1:
            print(f'  WARNING: models differ in {key}: {vals}')
    capped = [n for n, i in info.items() if i.get('cv_row_cap')]
    if capped:
        print(f'  WARNING: trained with --cv-row-cap (quick mode, not for the report): {capped}')
    speed_note = len({i["machine"] for i in info.values()}) > 1
    if speed_note:
        print('  NOTE: models were trained on different machines - train/inference times are not directly comparable;'
              ' say so in the report or re-time inference on one machine.')


def plots(comp, test):
    FIG.mkdir(parents=True, exist_ok=True)
    names = comp.Model.tolist(); x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, comp['CV macro F1 (mean)'], yerr=comp['CV macro F1 (std)'], capsize=5, color='#4C72B0')
    for i, r in comp.iterrows():
        ax.text(i, r['CV macro F1 (mean)'] + 0.003, f"{r['CV macro F1 (mean)']:.3f}", ha='center', fontsize=9)
    ax.set_ylabel('Macro F1 (trip-grouped CV, mean ± std)'); ax.set_title('Cross-validation macro F1 (80 train trips)')
    ax.set_ylim(max(0, comp['CV macro F1 (mean)'].min() - 0.1), 1); fig.tight_layout(); fig.savefig(FIG / 'cv_macro_f1.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.38
    ax.bar(x - w / 2, comp['Student-test top-1 accuracy'], w, label='Top-1 accuracy (observation)', color='#55A868')
    ax.bar(x + w / 2, comp['Student-test macro F1'], w, label='Macro F1 (candidate)', color='#C44E52')
    ax.set_xticks(x, names); ax.legend(); ax.set_title('Student-test performance (10 trips)')
    ax.set_ylim(max(0, min(comp['Student-test top-1 accuracy'].min(), comp['Student-test macro F1'].min()) - 0.1), 1)
    fig.tight_layout(); fig.savefig(FIG / 'student_test_top1_f1.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, comp['CV log loss'], 0.4, label='CV'); ax.bar(x + 0.2, comp['Student-test log loss'], 0.4, label='Student test')
    ax.set_xticks(x, names); ax.set_ylabel('Log loss (lower is better)'); ax.legend(); ax.set_title('Probability quality')
    fig.tight_layout(); fig.savefig(FIG / 'log_loss.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, comp['CV train-fold macro F1'], 0.4, label='Training folds'); ax.bar(x + 0.2, comp['CV macro F1 (mean)'], 0.4, label='Validation folds')
    ax.set_xticks(x, names); ax.set_ylim(0.5, 1); ax.legend(); ax.set_title('Training vs validation macro F1')
    fig.tight_layout(); fig.savefig(FIG / 'train_vs_val_f1.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.3))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = np.array(test[n]['confusion']); ax.imshow(cm, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f'{v:,}', ha='center', va='center', color='white' if v > cm.max() / 2 else 'black', fontsize=8)
        ax.set_xticks([0, 1], ['pred 0', 'pred 1']); ax.set_yticks([0, 1], ['true 0', 'true 1']); ax.set_title(n, fontsize=9)
    fig.suptitle('Student-test confusion matrices (threshold 0.5)'); fig.tight_layout(); fig.savefig(FIG / 'confusion_matrices.png', dpi=150); plt.close(fig)

    h = RES / 'mlp' / 'training_history.csv'
    if h.exists():   # only model with an epoch-wise history
        d = pd.read_csv(h); fig, ax1 = plt.subplots(figsize=(7, 4)); ax1.plot(d.epoch, d.train_loss, 'o-', label='training loss')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('training log loss'); ax2 = ax1.twinx()
        ax2.plot(d.epoch, d.val_accuracy, 's--', color='#C44E52', label='validation accuracy'); ax2.set_ylabel('validation accuracy')
        fig.legend(loc='upper right'); ax1.set_title('MLP training history (final refit)'); fig.tight_layout()
        fig.savefig(FIG / 'mlp_training_history.png', dpi=150); plt.close(fig)
    for n in ['lightgbm', 'random_forest']:
        f = RES / n / 'feature_importance.csv'
        if f.exists():
            d = pd.read_csv(f).iloc[::-1]; fig, ax = plt.subplots(figsize=(7, 6)); ax.barh(d.feature, d.importance)
            ax.set_title(f'{n} feature importance'); fig.tight_layout(); fig.savefig(FIG / f'{n}_feature_importance.png', dpi=150); plt.close(fig)


def main():
    comp, cv, test, info = load()
    if comp.empty:
        sys.exit('no trained models found under results/models/<model>/')
    consistency_warnings(info)
    comp.to_csv(RES / 'comparison.csv', index=False)
    plots(comp, test)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        cols = ['Model', 'CV macro F1 (mean)', 'CV macro F1 (std)', 'CV log loss', 'Student-test top-1 accuracy',
                'Student-test macro F1', 'Student-test log loss', 'Route continuity (classifier only)', 'Inference (ms / 1000 obs)']
        print(comp[cols].round(4).to_string(index=False))
    print(f'\nwrote {(RES / "comparison.csv").relative_to(ROOT)} and figures in {FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
