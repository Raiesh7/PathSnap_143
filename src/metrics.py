"""Evaluation metrics for candidate classification and map-matching.

All functions expect candidate rows restricted to label-valid observations
(observations whose reference label is trustworthy; see src/labels.py).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y_true, prob, threshold=0.5):
    """Binary candidate classification metrics (label 1 = candidate is the reference arc)."""
    y_true = np.asarray(y_true); prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred),
                precision_macro=precision_score(y_true, pred, average='macro', zero_division=0),
                recall_macro=recall_score(y_true, pred, average='macro', zero_division=0),
                f1_macro=f1_score(y_true, pred, average='macro'),
                f1_positive=f1_score(y_true, pred, pos_label=1),
                log_loss=log_loss(y_true, prob, labels=[0, 1]),
                confusion=confusion_matrix(y_true, pred, labels=[0, 1]).tolist())


def top1_predictions(cand, score_col='prob'):
    """Highest-scoring candidate per observation -> DataFrame[trip_id, observation_id, arc_id, sequence_id]."""
    idx = cand.groupby(['trip_id', 'observation_id'], sort=False)[score_col].idxmax()
    return cand.loc[idx.to_numpy(), ['trip_id', 'observation_id', 'arc_id', 'sequence_id']]


def observation_top1_accuracy(cand, observations, score_col='prob'):
    """Share of label-valid observations whose top-scored candidate is the reference arc.

    Observations without any candidate, or whose reference arc is not among the candidates,
    count as errors (so this already includes the candidate miss rate).
    """
    obs = observations[observations.label_valid]
    top = top1_predictions(cand, score_col).set_index(['trip_id', 'observation_id']).arc_id
    pred = top.reindex(pd.MultiIndex.from_frame(obs[['trip_id', 'observation_id']])).to_numpy()
    correct = pred == obs.true_arc.to_numpy()
    return dict(top1_accuracy=float(correct.mean()),
                candidate_miss_rate=float(1 - obs.true_in_candidates.mean()),
                top1_accuracy_when_covered=float(correct[obs.true_in_candidates.to_numpy()].mean()))


def route_continuity(top1, arc_endpoints):
    """Share of consecutive observation pairs (same sequence) whose predicted arcs are the same arc
    or directly connected (end node of the first = start node of the second). Label-free.

    arc_endpoints: dict trip_id -> (from_node_array, to_node_array)
    """
    ok = total = 0
    for trip_id, g in top1.sort_values(['trip_id', 'observation_id']).groupby('trip_id', sort=False):
        u, v = arc_endpoints[trip_id]
        a = g.arc_id.to_numpy(); s = g.sequence_id.to_numpy()
        same_seq = s[1:] == s[:-1]
        a0, a1 = a[:-1][same_seq], a[1:][same_seq]
        ok += int(((a0 == a1) | (v[a0] == u[a1])).sum()); total += int(same_seq.sum())
    return float(ok / total) if total else float('nan')


def route_overlap(predicted_arcs, reference_arcs, arc_lengths, canonical=None):
    """Length-weighted overlap between a predicted route and the reference route (sets of arcs).

    precision = length of predicted arcs that are on the reference / length of predicted arcs
    recall    = length of reference arcs that were predicted     / length of reference arcs
    canonical: optional array mapping duplicate arc rows to one id.
    """
    pred = np.unique(np.asarray(list(predicted_arcs), dtype=np.int64))
    ref = np.unique(np.asarray(list(reference_arcs), dtype=np.int64))
    if canonical is not None:
        pred, ref = np.unique(canonical[pred]), np.unique(canonical[ref])
    L = np.asarray(arc_lengths, dtype=float)
    both = np.intersect1d(pred, ref)
    p = L[both].sum() / L[pred].sum() if L[pred].sum() > 0 else float('nan')
    r = L[both].sum() / L[ref].sum() if L[ref].sum() > 0 else float('nan')
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return dict(route_precision=float(p), route_recall=float(r), route_f1=float(f))
