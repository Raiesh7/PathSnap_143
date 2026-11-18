"""Server-side evaluation of a finished prediction against the trip's reference route.

Called by the API only AFTER src/inference.py has produced its route. The reference route is
read here, used to compute metrics, and never returned: the API response carries numbers only,
so ground truth reaches neither the model nor the UI.

Metrics use the same definitions as the experiments (src/labels.py alignment, label-valid points).
"""
from pathlib import Path

from .data_loader import load_reference_route
from .labels import canonical_arc_ids, observation_labels
from .metrics import route_overlap


def _top1(choice, labels):
    """Share of label-valid observations whose chosen arc is the reference arc."""
    lab = labels[labels.label_valid].set_index('observation_id')
    pred = choice.set_index('observation_id').arc_id.reindex(lab.index).to_numpy()
    return float((pred == lab.true_arc.to_numpy()).mean()) if len(lab) else float('nan')


def _continuity(choice, trip):
    """Share of consecutive observation pairs (same sequence) on the same or a directly connected arc."""
    u, v = trip.arcs.from_node_id.to_numpy(), trip.arcs.to_node_id.to_numpy()
    c = choice.merge(trip.observations[['observation_id', 'sequence_id']], on='observation_id',
                     suffixes=('', '_obs')).sort_values('observation_id')
    a, s = c.arc_id.to_numpy(), c.sequence_id_obs.to_numpy()
    same = s[1:] == s[:-1]
    a0, a1 = a[:-1][same], a[1:][same]
    return float(((a0 == a1) | (v[a0] == u[a1])).mean()) if same.any() else float('nan')


def score_match(result, trip_folder, label_valid_within_m=50.0):
    """Metrics for a MatchResult. Returns only numbers - no reference arcs or geometry."""
    trip = result.trip
    route = load_reference_route(Path(trip_folder), len(trip.arcs))     # ground truth: server-side only
    labels = observation_labels(trip, route, valid_within_m=label_valid_within_m)
    canon, lengths = canonical_arc_ids(trip.arcs), trip.arcs.length_m.to_numpy()

    hmm_arcs = [a for seg in result.segments for a in seg]
    overlap = route_overlap(hmm_arcs, route, lengths, canon) if hmm_arcs else \
        dict(route_precision=float('nan'), route_recall=float('nan'), route_f1=float('nan'))
    clf_overlap = route_overlap(result.classifier_top1.arc_id, route, lengths, canon) \
        if len(result.classifier_top1) else dict(route_f1=float('nan'))
    del route                                                           # nothing route-derived leaves this function
    return dict(
        top1_accuracy=_top1(result.matched, labels),
        classifier_only_top1_accuracy=_top1(result.classifier_top1, labels),
        route_precision=overlap['route_precision'],
        route_recall=overlap['route_recall'],
        route_f1=overlap['route_f1'],
        classifier_only_route_f1=clf_overlap['route_f1'],
        route_continuity=_continuity(result.matched, trip),
        classifier_only_route_continuity=_continuity(result.classifier_top1, trip),
        scored_points=int(labels.label_valid.sum()),
        unscored_points=int((~labels.label_valid).sum()),
    )
