"""Development-only pointwise labels derived from the reference route.

NEVER imported by the API prediction path. The reference route is an ordered list of arcs,
not one label per GPS observation, so each valid observation is aligned to a route position
with a monotonic dynamic programme:

    choose j_0 <= j_1 <= ... <= j_{N-1}   minimising   sum_i min(d(p_i, route[j_i]), cap)

d is the point-to-segment distance. Monotonicity follows the direction of travel, so
loops, revisited roads and parallel streets are resolved by order rather than by distance only.

A label is trusted only if the observation lies within `valid_within_m` of its aligned arc.
Some supplied routes cover only part of the GPS track (e.g. training trip 00000054); those
observations get label_valid=False and are excluded from training and scoring.
"""
import numpy as np
import pandas as pd
from .candidates import arc_geometry, point_segment_distance


def align_observations_to_route(trip, route, cost_cap_m=200.0):
    obs = trip.observations[trip.observations.valid_observation]
    arcs = arc_geometry(trip)
    r = arcs.iloc[route]
    x1, y1, x2, y2 = (r[c].to_numpy() for c in ['x1', 'y1', 'x2', 'y2'])
    px, py = obs.x_m.to_numpy(), obs.y_m.to_numpy()
    n, m = len(px), len(route)
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    positions = np.arange(m)

    def cost(i):
        d, _ = point_segment_distance(px[i], py[i], x1, y1, x2, y2)
        return np.minimum(d, cost_cap_m)

    dp = cost(0)
    back[0] = positions
    for i in range(1, n):
        best = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp == best, positions, 0))
        dp = cost(i) + best
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    d, _ = point_segment_distance(px, py, x1[j], y1[j], x2[j], y2[j])
    return obs.observation_id.to_numpy(), j, d


def canonical_arc_ids(arcs):
    """Map duplicate directed node pairs to the first arc_id (the one candidates use)."""
    return arcs.groupby(['from_node_id', 'to_node_id']).arc_id.transform('min').to_numpy()


def observation_labels(trip, route, valid_within_m=50.0):
    obs_id, pos, dist = align_observations_to_route(trip, route)
    true_arc = canonical_arc_ids(trip.arcs)[route[pos]]
    return pd.DataFrame({'observation_id': obs_id, 'route_position': pos, 'true_arc': true_arc,
                         'align_dist_m': dist, 'label_valid': dist <= valid_within_m})


def attach_labels(candidates, labels):
    """Add `label` (1 = candidate is the aligned reference arc) and `label_valid` to candidates."""
    lab = labels.set_index('observation_id')
    true = lab.true_arc.reindex(candidates.observation_id).to_numpy()
    out = candidates.assign(label=(candidates.arc_id.to_numpy() == true).astype(np.int8),
                            label_valid=lab.label_valid.reindex(candidates.observation_id).fillna(False).to_numpy())
    hit = out.groupby('observation_id').label.max()
    labels = labels.assign(true_in_candidates=hit.reindex(labels.observation_id).fillna(0).astype(bool).to_numpy())
    return out, labels
