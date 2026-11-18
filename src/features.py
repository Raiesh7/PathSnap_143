"""Per-candidate features. Reference-free: uses only observations, arcs and the road graph.

Groups
------
geometry   : distance to arc, rank, distance relative to the best candidate, projection position
direction  : GPS heading vs arc heading (raw and smoothed over +/-2 observations)
motion     : speed, time step, speed flag
topology   : arc length, in/out degree at the arc's end nodes, two-way road indicator
context    : distance of the previous/next observation to the same arc; whether the arc is the
             same as / connected to the previous observation's nearest arc
             (the sequential HMM comes later - these only give the classifier local context)
"""
import numpy as np
import pandas as pd
from .candidates import arc_geometry, point_segment_distance

FEATURE_COLUMNS = [
    'dist_m', 'rank', 'n_candidates', 'dist_minus_best', 'dist_ratio_best', 'proj_frac', 'proj_edge_dist',
    'heading_diff', 'heading_diff_smooth', 'heading_available',
    'speed_mps', 'dt_s', 'speed_flag',
    'arc_length_m', 'from_in_degree', 'to_out_degree', 'two_way',
    'dist_prev_obs', 'dist_next_obs', 'same_as_prev_nearest', 'follows_prev_nearest',
]


def angle_difference(a, b):
    """Absolute difference between bearings in degrees, 0..180 (NaN if either is NaN)."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _smoothed_heading(obs, window=2):
    """Heading from observation i-window to i+window inside the same sequence (less GPS jitter)."""
    g = obs.groupby('sequence_id', sort=False)
    xs, ys = obs.x_m, obs.y_m
    xp = g.x_m.shift(window).fillna(g.x_m.shift(1)).fillna(xs)
    yp = g.y_m.shift(window).fillna(g.y_m.shift(1)).fillna(ys)
    xn = g.x_m.shift(-window).fillna(g.x_m.shift(-1)).fillna(xs)
    yn = g.y_m.shift(-window).fillna(g.y_m.shift(-1)).fillna(ys)
    dx, dy = (xn - xp).to_numpy(), (yn - yp).to_numpy()
    h = np.degrees(np.arctan2(dx, dy)) % 360.0
    h[np.hypot(dx, dy) < 3.0] = np.nan
    return h


def build_features(trip, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(**{c: pd.Series(dtype=float) for c in FEATURE_COLUMNS})
    obs = trip.observations[trip.observations.valid_observation].copy()
    obs['heading_smooth'] = _smoothed_heading(obs)
    # neighbours within the same sequence (gap-aware)
    g = obs.groupby('sequence_id', sort=False)
    obs['prev_id'] = g.observation_id.shift(1)
    obs['next_id'] = g.observation_id.shift(-1)
    obs = obs.set_index('observation_id')

    arcs = arc_geometry(trip)
    a = arcs.iloc[candidates.arc_id.to_numpy()]
    o = obs.loc[candidates.observation_id.to_numpy()]
    c = candidates.copy()

    # geometry
    best = c.groupby('observation_id').dist_m.transform('min')
    c['n_candidates'] = c.groupby('observation_id').arc_id.transform('size')
    c['dist_minus_best'] = c.dist_m - best
    c['dist_ratio_best'] = (c.dist_m + 1.0) / (best + 1.0)
    c['proj_edge_dist'] = np.minimum(c.proj_frac, 1 - c.proj_frac) * a.length_m.to_numpy()

    # direction
    arc_heading = a.heading_grid_deg.to_numpy()
    c['heading_diff'] = angle_difference(o.heading_grid_deg.to_numpy(), arc_heading)
    c['heading_diff_smooth'] = angle_difference(o.heading_smooth.to_numpy(), arc_heading)
    c['heading_available'] = np.isfinite(c.heading_diff_smooth).astype(np.int8)

    # motion
    c['speed_mps'] = o.speed_mps.to_numpy()
    c['dt_s'] = o.dt_s.to_numpy()
    c['speed_flag'] = o.speed_flag.to_numpy().astype(np.int8)

    # topology
    u, v = arcs.from_node_id.to_numpy(), arcs.to_node_id.to_numpy()
    n_nodes = len(trip.nodes)
    in_deg = np.bincount(v, minlength=n_nodes)
    out_deg = np.bincount(u, minlength=n_nodes)
    pairs = pd.MultiIndex.from_arrays([u, v])
    reverse = pd.MultiIndex.from_arrays([v, u]).isin(pairs)
    c['arc_length_m'] = a.length_m.to_numpy()
    c['from_in_degree'] = in_deg[a.from_node_id.to_numpy()]
    c['to_out_degree'] = out_deg[a.to_node_id.to_numpy()]
    c['two_way'] = reverse[c.arc_id.to_numpy()].astype(np.int8)

    # context: same arc evaluated at neighbouring observations
    ax1, ay1, ax2, ay2 = (a[k].to_numpy() for k in ['x1', 'y1', 'x2', 'y2'])
    for side in ['prev', 'next']:
        nid = o[f'{side}_id'].to_numpy()
        has = np.isfinite(nid)
        d = np.full(len(c), np.nan)
        if has.any():
            nb = obs.loc[nid[has].astype(np.int64)]
            d[has], _ = point_segment_distance(nb.x_m.to_numpy(), nb.y_m.to_numpy(), ax1[has], ay1[has], ax2[has], ay2[has])
        c[f'dist_{side}_obs'] = d

    # "Best guess" arc of each observation. ~70% of observations have a distance tie between the
    # two directions of a two-way road, so arcs within 1 m of the best distance are ordered by
    # smoothed heading agreement (direction of travel), then distance.
    key = c.assign(_far=(c.dist_minus_best > 1.0).astype(np.int8),
                   _hd=c.heading_diff_smooth.fillna(c.heading_diff).fillna(180.0))
    nearest = (key.sort_values(['observation_id', '_far', '_hd', 'dist_m'], kind='stable')
                  .drop_duplicates('observation_id').set_index('observation_id').arc_id)
    prev_near = nearest.reindex(o.prev_id.to_numpy()).to_numpy()   # NaN where no previous obs
    has_prev = np.isfinite(prev_near)
    same = np.zeros(len(c), np.int8); follows = np.zeros(len(c), np.int8)
    pn = prev_near[has_prev].astype(np.int64)
    same[has_prev] = (c.arc_id.to_numpy()[has_prev] == pn)
    ca = c.arc_id.to_numpy()[has_prev]
    # continues from the previous best arc, excluding the immediate U-turn onto its reverse arc
    follows[has_prev] = (u[ca] == v[pn]) & ~(v[ca] == u[pn])
    c['same_as_prev_nearest'] = same
    c['follows_prev_nearest'] = follows
    c['sequence_id'] = o.sequence_id.to_numpy()
    return c
