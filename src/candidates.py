"""Generate road-segment candidates for GPS observations.

Reference-free: uses only projected observations and the arc table, so the exact same
code runs at training time and inside the API.

For every valid observation, candidates are the K nearest *usable* arcs whose
perpendicular (point-to-segment) distance is <= radius. Usable means non-zero length and
first occurrence of a directed node pair (later duplicate rows are geometrically identical).
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    max_candidates: int = 8
    sample_spacing_m: float = 10.0   # KD-tree samples along arcs; exact distance computed afterwards

    def to_dict(self):
        return asdict(self)


def point_segment_distance(px, py, x1, y1, x2, y2):
    """Euclidean distance from points to segments and projection fraction t in [0, 1]."""
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    safe = np.where(length2 > 0, length2, 1.0)
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / safe, 0.0, 1.0)
    t = np.where(length2 > 0, t, 0.0)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)), t


def arc_geometry(trip):
    """Arc table with projected endpoint coordinates (row order = arc_id)."""
    arcs = trip.arcs
    nx, ny = trip.nodes.x_m.to_numpy(), trip.nodes.y_m.to_numpy()
    u, v = arcs.from_node_id.to_numpy(), arcs.to_node_id.to_numpy()
    return arcs.assign(x1=nx[u], y1=ny[u], x2=nx[v], y2=ny[v])


def usable_arc_mask(arcs):
    return (arcs.candidate_geometry_valid & ~arcs.duplicate_connection).to_numpy()


def generate_candidates(trip, config: CandidateConfig = CandidateConfig()) -> pd.DataFrame:
    """Return one row per (observation_id, arc_id) candidate with distance, projection and rank."""
    obs = trip.observations[trip.observations.valid_observation]
    arcs = arc_geometry(trip)
    if obs.empty:
        return pd.DataFrame(columns=['observation_id', 'arc_id', 'dist_m', 'proj_frac', 'rank'])
    px, py = obs.x_m.to_numpy(), obs.y_m.to_numpy()

    # restrict to arcs whose bounding box touches the trajectory bounding box (+ margin)
    m = config.radius_m + config.sample_spacing_m
    keep = usable_arc_mask(arcs) \
        & (np.maximum(arcs.x1, arcs.x2) >= px.min() - m).to_numpy() & (np.minimum(arcs.x1, arcs.x2) <= px.max() + m).to_numpy() \
        & (np.maximum(arcs.y1, arcs.y2) >= py.min() - m).to_numpy() & (np.minimum(arcs.y1, arcs.y2) <= py.max() + m).to_numpy()
    sub = arcs[keep]
    if sub.empty:
        return pd.DataFrame(columns=['observation_id', 'arc_id', 'dist_m', 'proj_frac', 'rank'])

    n = np.maximum(1, np.ceil(sub.length_m.to_numpy() / config.sample_spacing_m).astype(int))
    rep = np.repeat(np.arange(len(sub)), n + 1)
    frac = np.concatenate([np.linspace(0.0, 1.0, k + 1) for k in n])
    x1, y1, x2, y2 = (sub[c].to_numpy()[rep] for c in ['x1', 'y1', 'x2', 'y2'])
    samples = np.column_stack([x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)])
    sample_arc = sub.arc_id.to_numpy()[rep]

    # any segment within radius has a sample within radius + spacing / 2
    hits = cKDTree(samples).query_ball_point(np.column_stack([px, py]), r=config.radius_m + config.sample_spacing_m / 2)
    counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if counts.sum() == 0:
        return pd.DataFrame(columns=['observation_id', 'arc_id', 'dist_m', 'proj_frac', 'rank'])
    local = np.repeat(np.arange(len(px)), counts)
    pairs = pd.DataFrame({'local': local, 'arc_id': sample_arc[np.concatenate(hits).astype(np.int64)]}).drop_duplicates()

    a = arcs.iloc[pairs.arc_id.to_numpy()]
    d, t = point_segment_distance(px[pairs.local], py[pairs.local], a.x1.to_numpy(), a.y1.to_numpy(),
                                  a.x2.to_numpy(), a.y2.to_numpy())
    pairs = pairs.assign(dist_m=d, proj_frac=t)
    pairs = pairs[pairs.dist_m <= config.radius_m].sort_values(['local', 'dist_m', 'arc_id'], kind='stable')
    pairs['rank'] = pairs.groupby('local').cumcount()
    pairs = pairs[pairs['rank'] < config.max_candidates]
    pairs.insert(0, 'observation_id', obs.observation_id.to_numpy()[pairs.local.to_numpy()])
    return pairs.drop(columns='local').reset_index(drop=True)
