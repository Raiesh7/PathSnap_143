"""HMM + Viterbi route reconstruction (Newson & Krumm style), reference-free.

Hidden states  : candidate arcs of each observation (from src/candidates.py).
Emission       : the classifier probability of the candidate, log(max(p, floor)).
Transition     : how well the driving distance along the road network between two consecutive
                 candidates matches the straight-line distance between the two GPS fixes:
                     log P = -|route_distance - gps_distance| / beta
                 route_distance uses the arcs' projection points and directed shortest paths on
                 trip.graph, so one-way rules and connectivity are respected. Unreachable pairs
                 (no path within search_limit_m) get probability 0.
Decoding       : Viterbi per sequence (sequences are split at time gaps by preprocessing).
                 If no transition is possible at some step, decoding restarts there (an "HMM break")
                 instead of forcing an impossible route.

Nothing here reads reference routes or labels, so the API can use it unchanged.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

NEG_INF = -np.inf


@dataclass(frozen=True)
class HMMConfig:
    beta_m: float = 3.0               # transition scale; 3 m chosen on TRAIN out-of-fold predictions (04_evaluate_hmm.py)
    prob_floor: float = 1e-4          # probabilities below this are treated as this (avoids log(0))
    search_limit_m: float = 1000.0    # max network distance searched between consecutive fixes
    same_arc_backward_tol_m: float = 15.0  # GPS jitter allowance for small backward moves on one arc

    def to_dict(self):
        return asdict(self)


class ShortestPaths:
    """Cached single-source Dijkstra on the directed node graph (limited radius)."""

    def __init__(self, graph, limit_m):
        self.graph, self.limit, self.cache = graph, limit_m, {}

    def _run(self, source):
        if source not in self.cache:
            dist, pred = dijkstra(self.graph, directed=True, indices=source, limit=self.limit,
                                  return_predecessors=True)
            reach = np.flatnonzero(np.isfinite(dist))
            self.cache[source] = (dict(zip(reach.tolist(), dist[reach].tolist())),
                                  {int(n): int(pred[n]) for n in reach})
        return self.cache[source]

    def distance(self, source, target):
        return self._run(int(source))[0].get(int(target), np.inf)

    def node_path(self, source, target):
        """Node sequence source -> target (inclusive), or None if unreachable."""
        dist, pred = self._run(int(source))
        if int(target) not in dist:
            return None
        path, n = [int(target)], int(target)
        while n != int(source):
            n = pred[n]
            path.append(n)
        return path[::-1]


def _transition_matrix(prev, cur, gps_d, arcs_u, arcs_v, arcs_len, sp, cfg):
    """log transition probabilities, shape (len(prev), len(cur))."""
    out = np.full((len(prev), len(cur)), NEG_INF)
    for i, (a, pa) in enumerate(zip(prev.arc_id.to_numpy(), prev.proj_frac.to_numpy())):
        la, va = arcs_len[a], arcs_v[a]
        for j, (b, pb) in enumerate(zip(cur.arc_id.to_numpy(), cur.proj_frac.to_numpy())):
            lb = arcs_len[b]
            if a == b and (pb - pa) * la >= -cfg.same_arc_backward_tol_m:
                route = max(0.0, (pb - pa) * la)
            else:
                mid = 0.0 if va == arcs_u[b] else sp.distance(va, arcs_u[b])
                route = (1 - pa) * la + mid + pb * lb
            if np.isfinite(route) and route <= cfg.search_limit_m + la + lb:
                out[i, j] = -abs(route - gps_d) / cfg.beta_m
    return out


def viterbi_trip(trip, scored, cfg: HMMConfig = HMMConfig()):
    """Decode one trip.

    scored: candidate rows of this trip with observation_id, arc_id, proj_frac, prob
            (only valid observations that have candidates).
    Returns DataFrame[observation_id, arc_id, sequence_id, hmm_break] - one chosen arc per observation.
    """
    obs = trip.observations.set_index('observation_id')
    arcs = trip.arcs
    arcs_u, arcs_v = arcs.from_node_id.to_numpy(), arcs.to_node_id.to_numpy()
    arcs_len = arcs.length_m.to_numpy()
    sp = ShortestPaths(trip.graph, cfg.search_limit_m)

    scored = scored.sort_values(['observation_id', 'arc_id'])
    groups = {k: g for k, g in scored.groupby('observation_id', sort=True)}
    order = sorted(groups)
    seq = obs.loc[order, 'sequence_id'].to_numpy()
    xy = obs.loc[order, ['x_m', 'y_m']].to_numpy()

    chosen, breaks = [], []
    start = 0
    while start < len(order):
        # one sequence = consecutive observations with the same sequence_id
        end = start
        while end + 1 < len(order) and seq[end + 1] == seq[start]:
            end += 1
        ids = order[start:end + 1]
        states = [groups[o] for o in ids]
        emis = [np.log(np.maximum(s.prob.to_numpy(), cfg.prob_floor)) for s in states]
        score, back, restart = [emis[0]], [None], [True]
        for t in range(1, len(ids)):
            gps_d = float(np.hypot(*(xy[start + t] - xy[start + t - 1])))
            tr = _transition_matrix(states[t - 1], states[t], gps_d, arcs_u, arcs_v, arcs_len, sp, cfg)
            total = score[-1][:, None] + tr
            best_prev = total.argmax(axis=0)
            best = total[best_prev, np.arange(total.shape[1])]
            if not np.isfinite(best).any():          # nothing reachable -> restart decoding here
                score.append(emis[t]); back.append(None); restart.append(True)
            else:
                score.append(best + emis[t]); back.append(best_prev); restart.append(False)
        # backtrack, segment by segment
        pick = [0] * len(ids)
        pick[-1] = int(np.argmax(score[-1]))
        for t in range(len(ids) - 1, 0, -1):
            pick[t - 1] = int(np.argmax(score[t - 1])) if restart[t] else int(back[t][pick[t]])
        for t, o in enumerate(ids):
            chosen.append((o, int(states[t].arc_id.to_numpy()[pick[t]]), int(seq[start]), bool(restart[t] and t > 0)))
        start = end + 1
    return pd.DataFrame(chosen, columns=['observation_id', 'arc_id', 'sequence_id', 'hmm_break'])


def expand_route(trip, matched, cfg: HMMConfig = HMMConfig()):
    """Turn one-arc-per-observation into a continuous arc sequence by inserting the shortest
    connecting arcs between consecutive different arcs (same sequence). Returns a list of
    arc-id lists, one per sequence / HMM segment."""
    arcs = trip.arcs
    u, v = arcs.from_node_id.to_numpy(), arcs.to_node_id.to_numpy()
    pair_to_arc = (arcs[arcs.candidate_geometry_valid | (arcs.length_m >= 0)]
                   .sort_values('length_m').drop_duplicates(['from_node_id', 'to_node_id'])
                   .set_index(['from_node_id', 'to_node_id']).arc_id)
    sp = ShortestPaths(trip.graph, cfg.search_limit_m)
    segments, cur = [], []
    prev_seq = None
    for a, s, brk in matched[['arc_id', 'sequence_id', 'hmm_break']].itertuples(index=False):
        if prev_seq is None or s != prev_seq or brk:
            if cur:
                segments.append(cur)
            cur = [int(a)]
        elif a != cur[-1]:
            nodes = sp.node_path(v[cur[-1]], u[a]) if v[cur[-1]] != u[a] else [int(u[a])]
            if nodes is not None:
                for n1, n2 in zip(nodes[:-1], nodes[1:]):
                    cur.append(int(pair_to_arc.loc[(n1, n2)]))
            cur.append(int(a))
        prev_seq = s
    if cur:
        segments.append(cur)
    return segments
