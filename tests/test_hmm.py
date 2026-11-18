"""Behavioural tests for HMM/Viterbi decoding. Real-data tests skip if data/processed is missing."""
import sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import route_overlap  # noqa: E402

CACHE = ROOT / 'data' / 'processed' / 'student_test'


class RouteOverlapTests(unittest.TestCase):
    def test_length_weighted(self):
        r = route_overlap([1, 2, 3], [2, 3, 4], np.array([0, 10, 10, 10, 10.0]))
        self.assertAlmostEqual(r['route_precision'], 2 / 3); self.assertAlmostEqual(r['route_recall'], 2 / 3)

    def test_duplicates_are_canonicalised(self):
        canon = np.array([0, 1, 1, 3])                     # arc 2 duplicates arc 1
        r = route_overlap([2], [1], np.ones(4), canon)
        self.assertEqual(r['route_f1'], 1.0)


@unittest.skipUnless(CACHE.is_dir(), 'build data/processed first')
class ViterbiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.preprocessing import load_preprocessed_trip
        from src.candidates import generate_candidates
        from src.hmm import viterbi_trip, expand_route, HMMConfig
        cls.trip = load_preprocessed_trip(sorted(p for p in CACHE.iterdir() if p.is_dir())[0])
        cand = generate_candidates(cls.trip)
        cls.cand = cand.assign(prob=np.exp(-cand.dist_m / 10.0))   # label-free stand-in scores
        cls.cfg = HMMConfig()
        cls.matched = viterbi_trip(cls.trip, cls.cand, cls.cfg)
        cls.segments = expand_route(cls.trip, cls.matched, cls.cfg)

    def test_one_arc_per_observation_from_candidates(self):
        self.assertEqual(len(self.matched), self.cand.observation_id.nunique())
        pairs = set(zip(self.cand.observation_id, self.cand.arc_id))
        self.assertTrue(all((o, a) in pairs for o, a in zip(self.matched.observation_id, self.matched.arc_id)))

    def test_expanded_route_is_connected(self):
        u = self.trip.arcs.from_node_id.to_numpy(); v = self.trip.arcs.to_node_id.to_numpy()
        for seg in self.segments:
            for a, b in zip(seg[:-1], seg[1:]):
                self.assertEqual(v[a], u[b])


if __name__ == '__main__':
    unittest.main()
