"""Behavioural tests for candidate generation, labelling and features.

Run: python -m unittest discover -s tests -v
Real-data tests are skipped if data/processed has not been built yet.
"""
import sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidates import CandidateConfig, generate_candidates, point_segment_distance  # noqa: E402
from src.features import build_features, FEATURE_COLUMNS, angle_difference              # noqa: E402

CACHE = ROOT / 'data' / 'processed' / 'student_test'


class GeometryTests(unittest.TestCase):
    def test_point_segment_distance(self):
        d, t = point_segment_distance(np.array([5.0, -3.0, 14.0]), np.array([4.0, 0.0, 0.0]),
                                      np.zeros(3), np.zeros(3), np.full(3, 10.0), np.zeros(3))
        np.testing.assert_allclose(d, [4.0, 3.0, 4.0]); np.testing.assert_allclose(t, [0.5, 0.0, 1.0])

    def test_angle_difference_wraps(self):
        np.testing.assert_allclose(angle_difference(np.array([350.0, 10.0, 90.0]), np.array([10.0, 350.0, 270.0])),
                                   [20.0, 20.0, 180.0])


@unittest.skipUnless(CACHE.is_dir(), 'build data/processed first')
class RealTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.preprocessing import load_preprocessed_trip
        cls.trip = load_preprocessed_trip(sorted(p for p in CACHE.iterdir() if p.is_dir())[0])
        cls.cfg = CandidateConfig()
        cls.cand = generate_candidates(cls.trip, cls.cfg)

    def test_candidate_limits(self):
        self.assertLessEqual(self.cand.groupby('observation_id').size().max(), self.cfg.max_candidates)
        self.assertLessEqual(self.cand.dist_m.max(), self.cfg.radius_m + 1e-9)
        self.assertFalse(self.cand.duplicated(['observation_id', 'arc_id']).any())

    def test_only_usable_arcs(self):
        arcs = self.trip.arcs.iloc[self.cand.arc_id]
        self.assertTrue(arcs.candidate_geometry_valid.all())
        self.assertFalse(arcs.duplicate_connection.any())

    def test_features_do_not_need_reference(self):
        f = build_features(self.trip, self.cand)
        self.assertEqual(len(f), len(self.cand))
        self.assertTrue(set(FEATURE_COLUMNS) <= set(f.columns))
        self.assertFalse(any('label' in c for c in f.columns))

    def test_one_positive_per_observation(self):
        from src.data_loader import load_reference_route
        from src.labels import observation_labels, attach_labels
        raw = ROOT / 'data' / 'Student folder' / 'student_test' / self.trip.trip_id
        labels = observation_labels(self.trip, load_reference_route(raw, len(self.trip.arcs)))
        self.assertTrue(np.all(np.diff(labels.route_position) >= 0))   # monotonic alignment
        cand, _ = attach_labels(self.cand, labels)
        self.assertLessEqual(cand.groupby('observation_id').label.sum().max(), 1)


if __name__ == '__main__':
    unittest.main()
