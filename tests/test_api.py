"""API tests. Uses the STUDENT-TEST trips (never the evaluator trips) via PATHSNAP_TRIPS_DIR.

Run: python -m unittest tests.test_api -v
"""
import os, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDENT_TEST = ROOT / 'data' / 'Student folder' / 'student_test'
MODEL = ROOT / 'artifacts' / 'models' / 'lightgbm_tuned.joblib'


@unittest.skipUnless(STUDENT_TEST.is_dir() and MODEL.is_file(), 'needs student_test data and the tuned model')
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['PATHSNAP_TRIPS_DIR'] = str(STUDENT_TEST)
        from fastapi.testclient import TestClient
        from api.main import app
        cls.client = TestClient(app)

    def test_health_and_trips(self):
        h = self.client.get('/health').json()
        self.assertEqual(h['status'], 'ok'); self.assertEqual(h['features'], 21)
        t = self.client.get('/trips').json()
        self.assertEqual(t['count'], 10)
        self.assertNotIn('route', str(t).lower())

    def test_predict_response_has_no_ground_truth(self):
        r = self.client.post('/predict', json={'trip_id': '00000052'})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body), {'trip_id', 'counts', 'metrics', 'timings_s', 'model', 'gps_points', 'predicted_route'})
        for key in body['metrics']:                       # numbers only, no arcs or geometry
            self.assertIsInstance(body['metrics'][key], (int, float))
        self.assertGreater(len(body['predicted_route']), 0)
        self.assertEqual(len(body['predicted_route'][0][0]), 2)

    def test_predict_matches_hmm_experiment(self):
        """Live API result == results/hmm/per_trip.csv (same model, same HMM setting)."""
        import pandas as pd
        ref = pd.read_csv(ROOT / 'results' / 'hmm' / 'per_trip.csv', dtype={'trip_id': str}).set_index('trip_id')
        for tid in ['00000052', '00000065']:   # trips where every point has a trusted label
            m = self.client.post('/predict', json={'trip_id': tid}).json()['metrics']
            self.assertAlmostEqual(m['top1_accuracy'], ref.loc[tid, 'hmm_top1'], places=6)
            self.assertAlmostEqual(m['classifier_only_top1_accuracy'], ref.loc[tid, 'clf_top1'], places=6)

    def test_unknown_or_malicious_trip_rejected(self):
        for bad in ['99999999', '../train/00000001', '']:
            self.assertEqual(self.client.post('/predict', json={'trip_id': bad}).status_code, 404)


class PredictionPathTests(unittest.TestCase):
    def test_inference_module_never_touches_ground_truth(self):
        src = (ROOT / 'src' / 'inference.py').read_text()
        self.assertNotIn('load_reference_route', src)
        self.assertNotIn('labels', src.split('"""', 2)[2])     # no import of src/labels.py after the docstring
        self.assertNotIn("'.route'", src)


if __name__ == '__main__':
    unittest.main()
