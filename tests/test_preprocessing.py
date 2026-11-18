"""Behavioural checks for label leakage, row identity, gaps and graph semantics."""
from pathlib import Path
import tempfile
import unittest
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / '.audit_deps'))
import numpy as np
from scipy.sparse.csgraph import dijkstra
from src.config import PreprocessingConfig
from src.data_loader import RawTrip, load_trip, load_reference_route, iter_trip_folders
from src.preprocessing import preprocess_trip, load_preprocessed_trip


def example(track=None, nodes=None, arcs=None):
    return RawTrip('example', 'train',
        np.asarray(track if track is not None else [[8,49,0],[8.0001,49,1]], dtype=float),
        np.asarray(nodes if nodes is not None else [[8,49],[8.0001,49],[8.0002,49]], dtype=float),
        np.asarray(arcs if arcs is not None else [[0,1],[1,2]], dtype=np.int64), {})


class PreprocessingTests(unittest.TestCase):
    def test_metric_geometry_and_direction(self):
        p = preprocess_trip(example())
        self.assertTrue(6 < p.arcs.length_m.iloc[0] < 9)
        self.assertTrue(85 < p.observations.heading_grid_deg.iloc[1] < 95)
        self.assertTrue(np.isnan(p.observations.speed_mps.iloc[0]))
        self.assertTrue(np.isinf(dijkstra(p.graph, indices=2)[0]))

    def test_gap_and_invalid_rows_never_bridge_motion(self):
        raw = example(track=[[8,49,0],[8.0001,49,1],[8.0002,49,7],
                             [np.nan,49,8],[8.0003,49,9],[8.0004,49,10]])
        p = preprocess_trip(raw)
        np.testing.assert_array_equal(p.observations.sequence_id, [0,0,1,-1,2,2])
        self.assertTrue(p.observations.loc[[0,2,3,4], 'speed_mps'].isna().all())
        self.assertTrue(np.isfinite(p.observations.raw_implied_speed_mps.iloc[2]))
        np.testing.assert_array_equal(p.observations.observation_id, np.arange(6))

    def test_exact_boundary_and_nonpositive_time(self):
        p = preprocess_trip(example(track=[[8,49,0],[8.0001,49,5],[8.0002,49,5],[8.0003,49,4],[8.0004,49,5]]))
        np.testing.assert_array_equal(p.observations.sequence_id, [0,0,1,2,2])
        self.assertFalse(p.observations.gap_before.iloc[1])
        self.assertEqual(p.metadata['nonpositive_time_steps'], 2)

    def test_stationary_and_duplicates_retained_without_heading(self):
        p = preprocess_trip(example(track=[[8,49,0],[8,49,1],[8,49,1]]))
        self.assertEqual(len(p.observations), 3)
        self.assertEqual(p.observations.speed_mps.iloc[1], 0)
        self.assertTrue(p.observations.heading_grid_deg.isna().all())
        self.assertEqual(p.metadata['duplicate_observations'], 1)

    def test_speed_flag_does_not_delete_fast_travel(self):
        p = preprocess_trip(example(track=[[8,49,0],[8.01,49,1]]))
        self.assertTrue(p.observations.speed_flag.iloc[1])
        self.assertTrue(p.observations.motion_valid.iloc[1])
        self.assertEqual(p.metadata['sequence_count'], 1)

    def test_parallel_and_zero_length_arcs_keep_ids_and_shortest_cost(self):
        p = preprocess_trip(example(nodes=[[8,49],[8.0001,49],[8.0001,49],[8.0002,49]],
                                    arcs=[[0,1],[0,1],[1,2],[2,3]]))
        np.testing.assert_array_equal(p.arcs.arc_id, [0,1,2,3])
        self.assertEqual(p.metadata['duplicate_connections'], 1)
        self.assertEqual(p.metadata['zero_length_arcs'], 1)
        self.assertFalse(p.arcs.candidate_geometry_valid.iloc[2])
        self.assertAlmostEqual(p.graph[0,1], p.arcs.length_m.iloc[0])
        distance = dijkstra(p.graph, indices=0)[3]
        self.assertAlmostEqual(distance, p.arcs.length_m.iloc[[0,2,3]].sum())
        self.assertEqual(p.metadata['weak_components'], 1)

    def test_invalid_timestamp_restarts_next_valid_observation(self):
        p = preprocess_trip(example(track=[[8,49,0],[8,49,np.nan],[8,49,2]]))
        np.testing.assert_array_equal(p.observations.sequence_id, [0,-1,1])
        self.assertTrue(p.observations.speed_mps.isna().all())

    def test_all_invalid_observations_use_nodes_for_projection(self):
        p = preprocess_trip(example(track=[[np.nan,49,0],[8,100,1]]))
        self.assertEqual(p.metadata['valid_observations'], 0)
        self.assertEqual(p.metadata['sequence_count'], 0)

    def test_antimeridian_is_local(self):
        p = preprocess_trip(example(track=[[179.999,0,0],[-179.999,0,1]],
                                    nodes=[[179.999,0],[-179.999,0]], arcs=[[0,1]]))
        self.assertTrue(200 < p.arcs.length_m.iloc[0] < 250)

    def test_input_arrays_are_unchanged(self):
        raw = example()
        originals = [x.copy() for x in [raw.track,raw.nodes,raw.arcs]]
        preprocess_trip(raw)
        for a,b in zip(originals,[raw.track,raw.nodes,raw.arcs]):
            np.testing.assert_array_equal(a,b)

    def test_configuration_rejects_invalid_thresholds(self):
        for v in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                PreprocessingConfig(gap_seconds=v)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / 'trip'
        self.folder.mkdir()
        (self.folder/'trip.track').write_text('8 49 0\n8.0001 49 1\n')
        (self.folder/'trip.nodes').write_text('8 49\n8.0001 49\n')
        (self.folder/'trip.arcs').write_text('0 1\n')

    def tearDown(self):
        self.temp.cleanup()

    def test_preprocessing_needs_no_reference_and_ignores_poisoned_reference(self):
        first = preprocess_trip(load_trip(self.folder))
        (self.folder/'trip.route').write_text('THIS MUST NOT BE READ')
        second = preprocess_trip(load_trip(self.folder))
        self.assertFalse(second.metadata['reference_used'])
        self.assertEqual(first.metadata,second.metadata)
        self.assertTrue(first.observations.equals(second.observations))
        with self.assertRaises(ValueError):
            load_reference_route(self.folder,1)

    def test_malformed_blank_and_missing_files_fail(self):
        path = self.folder/'trip.arcs'
        for text in ['0 1\n\n','0 1 2\n','0 1.5\n','0 nan\n','0 2\n','']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_trip(self.folder)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_trip(self.folder)

    def test_invalid_nodes_rejected(self):
        (self.folder/'trip.nodes').write_text('8 100\n8 49\n')
        with self.assertRaises(ValueError):
            load_trip(self.folder)

    def test_evaluator_partition_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_trip_folders(Path(self.temp.name), ['evaluator_test']))

    def test_cache_roundtrip_without_pickle(self):
        spec = importlib.util.spec_from_file_location('batch', ROOT/'experiments/01b_preprocess_dataset.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trip = preprocess_trip(load_trip(self.folder))
        out = Path(self.temp.name)/'cache'
        module.save_cache(trip,out)
        restored = load_preprocessed_trip(out)
        self.assertTrue(restored.observations.equals(trip.observations))
        self.assertTrue(restored.arcs.equals(trip.arcs))
        self.assertEqual((restored.graph != trip.graph).nnz, 0)
        with np.load(out/'inputs.npz',allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['arcs__arc_id'],trip.arcs.arc_id)
            np.testing.assert_array_equal(arrays['observations__sequence_id'],trip.observations.sequence_id)
            self.assertTrue(all('route' not in name and 'label' not in name for name in arrays.files))


if __name__ == '__main__':
    unittest.main()
