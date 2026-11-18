"""Verify all local caches, audit parity and unchanged raw data; save evidence."""
from pathlib import Path
import sys
import json
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / '.audit_deps'))
import numpy as np
import pandas as pd
from src.preprocessing import load_preprocessed_trip


def main():
    output = ROOT / 'results/preprocessing'
    summary = json.loads((output/'summary.json').read_text())
    assert not summary['failures']
    for file, expected in summary['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == expected, file
    manifest = list(csv.DictReader((ROOT/'data/manifest.csv').open(newline='', encoding='utf-8')))
    assert len(manifest) == 360
    source = ROOT/'data/Student folder'
    hashes = {}
    for entry in manifest:
        path = source/entry['path']
        assert path.stat().st_size == int(entry['bytes']), str(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry['sha256'], str(path)
        hashes[entry['path']] = digest
    audit = pd.read_csv(ROOT/'results/audit/trip_inventory.csv', dtype={'trip_id':str})
    prepared = pd.read_csv(output/'trip_summary.csv', dtype={'trip_id':str})
    assert len(prepared) == 90
    totals = dict(observations=0, sequences=0, gaps=0, speed_flags=0)
    for row in prepared.to_dict('records'):
        part, trip_id = row['partition'], row['trip_id']
        trip = load_preprocessed_trip(ROOT/'data/processed'/part/trip_id)
        o, a, n = trip.observations, trip.arcs, trip.nodes
        reference = audit[(audit.partition == part) & (audit.trip_id == trip_id)].iloc[0]
        assert len(o) == reference.gps_points
        assert len(a) == reference.arcs
        assert len(n) == reference.nodes
        assert int(o.gap_before.sum()) == reference.gaps_over_5s
        assert int(o.speed_flag.sum()) == reference.steps_over_55mps
        assert trip.metadata['weak_components'] == reference.weak_components
        assert trip.metadata['strong_components'] == reference.strong_components
        assert int(a.zero_length.sum()) == reference.zero_length_arcs
        assert int(a.duplicate_connection.sum()) == reference.duplicate_directed_arcs
        assert o.loc[~o.motion_valid, ['speed_mps','step_distance_m','heading_grid_deg']].isna().all().all()
        assert int(o.sequence_start.sum()) == row['sequence_count']
        assert row['sequence_count'] == row['gap_count'] + 1  # This dataset has no invalid/time-reversed rows.
        assert o.valid_observation.all()
        assert not trip.metadata['reference_used']
        assert not any('label' in key or 'reference' in key or 'route' in key for key in o.columns)
        assert np.isfinite(n[['x_m','y_m']].to_numpy()).all()
        for extension, digest in trip.metadata['source_sha256'].items():
            assert digest == hashes[f'{part}/{trip_id}/{trip_id}.{extension}']
        totals['observations'] += len(o)
        totals['sequences'] += int(row['sequence_count'])
        totals['gaps'] += int(row['gap_count'])
        totals['speed_flags'] += int(row['speed_flags'])
    assert totals == dict(observations=227577, sequences=1350, gaps=1260, speed_flags=993)
    evidence = dict(passed=True, raw_files_hash_verified=len(manifest), caches_verified=len(prepared),
                    audit_parity=True, implementation_hashes_match=True, reference_features_absent=True,
                    boundary_motion_missing=True, totals=totals)
    (output/'validation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
