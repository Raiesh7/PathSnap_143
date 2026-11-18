"""Prepare reference-free development inputs; caches are optional and gitignored."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / '.audit_deps'))
import numpy as np
import pandas as pd
from src.config import DEFAULT_DATA_ROOT, PreprocessingConfig
from src.data_loader import iter_trip_folders, load_trip
from src.preprocessing import preprocess_trip


def save_cache(trip, destination):
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for label, table in [('observations', trip.observations), ('nodes', trip.nodes), ('arcs', trip.arcs)]:
        for column in table:
            arrays[f'{label}__{column}'] = table[column].to_numpy()
    arrays.update(graph_data=trip.graph.data, graph_indices=trip.graph.indices,
                  graph_indptr=trip.graph.indptr, graph_shape=np.asarray(trip.graph.shape))
    np.savez_compressed(destination / 'inputs.npz', **arrays)
    (destination / 'metadata.json').write_text(json.dumps(trip.metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/preprocessing')
    parser.add_argument('--cache-dir', type=Path, help='Optional derived cache, e.g. data/processed')
    parser.add_argument('--gap-seconds', type=float, default=5.0)
    args = parser.parse_args()
    config = PreprocessingConfig(gap_seconds=args.gap_seconds)
    rows, events, failures = [], [], []
    for partition, folder in iter_trip_folders(args.data_root):
        try:
            trip = preprocess_trip(load_trip(folder, partition), config)
            o = trip.observations
            assert np.array_equal(o.observation_id, np.arange(len(o)))
            assert o.loc[~o.motion_valid, ['speed_mps','heading_grid_deg','step_distance_m']].isna().all().all()
            assert np.array_equal(trip.arcs.arc_id, np.arange(len(trip.arcs)))
            row = {k: v for k, v in trip.metadata.items() if k not in {'config','source_sha256','projection_wkt'}}
            rows.append(row)
            selected = o[o.gap_before | o.speed_flag | ~o.valid_observation | o.nonpositive_time_before]
            for record in selected[['observation_id','dt_s','gap_before','speed_flag','valid_observation','nonpositive_time_before']].to_dict('records'):
                events.append(dict(partition=partition, trip_id=folder.name, **record))
            if args.cache_dir:
                save_cache(trip, args.cache_dir / partition / folder.name)
            print(f'{partition}/{folder.name}: {len(o)} observations; {trip.metadata["sequence_count"]} sequences', flush=True)
        except Exception as exc:
            failures.append(dict(partition=partition, trip_id=folder.name, error=str(exc)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / 'trip_summary.csv', index=False)
    pd.DataFrame(events).to_csv(args.output_dir / 'flagged_observations.csv', index=False)
    totals = {}
    for partition, group in (df.groupby('partition') if not df.empty else []):
        totals[partition] = dict(trips=len(group), **{k: int(group[k].sum()) for k in [
            'observation_count','invalid_observations','sequence_count','gap_count','speed_flags',
            'zero_length_arcs','duplicate_connections','nonpositive_time_steps','duplicate_observations']})
    summary = dict(schema_version=1, config=config.to_dict(), partitions=totals, failures=failures,
        reference_used=False, cache_written=args.cache_dir is not None,
        versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','pyproj']},
        implementation_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
            'src/config.py','src/data_loader.py','src/preprocessing.py','experiments/01b_preprocess_dataset.py']})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit('Preprocessing failures: inspect summary.json')


if __name__ == '__main__':
    main()
