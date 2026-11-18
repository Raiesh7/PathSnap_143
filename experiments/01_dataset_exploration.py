"""Read-only audit of development data. Run from any directory with Python 3.

python experiments/01_dataset_exploration.py
Outputs are deliberately retained under results/audit, maps/audit and report.
No labels are generated and no raw observations are changed by this script.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
# Optional repository-local dependencies; installed packages take precedence.
sys.path.append(str(ROOT / '.audit_deps'))
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import Transformer
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def read_table(path, width):
    """Validate schema without dropping malformed rows or changing row-based IDs."""
    rows, errors = [], []
    if not path.exists():
        return np.empty((0, width)), ['missing file']
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        try:
            if len(fields) != width:
                raise ValueError('wrong column count')
            rows.append([float(x) for x in fields])
        except ValueError:
            errors.append(f'line {line_no}: malformed row')
    if errors:
        # Dropping rows would silently renumber nodes/arcs: fail this trip instead.
        raise ValueError(f'{path}: {errors[:5]}')
    return np.asarray(rows, dtype=float).reshape(-1, width), errors


def nearest_route_distance(points, starts, ends):
    """Exact point-to-segment minima in projected metres, bounded-memory batches."""
    if not len(starts):
        return np.full(len(points), np.nan)
    vector = ends - starts
    norm = (vector * vector).sum(axis=1)
    result = []
    for offset in range(0, len(points), 128):
        delta = points[offset:offset + 128, None, :] - starts[None, :, :]
        fraction = np.clip((delta * vector).sum(axis=2) / np.maximum(norm, 1e-12), 0, 1)
        residual = delta - fraction[:, :, None] * vector
        result.extend(np.sqrt((residual * residual).sum(axis=2).min(axis=1)))
    return np.asarray(result)


def load_trip(folder):
    return {ext: read_table(folder / f'{folder.name}.{ext}', width)[0]
            for ext, width in [('track', 3), ('nodes', 2), ('arcs', 2), ('route', 1)]}


def audit_trip(folder, partition, graph_cache):
    data = load_trip(folder)
    track, nodes, arcs_raw, route_raw = [data[k] for k in ('track', 'nodes', 'arcs', 'route')]
    row = dict(partition=partition, trip_id=folder.name,
               gps_points=len(track), nodes=len(nodes), arcs=len(arcs_raw), route_entries=len(route_raw))
    for ext in data:
        row[f'{ext}_bytes'] = (folder / f'{folder.name}.{ext}').stat().st_size
        row[f'{ext}_nonfinite_values'] = int((~np.isfinite(data[ext])).sum())
    valid_gps = np.isfinite(track).all(axis=1) & (abs(track[:, 0]) <= 180) & (abs(track[:, 1]) <= 90)
    valid_nodes = np.isfinite(nodes).all(axis=1) & (abs(nodes[:, 0]) <= 180) & (abs(nodes[:, 1]) <= 90)
    row['invalid_gps_rows'] = int((~valid_gps).sum())
    row['invalid_node_rows'] = int((~valid_nodes).sum())
    row['duplicate_track_rows'] = int(len(track) - len(np.unique(track, axis=0)))
    row['consecutive_stationary_pairs'] = int(np.all(np.diff(track[:, :2], axis=0) == 0, axis=1).sum())
    dt = np.diff(track[:, 2])
    for name, value in [('nonpositive_time_steps', (dt <= 0).sum()), ('one_second_steps', (dt == 1).sum()),
                        ('gaps_over_1s', (dt > 1).sum()), ('gaps_over_5s', (dt > 5).sum())]:
        row[name] = int(value)
    row['max_gap_s'] = float(dt.max()) if len(dt) else 0
    row['median_step_s'] = float(np.median(dt)) if len(dt) else None
    row['duration_s'] = float(track[-1, 2] - track[0, 2]) if len(track) else 0
    for i, label in enumerate(['longitude', 'latitude']):
        row[f'min_{label}'] = float(track[valid_gps, i].min()) if valid_gps.any() else None
        row[f'max_{label}'] = float(track[valid_gps, i].max()) if valid_gps.any() else None
    arc_valid = np.isfinite(arcs_raw).all(axis=1) & (arcs_raw == np.floor(arcs_raw)).all(axis=1) & (arcs_raw >= 0).all(axis=1) & (arcs_raw < len(nodes)).all(axis=1)
    route_flat = route_raw.ravel()
    route_valid = np.isfinite(route_flat) & (route_flat == np.floor(route_flat)) & (route_flat >= 0) & (route_flat < len(arcs_raw))
    row['invalid_arc_rows'] = int((~arc_valid).sum())
    row['invalid_route_ids'] = int((~route_valid).sum())
    if not arc_valid.all() or not route_valid.all() or not valid_nodes.all() or not valid_gps.all():
        row['geometry_audit_skipped'] = True
        return row, data
    arcs, route = arcs_raw.astype(np.int64), route_flat.astype(np.int64)
    row['self_loop_arcs'] = int((arcs[:, 0] == arcs[:, 1]).sum())
    row['duplicate_directed_arcs'] = int(len(arcs) - len(np.unique(arcs, axis=0)))
    digest = hashlib.sha256(nodes.tobytes() + arcs.tobytes()).hexdigest()
    row['network_sha256'] = digest
    if digest not in graph_cache:
        graph = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
        weak_n, weak_labels = connected_components(graph, directed=True, connection='weak')
        strong_n = connected_components(graph, directed=True, connection='strong', return_labels=False)
        degree = np.bincount(arcs.ravel(), minlength=len(nodes))
        pairs = set(map(tuple, arcs.tolist()))
        graph_cache[digest] = dict(weak_components=int(weak_n), strong_components=int(strong_n),
            isolated_nodes=int((degree == 0).sum()), largest_weak_component_nodes=int(np.bincount(weak_labels).max()),
            arcs_with_reverse=int(sum((b, a) in pairs for a, b in pairs)))
    row.update(graph_cache[digest])
    # Per-trip local UTM zone, chosen from the GPS extent. Always longitude first.
    lon, lat = np.median(track[:, :2], axis=0)
    epsg = (32600 if lat >= 0 else 32700) + min(60, max(1, int((lon + 180) // 6) + 1))
    row['projection_epsg'] = epsg
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    gps_xy = np.column_stack(transformer.transform(track[:, 0], track[:, 1]))
    node_xy = np.column_stack(transformer.transform(nodes[:, 0], nodes[:, 1]))
    step_m = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
    speed = np.divide(step_m, dt, out=np.full(len(dt), np.nan), where=dt > 0)
    row['gps_polyline_length_m'] = float(step_m.sum())
    row['max_implied_speed_mps'] = float(np.nanmax(speed)) if len(speed) else 0
    row['steps_over_55mps'] = int((speed > 55).sum())
    lengths = np.linalg.norm(node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]], axis=1)
    row['zero_length_arcs'] = int((lengths < 1e-6).sum())
    route_arcs = arcs[route]
    row['reference_length_m'] = float(lengths[route].sum())
    row['route_unique_arcs'] = int(len(np.unique(route)))
    row['route_consecutive_repeats'] = int((np.diff(route) == 0).sum())
    row['route_disconnected_pairs'] = int((route_arcs[:-1, 1] != route_arcs[1:, 0]).sum())
    distances = nearest_route_distance(gps_xy, node_xy[route_arcs[:, 0]], node_xy[route_arcs[:, 1]])
    for q in [50, 95, 100]:
        row[f'gps_to_reference_p{q}_m'] = float(np.nanpercentile(distances, q)) if len(route) else None
    row['gps_over_50m_from_reference'] = int((distances > 50).sum())
    row['gps_over_100m_from_reference'] = int((distances > 100).sum())
    data.update(gps_xy=gps_xy, node_xy=node_xy, route_arcs=route_arcs, distances=distances)
    return row, data


def make_map(folder, output, title):
    d = load_trip(folder)
    gps, nodes = d['track'], d['nodes']
    arcs, route = d['arcs'].astype(int), d['route'].astype(int).ravel()
    # A local network window avoids embedding the entire regional graph.
    lo, hi = gps[:, :2].min(axis=0) - .003, gps[:, :2].max(axis=0) + .003
    a, b = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    mask = (np.maximum(a, b) >= lo).all(axis=1) & (np.minimum(a, b) <= hi).all(axis=1)
    m = folium.Map(location=[float(gps[:, 1].mean()), float(gps[:, 0].mean())], tiles='OpenStreetMap')
    network = folium.FeatureGroup(name='Road network (local window)', show=True)
    features = [dict(type='Feature', properties={}, geometry=dict(type='LineString', coordinates=[x.tolist(), y.tolist()])) for x, y in zip(a[mask], b[mask])]
    folium.GeoJson(dict(type='FeatureCollection', features=features), style_function=lambda _: dict(color='#777777', weight=1, opacity=.45)).add_to(network)
    network.add_to(m)
    reference = folium.FeatureGroup(name='Reference route (development labels)', show=True)
    for arc_id in route:
        folium.PolyLine(nodes[arcs[arc_id]][:, ::-1].tolist(), color='#16a34a', weight=5, opacity=.8).add_to(reference)
    reference.add_to(m)
    observations = folium.FeatureGroup(name='Raw GPS observations', show=True)
    for i, (lon, lat, timestamp) in enumerate(gps):
        folium.CircleMarker([lat, lon], radius=2, color='#2563eb', fill=True, fill_opacity=.8,
                            tooltip=f'GPS row {i}; t={timestamp:g}s').add_to(observations)
    observations.add_to(m)
    gaps = folium.FeatureGroup(name='Sampling gaps >5 seconds', show=True)
    for i in np.flatnonzero(np.diff(gps[:, 2]) > 5):
        folium.PolyLine(gps[i:i+2, 1::-1].tolist(), color='#ef4444', weight=3, dash_array='6,6',
                        tooltip=f'{gps[i+1, 2]-gps[i, 2]:g}s gap; straight connector is not a known path').add_to(gaps)
    gaps.add_to(m)
    folium.Marker(gps[0, 1::-1].tolist(), tooltip='First observation').add_to(m)
    folium.Marker(gps[-1, 1::-1].tolist(), tooltip='Last observation', icon=folium.Icon(color='red')).add_to(m)
    m.get_root().html.add_child(folium.Element(f'<div style="position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:10px;border:1px solid #888"><b>{title}</b><br>Blue: raw GPS | Green: reference | Grey: roads | Red dashed: time gap<br>Development EDA only; no predicted route has been generated.</div>'))
    folium.LayerControl().add_to(m)
    m.fit_bounds(gps[:, 1::-1].tolist())
    m.save(str(output))
    # Folium templates emit whitespace-only lines; keep generated HTML diff-clean.
    output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data' / 'Student folder')
    args = parser.parse_args()
    out, maps = ROOT / 'results/audit', ROOT / 'maps/audit'
    out.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    records, cache, failures, events = [], {}, [], []
    for partition in ['train', 'student_test']:
        for folder in sorted((args.data_root / partition).iterdir()):
            if not folder.is_dir():
                continue
            try:
                row, data = audit_trip(folder, partition, cache)
                records.append(row)
                track = data['track']
                for i in np.flatnonzero(np.diff(track[:, 2]) > 5):
                    events.append(dict(partition=partition, trip_id=folder.name, kind='gap_over_5s',
                                       from_row=int(i), to_row=int(i+1), value=float(track[i+1, 2]-track[i, 2])))
                if 'route_arcs' in data:
                    r = data['route_arcs']
                    for i in np.flatnonzero(r[:-1, 1] != r[1:, 0]):
                        events.append(dict(partition=partition, trip_id=folder.name, kind='reference_endpoint_break',
                            from_row=int(i), to_row=int(i+1), from_arc=int(data['route'][i, 0]), to_arc=int(data['route'][i+1, 0]),
                            from_end_node=int(r[i, 1]), to_start_node=int(r[i+1, 0])))
                print(f'{partition}/{folder.name}: {row["gps_points"]} GPS points', flush=True)
            except Exception as exc:
                failures.append(dict(partition=partition, trip_id=folder.name, error=str(exc)))
    df = pd.DataFrame(records)
    df.to_csv(out / 'trip_inventory.csv', index=False)
    pd.DataFrame(events).to_csv(out / 'events.csv', index=False)
    versions = {name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','pyproj','folium','matplotlib']}
    (out / 'environment.json').write_text(json.dumps(dict(python=sys.version, packages=versions), indent=2))
    summary = dict(audited_trips=len(df), failures=failures, partitions={}, unique_networks=len(cache))
    total_fields = ['gps_points', 'nodes', 'arcs', 'route_entries', 'invalid_gps_rows', 'invalid_node_rows',
        'invalid_arc_rows', 'invalid_route_ids', 'duplicate_track_rows', 'nonpositive_time_steps', 'one_second_steps',
        'gaps_over_1s', 'gaps_over_5s', 'steps_over_55mps', 'route_disconnected_pairs', 'gps_over_50m_from_reference']
    for partition, group in df.groupby('partition'):
        summary['partitions'][partition] = dict(trips=len(group), **{key: int(group[key].sum()) for key in total_fields},
            max_gap_s=float(group.max_gap_s.max()), gps_points_min=int(group.gps_points.min()),
            gps_points_median=float(group.gps_points.median()), gps_points_max=int(group.gps_points.max()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Save anomaly flags independently of explanatory prose.
    anomaly_cols = ['invalid_gps_rows', 'invalid_node_rows', 'invalid_arc_rows', 'invalid_route_ids',
                    'duplicate_track_rows', 'nonpositive_time_steps', 'gaps_over_5s', 'steps_over_55mps',
                    'route_disconnected_pairs', 'gps_over_50m_from_reference']
    df.loc[(df[anomaly_cols] > 0).any(axis=1), ['partition', 'trip_id'] + anomaly_cols].to_csv(out / 'anomalies.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for partition, group in df.groupby('partition'):
        axes[0, 0].hist(group.gps_points, bins=15, alpha=.6, label=partition)
        axes[0, 1].hist(group.max_gap_s, bins=np.geomspace(1, df.max_gap_s.max()+1, 20), alpha=.6, label=partition)
        axes[1, 0].scatter(group.gps_points, group.route_entries, label=partition, alpha=.7)
        axes[1, 1].scatter(group.max_gap_s, group.gps_to_reference_p95_m, label=partition, alpha=.7)
    for ax, title, xlabel, ylabel in zip(axes.ravel(), ['Trip sizes', 'Largest sampling gap per trip', 'GPS rows and reference rows', 'Gaps and reference proximity'],
            ['GPS observations', 'Maximum gap (s)', 'GPS observations', 'Maximum gap (s)'],
            ['Trips', 'Trips', 'Reference route entries', '95th percentile GPS-to-reference distance (m)']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    axes[0, 1].set_xscale('log')
    axes[1, 1].set_xscale('log')
    fig.tight_layout()
    fig.savefig(out / 'dataset_overview.png', dpi=160)
    plt.close(fig)
    train = df[df.partition == 'train']
    selected = list(dict.fromkeys([train.iloc[0].trip_id, train.loc[train.max_gap_s.idxmax()].trip_id,
                                  train.loc[train.gps_over_50m_from_reference.idxmax()].trip_id]))
    for trip_id in selected:
        folder = args.data_root / 'train' / trip_id
        map_path = maps / f'train_{trip_id}.html'
        make_map(folder, map_path, f'Training trip {trip_id}')
        _, d = audit_trip(folder, 'train', cache)
        render_inspection(d, trip_id, out)
    write_report(df, summary, selected)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit('Some trips failed audit. Inspect summary.json before continuing.')


def render_inspection(d, trip_id, out):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        origin = d['gps_xy'].mean(axis=0)
        xy, nxy = d['gps_xy'] - origin, d['node_xy'] - origin
        arcs = d['arcs'].astype(int)
        segs = nxy[arcs]
        lo, hi = xy.min(axis=0)-300, xy.max(axis=0)+300
        mask = (segs.max(axis=1) >= lo).all(axis=1) & (segs.min(axis=1) <= hi).all(axis=1)
        axes[0].add_collection(LineCollection(segs[mask], colors='#bbbbbb', linewidths=.4))
        axes[0].add_collection(LineCollection(nxy[d['route_arcs']], colors='#16a34a', linewidths=3, label='Reference route'))
        axes[0].scatter(*xy.T, s=1, color='#2563eb', alpha=.6, label='Raw GPS', zorder=3)
        axes[0].set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), xlabel='Easting relative to trip centre (m)', ylabel='Northing relative to trip centre (m)', title=f'Training trip {trip_id}')
        axes[0].set_aspect('equal')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.02, .005), ncol=2)
        worst = int(np.nanargmax(d['distances']))
        centre = xy[worst]
        radius = max(150, min(800, float(d['distances'][worst])*1.5))
        axes[1].add_collection(LineCollection(segs[mask], colors='#bbbbbb', linewidths=.7))
        axes[1].add_collection(LineCollection(nxy[d['route_arcs']], colors='#16a34a', linewidths=3))
        axes[1].scatter(*xy.T, s=5, color='#2563eb', zorder=3)
        axes[1].scatter(*centre, s=70, facecolors='none', edgecolors='#ef4444', zorder=4)
        axes[1].set(xlim=(centre[0]-radius,centre[0]+radius), ylim=(centre[1]-radius,centre[1]+radius),
                    xlabel='Relative easting (m)', ylabel='Relative northing (m)', title=f'Zoom near largest deviation (row {worst})')
        axes[1].set_aspect('equal')
        axes[2].scatter(np.arange(len(xy)), d['distances'], s=2)
        for i in np.flatnonzero(np.diff(d['track'][:,2])>5):
            axes[2].axvline(i+.5, color='#ef4444', alpha=.3, linewidth=.5)
        axes[2].set(xlabel='GPS observation row (red lines: gaps >5s)', ylabel='Nearest reference-segment distance (m)', title='Reference proximity; not prediction accuracy')
        fig.tight_layout(rect=(0, .06, 1, 1))
        fig.savefig(out / f'train_{trip_id}_inspection.png', dpi=170)
        plt.close(fig)


def write_report(df, summary, selected):
    lines = ['# Dataset audit and exploratory visual inspection', '',
        'Scope: development data only (`train` and `student_test`). No evaluator-testing data was inspected. '
        'This is a reproducible EDA record for the technical report, not a claim of model performance.', '',
        '## Reproduce', '', '```powershell', 'python -m pip install -r requirements.txt',
        'python experiments/01_dataset_exploration.py', '```', '',
        'The audit is read-only: no GPS samples, graph rows or reference labels are modified. '
        'Malformed rows fail the trip rather than silently renumbering row-based IDs. Expected file widths, '
        'finite values, coordinate ranges and integer ID bounds are checked.', '', '## Partition inventory', '',
        '| Partition | Trips | GPS observations | Reference entries | GPS rows per trip (min / median / max) |',
        '|---|---:|---:|---:|---|']
    for name, s in summary['partitions'].items():
        lines.append(f'| {name} | {s["trips"]} | {s["gps_points"]:,} | {s["route_entries"]:,} | {s["gps_points_min"]} / {s["gps_points_median"]:g} / {s["gps_points_max"]} |')
    lines += ['', 'Node and arc counts in the CSV are per trip; their sums must not be interpreted as unique physical roads. '
              f'Exact coordinate/connectivity hashing found {summary["unique_networks"]} distinct supplied graph representations.', '',
              '## Measured quality findings', '']
    for name, s in summary['partitions'].items():
        group = df[df.partition == name]
        steps = int((group.gps_points - 1).sum())
        lines += [f'### {name}', '',
            f'- Invalid GPS/node/arc/route rows: {s["invalid_gps_rows"]} / {s["invalid_node_rows"]} / {s["invalid_arc_rows"]} / {s["invalid_route_ids"]}.',
            f'- Exact duplicate GPS rows: {s["duplicate_track_rows"]}; nonpositive timestamp steps: {s["nonpositive_time_steps"]}.',
            f'- Exactly 1-second steps: {s["one_second_steps"]:,}/{steps:,} ({100*s["one_second_steps"]/steps:.2f}%). '
            f'Gaps above 1 second: {s["gaps_over_1s"]:,}; above 5 seconds: {s["gaps_over_5s"]:,}; maximum gap: {s["max_gap_s"]:g} seconds.',
            f'- Steps with implied speed above 55 m/s (198 km/h): {s["steps_over_55mps"]}. This is a diagnostic flag, not an automatic removal rule.',
            f'- Reference consecutive pairs that do not share the directed endpoint: {s["route_disconnected_pairs"]}.',
            f'- GPS observations farther than 50 m from every reference segment: {s["gps_over_50m_from_reference"]:,}. '
            'Distances use the nearest segment anywhere on the reference route, without temporal alignment.',
            f'- Weak components per supplied graph: {int(group.weak_components.min())}–{int(group.weak_components.max())}; '
            f'strong components: {int(group.strong_components.min())}–{int(group.strong_components.max())}; '
            f'isolated nodes: {int(group.isolated_nodes.min())}–{int(group.isolated_nodes.max())}.', '']
    lines += ['## Figures and maps', '', '![Dataset overview](../results/audit/dataset_overview.png)', '']
    for trip_id in selected:
        lines += [f'### Training trip {trip_id}', '',
            f'![Trip {trip_id} inspection](../results/audit/train_{trip_id}_inspection.png)', '',
            f'[Interactive Folium map](../maps/audit/train_{trip_id}.html)', '']
    lines += ['Selection is deterministic: the first training trip, the training trip with the largest sampling gap, '
              'and the training trip with the most observations above 50 m from the reference '
              '(deduplicated if identical). Maps contain raw GPS, local road geometry, reference route, start/end markers '
              'and gaps above 5 seconds. Dashed gap connectors do not establish the actual travelled path. '
              'No predicted-route layer is available at this stage. Folium maps require internet access for CDN scripts and base tiles; '
              'the PNG figures work offline.', '',
        '## Preprocessing and feature decisions', '',
        '- Preserve original row indices: node IDs and arc IDs are zero-based row positions within each trip.',
        '- Treat IDs as local to each trip/network. An arc ID is not a globally reusable road class. The supplied trips span multiple geographical regions.',
        '- Preserve raw observations. Flag duplicates, jumps and gaps before deciding on any cleaning policy.',
        '- Use per-trip WGS84 UTM coordinates for metric geometry, with longitude-first transformations. EPSG codes are saved in the inventory. '
        'Check that a trip is local to its chosen zone before reusing this projection for future datasets.',
        '- Keep arc direction as supplied. Weak connectivity describes physical reachability ignoring direction; strong connectivity describes directed reachability.',
        '- Reference distance is an EDA diagnostic only. It must never be an inference feature, because reference routes are unavailable to the prediction pipeline.',
        '- Candidate generation, speed/heading features, scaling, imputation and pointwise label generation are not implemented by this audit.',
        '- Future cross-validation must group by trip. Any learned preprocessing must be fitted within each training fold. '
        'Student-testing data is permitted for development comparison; evaluator-testing data remains reserved.', '',
        '## Label alignment limitation', '',
        'The reference file records an ordered arc traversal sequence, while the track records timed GPS observations. '
        'These are different units and cannot be joined by row number. A vehicle may produce many GPS observations on one arc, '
        'and may traverse multiple arcs between observations. Repeated route arcs also require preserving traversal order. '
        'A separate, documented, order-aware reference-alignment method is needed before supervised pointwise training and evaluation. '
        'Any derived labels are reference-derived estimates, not independently measured pointwise truth. '
        'Nearest-reference distances do not solve that alignment and are not classification accuracy.', '',
        '## Saved evidence', '',
        '- `results/audit/trip_inventory.csv`: per-trip schema, counts, timestamps, geometry, graph and reference checks.',
        '- `results/audit/anomalies.csv`: trips with the documented diagnostic flags.',
        '- `results/audit/events.csv`: GPS gap row pairs and broken reference endpoint pairs, with original zero-based row indices.',
        '- `results/audit/environment.json`: Python and package versions used for this run.',
        '- `results/audit/summary.json`: partition totals and parsing failures.',
        '- `results/audit/*.png`: reproducible static figures.',
        '- `maps/audit/*.html`: interactive Folium development maps.', '',
        'Source requirements: Dataset Exploration.pdf (sections 3–9), Deliverables.pdf (dataset/EDA section and Folium requirements), '
        'and Models and Training Guidelines.pdf (group-aware validation and observation-level evaluation). '
        'Original PDFs remain in the sibling PathSnap_student_materials folder.', '']
    (ROOT / 'report/dataset_audit.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

