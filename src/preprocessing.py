"""Reference-free, row-preserving preprocessing for training and inference.

NaN derived values indicate unavailable motion, not zero speed/heading.
No interpolation, reference-based correction, imputation or scaling is performed.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .config import PreprocessingConfig
from .data_loader import RawTrip


@dataclass
class PreprocessedTrip:
    trip_id: str
    partition: str
    observations: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    graph: csr_matrix
    metadata: dict


def load_preprocessed_trip(folder: Path) -> PreprocessedTrip:
    """Load the batch-produced cache without pickle or access to raw labels."""
    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1 or metadata.get('reference_used') is not False:
        raise ValueError(f'{folder}: unsupported or reference-contaminated cache')
    with np.load(folder / 'inputs.npz', allow_pickle=False) as arrays:
        tables = {}
        for name in ['observations', 'nodes', 'arcs']:
            prefix = name + '__'
            tables[name] = pd.DataFrame({key[len(prefix):]: arrays[key] for key in arrays.files if key.startswith(prefix)})
        shape = tuple(int(n) for n in arrays['graph_shape'])
        graph = csr_matrix((arrays['graph_data'], arrays['graph_indices'], arrays['graph_indptr']), shape=shape)
    for name, id_column, count_key in [('observations','observation_id','observation_count'),
                                      ('nodes','node_id','node_count'), ('arcs','arc_id','arc_count')]:
        if len(tables[name]) != metadata[count_key] or not np.array_equal(tables[name][id_column], np.arange(len(tables[name]))):
            raise ValueError(f'{folder}: cached {name} row identity mismatch')
    if graph.shape != (len(tables['nodes']), len(tables['nodes'])):
        raise ValueError(f'{folder}: cached graph shape mismatch')
    return PreprocessedTrip(metadata['trip_id'], metadata['partition'], tables['observations'],
                            tables['nodes'], tables['arcs'], graph, metadata)


def _projection(track, nodes, valid):
    coordinates = track[valid, :2] if valid.any() else nodes
    lon, lat = np.median(coordinates, axis=0)
    if np.ptp(coordinates[:, 0]) > 180:
        angles = np.radians(coordinates[:, 0])
        lon = float(np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())))
    all_coords = np.vstack([coordinates, nodes])
    wrapped_delta = (all_coords[:, 0] - lon + 180) % 360 - 180
    if -80 <= lat <= 84 and np.max(abs(wrapped_delta)) <= 6:
        zone = min(60, max(1, int((lon + 180) // 6) + 1))
        crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    else:
        crs = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    return crs, Transformer.from_crs(4326, crs, always_xy=True)


def preprocess_trip(raw: RawTrip, config=None) -> PreprocessedTrip:
    config = config or PreprocessingConfig()
    track, nodes, arcs = raw.track, raw.nodes, raw.arcs
    count = len(track)
    valid_coordinate = np.isfinite(track[:, :2]).all(axis=1) & (abs(track[:, 0]) <= 180) & (abs(track[:, 1]) <= 90)
    valid_time = np.isfinite(track[:, 2])
    valid = valid_coordinate & valid_time
    crs, transformer = _projection(track, nodes, valid)
    xy = np.full((count, 2), np.nan)
    xy[valid_coordinate] = np.column_stack(transformer.transform(*track[valid_coordinate, :2].T))
    nxy = np.column_stack(transformer.transform(*nodes.T))
    if not np.isfinite(nxy).all() or not np.isfinite(xy[valid_coordinate]).all():
        raise ValueError(f'{raw.trip_id}: nonfinite projected geometry')
    # Preserve original input order. Never sort timestamps or join over invalid rows.
    dt = np.full(count, np.nan)
    dt[1:] = np.diff(track[:, 2])
    both_valid = np.zeros(count, dtype=bool)
    both_valid[1:] = valid[1:] & valid[:-1]
    time_reversal = both_valid & (dt <= 0)
    gap = both_valid & (dt > config.gap_seconds)
    usable_step = both_valid & (dt > 0) & ~gap
    restart = valid & ~usable_step
    sequence = np.cumsum(restart).astype(np.int64) - 1
    sequence[~valid] = -1
    displacement = np.full(count, np.nan)
    raw_speed = np.full(count, np.nan)
    positive_pair = both_valid & (dt > 0)
    delta = np.full_like(xy, np.nan)
    delta[1:] = np.diff(xy, axis=0)
    displacement[positive_pair] = np.linalg.norm(delta[positive_pair], axis=1)
    raw_speed[positive_pair] = displacement[positive_pair] / dt[positive_pair]
    speed = np.where(usable_step, raw_speed, np.nan)
    heading = np.full(count, np.nan)
    heading_valid = usable_step & (displacement >= config.heading_min_distance_m)
    # Bearing in the projected grid: clockwise from north, not geodesic true north.
    heading[heading_valid] = np.degrees(np.arctan2(delta[heading_valid, 0], delta[heading_valid, 1])) % 360
    observations = pd.DataFrame(dict(
        observation_id=np.arange(count), longitude=track[:, 0], latitude=track[:, 1], timestamp_s=track[:, 2],
        x_m=xy[:, 0], y_m=xy[:, 1], valid_coordinate=valid_coordinate, valid_timestamp=valid_time,
        valid_observation=valid, duplicate_row=pd.DataFrame(track).duplicated().to_numpy(),
        dt_s=dt, gap_before=gap, nonpositive_time_before=time_reversal,
        sequence_id=sequence, sequence_start=restart, motion_valid=usable_step,
        step_distance_m=np.where(usable_step, displacement, np.nan),
        raw_step_distance_m=displacement, raw_implied_speed_mps=raw_speed,
        speed_mps=speed, speed_flag=raw_speed > config.speed_flag_mps,
        heading_grid_deg=heading, heading_valid=heading_valid))
    vector = nxy[arcs[:, 1]] - nxy[arcs[:, 0]]
    length = np.linalg.norm(vector, axis=1)
    zero = length <= config.zero_length_tolerance_m
    arc_heading = np.full(len(arcs), np.nan)
    arc_heading[~zero] = np.degrees(np.arctan2(vector[~zero, 0], vector[~zero, 1])) % 360
    # Parallel arc rows retain their IDs. Only the node-distance graph collapses
    # parallel connections, using minimum length rather than scipy's default sum.
    unique_pairs, inverse = np.unique(arcs, axis=0, return_inverse=True)
    minimum = np.full(len(unique_pairs), np.inf)
    np.minimum.at(minimum, inverse, length)
    graph = csr_matrix((minimum, (unique_pairs[:, 0], unique_pairs[:, 1])), shape=(len(nodes), len(nodes)))
    weak_count, weak_labels = connected_components(graph, directed=True, connection='weak')
    strong_count, strong_labels = connected_components(graph, directed=True, connection='strong')
    node_table = pd.DataFrame(dict(node_id=np.arange(len(nodes)), longitude=nodes[:, 0], latitude=nodes[:, 1],
        x_m=nxy[:, 0], y_m=nxy[:, 1], weak_component=weak_labels, strong_component=strong_labels))
    arc_table = pd.DataFrame(dict(arc_id=np.arange(len(arcs)), from_node_id=arcs[:, 0], to_node_id=arcs[:, 1],
        length_m=length, heading_grid_deg=arc_heading, zero_length=zero,
        candidate_geometry_valid=~zero, duplicate_connection=pd.DataFrame(arcs).duplicated().to_numpy()))
    centre = np.median(xy[valid], axis=0) if valid.any() else np.median(nxy, axis=0)
    radius = float(np.linalg.norm(np.vstack([nxy, xy[valid_coordinate]]) - centre, axis=1).max())
    metadata = dict(schema_version=1, trip_id=raw.trip_id, partition=raw.partition,
        source_sha256=raw.source_sha256, config=config.to_dict(), projection=crs.to_string(),
        projection_wkt=crs.to_wkt(), extent_radius_m=radius,
        projection_needs_review=radius > config.projection_review_radius_m,
        observation_count=count, valid_observations=int(valid.sum()), invalid_observations=int((~valid).sum()),
        sequence_count=int(restart.sum()), gap_count=int(gap.sum()), nonpositive_time_steps=int(time_reversal.sum()),
        duplicate_observations=int(observations.duplicate_row.sum()), speed_flags=int(observations.speed_flag.sum()),
        node_count=len(nodes), arc_count=len(arcs), zero_length_arcs=int(zero.sum()),
        duplicate_connections=int(arc_table.duplicate_connection.sum()), weak_components=int(weak_count),
        strong_components=int(strong_count), reference_used=False)
    return PreprocessedTrip(raw.trip_id, raw.partition, observations, node_table, arc_table, graph, metadata)

