"""Strict raw loaders with reference routes separated from prediction inputs."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass(frozen=True)
class RawTrip:
    trip_id: str
    partition: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    source_sha256: dict


def read_table(path: Path, width: int, integer: bool = False) -> np.ndarray:
    """Reject empty, malformed or blank rows; never drop rows that define IDs."""
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()
    if not lines:
        raise ValueError(f'{path}: empty file')
    for line_number, line in enumerate(lines, 1):
        if len(line.split()) != width:
            raise ValueError(f'{path}:{line_number}: expected {width} columns')
    try:
        values = np.loadtxt(path, dtype=np.float64, comments=None, ndmin=2)
    except ValueError as exc:
        raise ValueError(f'{path}: nonnumeric input') from exc
    if values.shape != (len(lines), width):
        raise ValueError(f'{path}: unexpected table shape {values.shape}')
    if integer:
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**53).any() or (values != np.floor(values)).any()):
            raise ValueError(f'{path}: IDs must be finite, nonnegative exact integers below 2**53')
        values = values.astype(np.int64)
    return values


def load_trip(folder: Path, partition: str = '') -> RawTrip:
    """Load only track, nodes and arcs. Never open a .route file here."""
    folder = Path(folder)
    tables, hashes = {}, {}
    for extension, width in [('track', 3), ('nodes', 2), ('arcs', 2)]:
        path = folder / f'{folder.name}.{extension}'
        tables[extension] = read_table(path, width, integer=extension == 'arcs')
        hashes[extension] = hashlib.sha256(path.read_bytes()).hexdigest()
    nodes, arcs = tables['nodes'], tables['arcs']
    if not np.isfinite(nodes).all() or (abs(nodes[:, 0]) > 180).any() or (abs(nodes[:, 1]) > 90).any():
        raise ValueError(f'{folder}: invalid node coordinates; cannot construct a reliable graph')
    if (arcs >= len(nodes)).any():
        raise ValueError(f'{folder}: arc refers to a node outside the node table')
    return RawTrip(folder.name, partition, tables['track'], nodes, arcs, hashes)


def load_reference_route(folder: Path, arc_count: int) -> np.ndarray:
    """Explicit development/scoring-only access; not used by preprocessing."""
    folder = Path(folder)
    route = read_table(folder / f'{folder.name}.route', 1, integer=True).ravel()
    if (route >= arc_count).any():
        raise ValueError(f'{folder}: reference arc ID outside the arc table')
    return route


def iter_trip_folders(data_root: Path, partitions=('train', 'student_test')):
    """Development batch entry point deliberately excludes evaluator partitions."""
    for partition in partitions:
        if partition not in {'train', 'student_test'}:
            raise ValueError(f'Unsupported development partition: {partition}')
        directory = Path(data_root) / partition
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        folders = sorted(p for p in directory.iterdir() if p.is_dir())
        if not folders:
            raise ValueError(f'{directory}: no trips')
        for folder in folders:
            yield partition, folder

