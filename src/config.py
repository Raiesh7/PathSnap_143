"""Explicit, serializable preprocessing policy; thresholds are not tuned models."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'Student folder'


@dataclass(frozen=True)
class PreprocessingConfig:
    # Mirrors the audit's gap flag. Revisit with training-only validation later.
    gap_seconds: float = 5.0
    speed_flag_mps: float = 55.0
    zero_length_tolerance_m: float = 1e-6
    heading_min_distance_m: float = 1.0
    projection_review_radius_m: float = 250_000.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')

    def to_dict(self):
        return asdict(self)
