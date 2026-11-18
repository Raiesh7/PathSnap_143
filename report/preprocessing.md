# Preprocessing implementation and results

## Scope and reproducibility

This stage turns raw development trips into row-preserving metric geometry,
directed road graphs and gap-aware observation tables. It does not generate
candidates, derive pointwise reference labels, fit scalers, train classifiers or
decode routes. Reference-route alignment remains a separate next stage.

Run from the repository root after installing `requirements.txt`:

```powershell
python experiments/01b_preprocess_dataset.py --cache-dir data/processed
python -m unittest discover -s tests -v
python tests/validate_preprocessed_dataset.py
```

Without `--cache-dir`, the batch command produces just summary tables. Use
`--data-root PATH`, `--output-dir PATH`, and `--gap-seconds VALUE` to override
defaults. The batch runner reads only `train` and `student_test`. All thresholds
and implementation SHA-256 hashes are recorded in the generated summary; source
file hashes and projection details are recorded with each cached trip.

## Implemented decisions

| Concern | Implemented rule | Reason and consequence |
|---|---|---|
| File schema | Require exact column widths and numeric values; reject blank/malformed/empty graph tables | Dropping a row would change row-based IDs. A failing trip is reported, not silently repaired. |
| Node and arc IDs | Preserve every original node/arc row and zero-based ID; validate integer IDs and bounds | IDs are local to each trip, not global class labels. |
| Invalid GPS or timestamps | Retain observations; mark invalid rows with `valid_observation=False`, `sequence_id=-1` | Invalid rows cannot supply candidates or motion. The next valid observation starts a new sequence. |
| Invalid graph geometry | Reject invalid coordinates or node references | Downstream routing requires a usable graph. |
| Coordinates | Project longitude-first WGS84 coordinates into per-trip metric geometry | Use local UTM for suitable extents/latitudes; fall back to local azimuthal equidistant otherwise. Antimeridian selection is handled. |
| Projection extent | Flag extents beyond 250 km from the trip centre for review | The flag is a conservative review rule, not an accuracy guarantee. The observed maximum radius was 48.35 km; no supplied trip triggered it. |
| Time order | Preserve input order; break at nonpositive timestamp steps | Sorting could alter the observed trajectory and conceal data problems. |
| Sampling gaps | Start a new sequence when consecutive valid observations are more than 5 seconds apart | Five seconds reuses the audit diagnostic threshold as an initial policy. It is configurable and is not a tuned optimum. No interpolation is performed. |
| Motion | Compute incoming speed, displacement and heading only for positive-time steps within a sequence | Sequence starts and invalid/gap boundaries have missing motion values, not invented zero values. |
| Stationary observations | Retain points; omit heading when displacement is below 1 metre | Tiny displacement produces unstable direction. Valid stationary speed can still equal zero. |
| Fast travel | Flag implied speed above 55 m/s (198 km/h); retain the observation and sequence | High speed is not proof of GPS error. No automatic clipping or filtering is justified by the audit. |
| Exact duplicate observations | Flag later duplicates; preserve their rows | Duplicate timestamps also form sequence boundaries. |
| Zero-length road segments | Preserve arc IDs and graph connectivity, set `candidate_geometry_valid=False`, omit direction | Avoid division by zero and undefined heading. Candidate generation must explicitly account for point-like arcs rather than silently label them as ordinary segments. |
| Duplicate road connections | Preserve all arc rows; use the minimum length per directed node pair in the routing matrix | Sparse matrix construction must not sum duplicate edge costs. The routing matrix does not identify a unique parallel arc; use the original arc table for arc correspondence. |
| Disconnected roads | Preserve component structure; store weak and strong component IDs | Do not fabricate connecting roads or turn the graph undirected. |
| Reference data | The default loader and preprocessing never open `.route` files | Reference geometry cannot affect prediction-time preprocessing. A separate explicit reference loader exists for later development/scoring. |

The 5-second, 55-m/s and 1-metre defaults are explicit starting policies.
Any future adjustment must be justified with training-trip validation. They were
not selected by optimising student-testing or reserved evaluator results.

## Measured effects on the supplied dataset

| Quantity | Training | Student testing | Total |
|---|---:|---:|---:|
| Trips processed | 80 | 10 | 90 |
| Observations retained | 208,417 | 19,160 | 227,577 |
| Invalid observations | 0 | 0 | 0 |
| Exact duplicate observations | 0 | 0 | 0 |
| Nonpositive timestamp steps | 0 | 0 | 0 |
| Gaps above 5 seconds | 1,216 | 44 | 1,260 |
| Resulting sequences | 1,296 | 54 | 1,350 |
| Speed flags above 55 m/s | 989 | 4 | 993 |
| Zero-length arc rows retained | 100 | 6 | 106 |
| Duplicate connection rows retained (after first occurrence) | 1,775 | 128 | 1,903 |

No observation or arc was removed, no coordinates were corrected, and no
reference-route break was repaired. In particular, the sustained reference
disagreement in training trip `00000054` remains for later alignment analysis.

The 1,350 sequences are subdivisions of the original trips, not independent
cross-validation groups. All sequences from one trip must stay in the same fold.

## Output contract

`src.data_loader.load_trip(folder, partition)` returns track, nodes, arcs and their
source hashes. It works without a reference file. Malformed reference contents
do not affect it. `load_reference_route` is an explicit separate operation.

`src.preprocessing.preprocess_trip(raw_trip, config)` returns:

- `observations`: original observation ID, longitude, latitude, timestamp,
  projected coordinates, validity/duplicate/gap/speed flags, sequence ID, incoming
  displacement, speed and projected-grid heading.
- `nodes`: original node IDs, coordinates and weak/strong component IDs.
- `arcs`: original arc IDs and endpoints, metric lengths, grid headings,
  zero-length and duplicate flags.
- `graph`: directed CSR node-to-node minimum-distance matrix, with explicit
  zero-length edges retained.
- `metadata`: source hashes, policy, coordinate system and measured counts.

`speed_mps`, `step_distance_m` and `heading_grid_deg` are unavailable at sequence
boundaries. `raw_implied_speed_mps` and `raw_step_distance_m` are separate diagnostic
fields that can span positive-time GPS gaps; **do not substitute them into
gap-aware motion features**. Headings are clockwise from projected north; GPS
and road headings share that same projection.

The optional `data/processed/<partition>/<trip>/inputs.npz` contains typed arrays,
plus `metadata.json`. Load it with `load_preprocessed_trip(folder)`; NumPy pickle
loading is disabled. Caches are reproducible and gitignored, while small summary
tables and this documentation can be committed. Rebuild caches if the source
files, policy or implementation changes; the loader checks schema and row IDs
but does not automatically compare current source files to recorded hashes.

## Evidence and validation

- `results/preprocessing/trip_summary.csv`: counts and projection diagnostics
  for every trip.
- `results/preprocessing/flagged_observations.csv`: original observation IDs of
  invalid observations, gaps, nonpositive time steps and speed flags.
- `results/preprocessing/summary.json`: actual totals, policy, versions and code
  hashes; the run completed without trip failures.
- `results/preprocessing/run.log`: batch execution output.
- `results/preprocessing/tests.log`: 16 passing behavioural tests.
- `results/preprocessing/validation.json`: checks of all saved caches against
  the audit and source manifest.

Tests exercise directed routing, explicit zero-length edges, duplicate-edge
costs, ID preservation, speed/gap boundaries, invalid observations, malformed
files, stationary heading, antimeridian geometry, cache round trips and
preprocessing independence from missing or poisoned reference routes.

No evaluator-testing data was used and no model-performance claim is made here.
The next stage is candidate generation and order-aware reference-label alignment,
followed by feature engineering and trip-grouped model training.
