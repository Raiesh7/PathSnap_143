# Feature engineering

## Prediction task and row structure

Each model scores whether a candidate road segment matches a GPS observation.
One observation with six candidates produces six feature rows. The implementation
in [src/features.py](../src/features.py) defines 21 predictor columns in
`FEATURE_COLUMNS`. These features use GPS observations and the road network;
reference routes are used separately to derive labels.

Candidate generation uses a 50-metre radius and keeps up to eight usable arcs per
valid observation. It excludes zero-length geometry and later duplicate directed
connections. Candidates are ranked by exact point-to-segment distance, with arc
ID as a deterministic distance-tie ordering. The following definitions describe
the current implementation, not a claim that every feature independently improves
accuracy. No feature-ablation experiment has established individual contributions.

## Geometry features

Let d be the observation's distance to this candidate, d_best the minimum distance
among its retained candidates, t its clipped projection fraction along the arc,
and L the arc length. Coordinates and distances use the trip's projected metric
coordinate system.

| Feature | Calculation and units | Why included |
|---|---|---|
| `dist_m` | Distance from the GPS point to the closest point on the segment, metres | Measures absolute spatial agreement. |
| `rank` | Zero-based distance rank among retained candidates | Provides the candidate's relative proximity order. |
| `n_candidates` | Number of retained candidates for the observation | Describes local ambiguity and candidate-set size; repeated across its rows. |
| `dist_minus_best` | d minus d_best, metres | Indicates how much spatial fit is lost relative to the closest candidate. |
| `dist_ratio_best` | (d + 1) divided by (d_best + 1), dimensionless | Expresses relative distance; the one-metre offset avoids division by zero. |
| `proj_frac` | t between 0 and 1; 0 at the start node and 1 at the end node | Describes the observation's projected position along the segment. |
| `proj_edge_dist` | min(t, 1 - t) multiplied by L, metres | Measures distance along the segment to the nearer endpoint, giving junction and endpoint context. |

The projection uses a clipped point-to-segment calculation, not distance to an
infinite line. Absolute and relative proximity provide different information:
the nearest road may still be far away, while several roads can have similar
distances. Distance alone cannot distinguish opposite directions sharing geometry.

## Direction features

Angular differences are wrapped into the range 0 to 180 degrees. Zero means
agreement and 180 means opposing directions. GPS and road headings both use
projected grid north, measured clockwise.

| Feature | Calculation and units | Why included |
|---|---|---|
| `heading_diff` | Wrapped absolute difference between incoming GPS heading and candidate arc heading, degrees | Helps distinguish directed arcs and nearby roads pointing in different directions. |
| `heading_diff_smooth` | Same comparison using a heading estimated from surrounding observations, degrees | Reduces sensitivity to short-term positional jitter. |
| `heading_available` | 1 if the smoothed heading difference is finite, otherwise 0 | Makes the availability of the smoothed directional clue explicit. |

The smoothed heading uses positions up to two observations before and after the
current observation, within its preprocessing sequence. Near sequence ends it
falls back to the immediate neighbour, then the current position. It is missing
when the resulting displacement is below three metres. The incoming heading from
preprocessing is missing at sequence starts and for steps below one metre.
`heading_available` specifically describes the smoothed difference, not every
possible direction feature.

## Motion features

| Feature | Calculation and units | Why included |
|---|---|---|
| `speed_mps` | Incoming projected displacement divided by elapsed time within a valid sequence, metres per second | Provides movement context for interpreting candidate geometry. |
| `dt_s` | Raw current timestamp minus preceding timestamp, seconds | Represents sampling interval, including a large positive gap at a sequence restart. |
| `speed_flag` | 1 if raw implied speed exceeds 55 metres per second, otherwise 0 | Signals an unusual movement estimate without deleting fast observations. |

`speed_mps` is missing at sequence starts and invalid transitions. Valid stationary
speed may be zero. The flag is a diagnostic: high speed is not proof of a GPS
error. Raw implied speed can span positive-time gaps, while the ordinary speed
feature does not. These three values are observation-level information repeated
for each candidate; they allow interactions with candidate-specific features.

## Road topology features

| Feature | Calculation and units | Why included |
|---|---|---|
| `arc_length_m` | Projected endpoint distance, metres | Describes the spatial scale of the candidate segment. |
| `from_in_degree` | Number of supplied arc rows ending at the candidate's start node | Describes incoming connectivity at the start of the arc. |
| `to_out_degree` | Number of supplied arc rows starting at the candidate's end node | Describes outgoing connectivity at the end of the arc. |
| `two_way` | 1 if a reverse node connection exists in the supplied arc table, otherwise 0 | Indicates whether both directed connections are represented. |

Degrees count arc rows, not unique neighbouring nodes; duplicate connections can
increase these counts. A reverse arc's presence describes the supplied graph and
does not independently verify traffic regulations. These features do not provide
road class or speed limits, which are not supplied attributes.

## Local context features

The previous and next observations are selected within the same preprocessing
sequence. Context therefore does not cross a sequence boundary.

| Feature | Calculation and units | Why included |
|---|---|---|
| `dist_prev_obs` | Previous observation's distance to this same candidate segment, metres | Indicates whether the candidate also explains preceding movement. |
| `dist_next_obs` | Next observation's distance to this same candidate segment, metres | Indicates whether the candidate also explains following movement. |
| `same_as_prev_nearest` | 1 if the current candidate equals the previous observation's heuristic best-guess arc | Supplies a local persistence clue. |
| `follows_prev_nearest` | 1 if the previous best arc's end node equals this candidate's start node, excluding its immediate reverse connection | Supplies a local directed-connectivity clue. |

The previous best guess is not a reference label. The implementation first
prioritises candidates no more than one metre farther than the best distance.
Within that group it orders by smoothed heading difference, falling back to
incoming heading difference and then 180 degrees when neither is available;
distance breaks remaining ties. The first ordered candidate becomes the guess.

Neighbour distances are missing when the neighbour is unavailable. The two
binary context flags default to zero when no previous best-guess candidate is
available; zero then does not prove a road switch or disconnection. These local
features do not replace the HMM/Viterbi sequence decoder.

## Labels identifiers and missing values

`label`, `label_valid`, reference arc IDs, alignment distances, trip IDs,
observation IDs, candidate arc IDs and sequence IDs are not predictor columns.
The feature function copies its input table, so it can preserve attached labels
and identifiers. Training explicitly selects only `FEATURE_COLUMNS` to prevent
them entering the estimator.

Logistic Regression, KNN and MLP use training-fold median imputation, missing-value
indicators and scaling. Random Forest uses training-fold imputation and missing
indicators. LightGBM handles missing values natively. Thus 21 original predictors
can become more than 21 estimator inputs in pipelines that add missing indicators.
Learned transforms remain inside the training pipeline and are fitted separately
in each cross-validation fold.

## Scope and limitations

- Smoothed heading and `dist_next_obs` use future positions. The design supports
  reconstruction of a recorded trip; immediate online prediction needs causal
  alternatives or a defined look-ahead delay.
- Validation groups by original trip, not observation or sequence. All sequences
  from one trip stay together.
- The one-metre incoming-heading threshold, three-metre smoothed-heading threshold
  and one-metre best-distance band are engineering choices, not experimentally
  established optimal values.
- Feature importances show associations used by a fitted model, not proof of
  causal effects or independent benefit. Redundant geometry and direction features
  may share information.
- No feature or threshold was changed based on reserved evaluator-test results.

For example, a candidate eight metres away with a five-degree direction mismatch
can be more plausible than one six metres away with an 85-degree mismatch. The
classifier learns how to combine these clues. This example illustrates the
feature rationale; it is not a measured experimental result.

## Implementation and evidence

- [Feature definitions and construction](../src/features.py)
- [Candidate geometry and distance calculation](../src/candidates.py)
- [Projection and gap-aware motion](../src/preprocessing.py)
- [Reference-derived labels](../src/labels.py)
- [Estimator preprocessing](../src/models.py)
- [Training feature selection and shared folds](../src/training.py)
- [Candidate and feature behavioural tests](../tests/test_candidates.py)
- [Candidate coverage summaries](../results/candidates/summary.csv)

Tests check feature availability without reference labels, geometric distances,
wrapped angles, candidate constraints and unchanged candidate row counts. They
do not establish that all 21 features are necessary or improve generalisation.
