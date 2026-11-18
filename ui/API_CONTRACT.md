# Implemented API contract

The UI connects to api/main.py using these operations:

| Operation | Request | Response |
|---|---|---|
| Readiness | GET /health | status equal to ok |
| List trips | GET /trips | count and trips array containing trip_id strings |
| Predict | POST /predict | Body containing only the selected trip_id |

Trip IDs remain strings to preserve leading zeroes. There is no preview endpoint.
Trip selection updates local metadata; the Reconstruct button requests inference.

## Prediction mapping

| API field | UI handling |
|---|---|
| gps_points | [latitude, longitude, timestamp] converted once to longitude/latitude |
| predicted_route | Each array becomes a separate GeoJSON MultiLineString member |
| metrics.top1_accuracy | Fraction displayed as a percentage |
| metrics.route_continuity | Fraction displayed as a percentage |
| metrics.route_f1 | Route overlap F1 displayed to four decimal places |
| metrics.classifier_only_top1_accuracy | Aggregate comparison percentage |
| metrics.scored_points | Scored count against returned valid GPS observations |
| counts.route_segments | Number of route sections |
| timings_s.total_prediction | Prediction seconds, excluding scoring/network time |
| metrics equal to null | Scores unavailable; prediction still usable |

The API also returns candidate counts, stage timings, model information and other
aggregate metrics. The adapter whitelists only required display fields. Reference
geometry and pointwise labels are never forwarded to the interface.

Duration is the GPS timestamp span. Distance sums geodesic distances along each
predicted section individually. Gaps are never connected. Sections with fewer than
two vertices are omitted from rendered geometry. There is no classifier-only
geometry in this API response.

The adapter rejects mismatched trip IDs, invalid coordinates, nonfinite timings,
invalid fractions and negative/noninteger counts. Export is a FeatureCollection
with predicted MultiLineString geometry and properties trip_id, demo false and
source PathSnap /predict.

## Failures and network setup

Non-2xx responses, unreachable services, malformed responses and a 120-second
timeout show errors with retry. Changing trips aborts the browser request and
ignores stale results; it does not guarantee server computation stops. Empty routes
retain returned GPS observations and show an error. Empty trip lists show a
connected-but-no-data state. API mode never falls back to synthetic fixtures.

The Connection dialog accepts HTTP/HTTPS backend URLs without credentials, query
strings or fragments. The backend must allow cross-origin GET and POST JSON
preflight. The default API CORS configuration permits all origins; PATHSNAP_CORS
can restrict this to the frontend origin. Prediction uses track/nodes/arcs, with
reference scoring performed separately on the server after prediction.
