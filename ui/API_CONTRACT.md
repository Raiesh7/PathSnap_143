# Proposed API contract

This is a frontend integration proposal, not an implemented or agreed backend.
The UI currently uses synthetic fixtures and sends no backend requests.

## Operations

| Operation | Route | Response |
|---|---|---|
| List trips | GET /trips | Object containing a trips array with id, name and optional description |
| Preview observations | GET /trips/{trip_id} | Trip metadata and observed GPS, without reference labels |
| Reconstruct | POST /match | Predicted geometry and optional aggregate scoring metrics |

The reconstruction request body contains only `{"trip_id":"00000015"}`.
The backend looks up inputs and runs the frozen prediction pipeline. Reference
labels are used only in separate server-side scoring after prediction.

## Response shape for preview and reconstruction

Coordinates follow **GeoJSON longitude, latitude order**. Routes use
MultiLineString so separate sequences or gaps are not joined by invented lines.
The preview response can omit metrics and route geometry. Metadata values can be
omitted when unavailable. The following numbers and coordinates are illustrative.

```json
{
  "trip": {
    "id": "00000015",
    "name": "Trip 00000015",
    "description": "GPS trajectory",
    "duration_s": 510,
    "distance_m": 3260
  },
  "observations": [
    {"id": 0, "coordinates": [77.5947, 12.9756], "timestamp_s": 0},
    {"id": 1, "coordinates": [77.5955, 12.9750], "timestamp_s": 3}
  ],
  "matched_route": {
    "type": "MultiLineString",
    "coordinates": [[[77.5947, 12.9756], [77.5955, 12.9750]]]
  },
  "classifier_route": null,
  "metrics": {
    "top1_accuracy": null,
    "route_continuity": null,
    "processing_ms": 842
  },
  "warnings": [],
  "demo": false
}
```

`classifier_route` uses the same geometry type and is optional. The API must
split its geometry at sequence breaks or missing predictions. Do not connect
unobserved gaps simply to draw a continuous line.

Accuracy and continuity are fractions from 0 to 1, not percentages. Processing
time is nonnegative milliseconds. Null or missing metrics display as unavailable;
the frontend never calculates accuracy from ground truth. `demo` must be false
for real inference. The displayed number of observations is derived from the
observation array, not trusted from metadata.

Trip IDs must remain strings to preserve leading zeroes. Return only finite,
valid coordinates; preserve original observation IDs if preprocessing excludes
invalid positions. HTML in service strings is rendered as plain text.

## Failure behaviour

Use non-2xx responses for request or backend failures. The adapter displays a
generic message with HTTP status and supports retry. It also handles unreachable
services, invalid response shapes, cancelled requests and a 30-second timeout.
For a completed request with no valid route, return matched_route null and a
clear warning; the UI presents the empty-route message without invented geometry.

Model loading, reference data, evaluation labels, filesystem paths, secrets and
training operations stay on the backend. Never return reference routes or
pointwise ground-truth labels to this UI.
