# PathSnap UI

The frontend connects to FastAPI by default and displays real GPS observations,
tuned LightGBM + HMM/Viterbi routes, server-side scores and GeoJSON exports.
No frontend dependency installation or build is required.

## Run locally

From the repository root, install Python dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Start the backend on development trips for rehearsal:

```powershell
$env:PATHSNAP_TRIPS_DIR = "data/Student folder/student_test"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, from the repository root:

```powershell
python -m http.server 5173 --bind 127.0.0.1 --directory ui
```

Open http://127.0.0.1:5173. Select a trip and click Reconstruct route.
The default API URL is http://127.0.0.1:8000. The Connection dialog changes
this address and remembers it in browser storage. Selection alone does not run
inference; GPS observations arrive with the prediction response.

## Two laptops

Run the backend on Laptop 1 with `--host 0.0.0.0`. Serve the frontend on Laptop 2
and enter Laptop 1's reachable URL in Connection, for example
`http://192.168.1.20:8000`. The network must permit this connection. If overriding
`PATHSNAP_CORS`, allow the actual frontend origin. HTTPS pages require an HTTPS
API. See [backend instructions](../api/README.md). Physical two-laptop operation
was not tested; local cross-origin requests were verified.

## Display behaviour

- GPS points and predicted route sections remain separate across gaps.
- Scores come from the backend; missing scores show as unavailable.
- Prediction time excludes server-side scoring and network transfer.
- Duration is the GPS timestamp span. Distance sums geodesic distances within
  predicted sections without bridging gaps.
- Classifier-only accuracy is an aggregate comparison. The API supplies no
  classifier geometry, so that map control is hidden.
- Export contains predicted MultiLineString geometry with `demo: false`.
- Reference routes and pointwise labels stay on the backend.
- Changing trips cancels the browser request and discards stale results.
- Connection errors and empty trip lists show actionable messages.

OpenStreetMap street tiles require internet and retain attribution. If tiles
fail, GPS and route overlays remain usable with a basemap notice. Leaflet is
vendored locally. Synthetic fixtures are available only by explicitly setting
`mode: 'demo'` in config.js. API failures never silently use demo results.

## Verification

From ui, using Node.js 18 or newer:

```powershell
npm run check
npm test
node tests/live-api-smoke.js http://127.0.0.1:8000
```

The opt-in smoke check refuses trip lists other than the ten student-test IDs.
It checks real predictions through the frontend adapter and CORS preflight.
It does not train or tune. See [QA.md](QA.md) and [API_CONTRACT.md](API_CONTRACT.md).
