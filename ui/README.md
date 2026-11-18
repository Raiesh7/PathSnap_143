# PathSnap trajectory workspace

Local frontend for trip selection, GPS and predicted-route inspection, aggregate
results and GeoJSON export. This branch implements the interface before backend
integration. It does **not** run a model or contain evaluator inputs, reference
routes, labels or real evaluation scores.

## Run locally

From the repository root:

```powershell
python -m http.server 5173 --bind 127.0.0.1 --directory ui
```

Open http://127.0.0.1:5173. No npm install or frontend build is required. Serve
over HTTP rather than opening index.html directly, because it uses ES modules.
The local vendored Leaflet 1.9.4 library makes the demo map work offline.

The current service is **demo mode**. All ten demo IDs, GPS points, street
geometry, routes and metrics are synthetic fixtures, explicitly labelled in the
interface and exported GeoJSON. They must not be presented as model performance
or used in the final evaluator demonstration.

## Interface behaviour

- Select one of ten synthetic journeys and inspect its observations.
- Reconstruct a route through a labelled simulated loading flow.
- Toggle GPS observations, classifier path and HMM/Viterbi route independently.
- Pan, zoom, fit the route and inspect observation tooltips.
- View illustrative aggregate scores; unavailable scores display an em dash.
- Export the current route as GeoJSON, marked `demo: true` in demo mode.
- Changing trips cancels an in-flight reconstruction and clears stale results.
- Loading, empty-route, service error and retry states are handled.
- Responsive layout, labelled controls, keyboard focus, live announcements and
  reduced-motion support are included.

## Backend connection seam

`services/trajectory-service.js` exposes `listTrips`, `preview` and
`reconstruct`. `app.js` consumes that interface. A separate API adapter is
provided, but is not used in demo mode and has not been tested against a real
backend. The proposed routes and JSON shapes are documented in
[API_CONTRACT.md](API_CONTRACT.md); agree them with the backend teammate first.

Then set `mode: 'api'` and the Laptop 1 URL in `config.js`, for example
`http://192.168.1.20:8000`. Serve this UI on Laptop 2. The API must allow its
actual frontend origin in CORS and remain accessible on the local network.
Production HTTPS pages also need an HTTPS API to avoid mixed-content blocking.

For real trips, the map uses OpenStreetMap tiles, which require internet and
display attribution. The offline street background is used only for synthetic
demo fixtures. Ground-truth routes must remain server-side; only observed GPS,
predicted geometry and aggregate metrics belong in the response.

## Checks

From the ui directory, using Node.js 18 or newer:

```powershell
npm run check
npm test
```

Service tests cover the ten fixtures, validation, omission of reference fields,
request payloads, useful failures and cancellation. Browser testing is also
needed for maps, exports, keyboard interaction and responsive layout.

## Files

- `index.html`, `styles.css`: layout and visual design.
- `app.js`: interface state and events.
- `map-view.js`: Leaflet map and route layers.
- `services/demo-data.js`: explicitly synthetic fixtures.
- `services/trajectory-service.js`: demo and API adapters.
- `config.js`: mode and API base URL.
- `vendor/leaflet/`: Leaflet 1.9.4, distributed under its included BSD license.

