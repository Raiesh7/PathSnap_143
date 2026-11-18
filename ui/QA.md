# Frontend verification

The interface is a synthetic preview, not a deployed evaluator system. The actual
backend has not been integrated or tested. No Python prediction or training code
was changed for this frontend.

## Automated checks

`npm run check` passes syntax checks for the application, map and service modules.
`npm test` passes six service-contract tests:

- All ten synthetic fixtures have valid coordinates and route geometry.
- Reference fields and pointwise labels are omitted by the response adapter.
- Mismatched trips, invalid coordinates and invalid metrics are rejected.
- The reconstruction request contains only the selected trip ID.
- HTTP and network failures produce useful errors.
- Cancellation stops a pending simulated reconstruction.

## Browser checks

The preview was exercised in the Codex in-app browser at desktop and mobile
breakpoints (1440 by 1050 and 390 by 844 pixels). Temporary viewport overrides
were reset after testing.

| Check | Observed result |
|---|---|
| Initial load | Ten demo choices, visible GPS track, disabled export and unfilled result metrics |
| Reconstruction | Loading overlay and phases, followed by reconstructed geometry and labelled demo metrics |
| Layer controls | GPS and classifier switches update checked states and visible map layers |
| Mid-run trip change | Previous request cancelled; new trip metadata shown with results cleared |
| Export | Downloaded GeoJSON parsed successfully; demo flag true and MultiLineString geometry retained |
| Help dialog | Explanation opens; Escape dismisses it and returns focus to the help control |
| Desktop layout | Sidebar, map and metric cards render without overlap |
| Mobile layout | Single-column controls, map and result cards fit without horizontal overflow |
| Console | No browser errors observed in the completed demo flow |

Error messaging is covered by service tests; a real API outage was not exercised
through the browser. GeoJSON export was tested with synthetic data only. No real
prediction quality, network deployment, CORS configuration or API compatibility
is claimed by these checks.
