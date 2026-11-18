# Frontend verification

The UI is connected to the pushed FastAPI implementation. No training, tuning or
evaluator-test data was used for these integration checks.

## Automated checks

- Frontend syntax checks pass.
- Twelve service tests cover response validation, coordinate conversion, separate
  route sections, aggregate field filtering, null metrics, real endpoint payloads,
  empty trip lists, URL validation, failures and cancellation.
- The existing Python suite passes all 34 tests, including API/inference checks.
- The opt-in live smoke check passes on all ten student-test trips through the
  actual frontend adapter. GPS counts and route section counts match the API.
  Cross-origin POST preflight also passes. The check refuses a different trip list.

## Browser checks

- Health and trip list load from the real local API.
- Reconstruction shows real GPS, HMM routes, aggregate scores and prediction time.
- Trip 00000065 renders six separate route sections.
- Downloaded GeoJSON parses with six sections, demo false and the API source.
- An unreachable backend address displays an error; restoring the correct address
  reconnects successfully without a synthetic fallback.
- Desktop layout and live OpenStreetMap attribution were inspected.

The two-laptop network deployment was not tested. Reference geometry remains
server-side. Browser checks establish integration behaviour, not new model-quality
claims. See README.md for reproducible startup commands.
