# Visual inspection notes

These notes accompany `dataset_audit.md` and the saved development-data figures.
They describe inspected plots, not model predictions or measured model performance.

## Training trip 00000001: relatively close reference agreement

The full-route plot shows GPS observations following the reference corridor. The
zoom reveals lateral scatter, including observations nearer a neighbouring road
than the reference line. Every observation is within approximately 28 m of some
reference segment. This illustrates why proximity alone can be ambiguous even
when the overall route looks well aligned.

The audit flags 171 implied speeds above 55 m/s. A speed threshold alone cannot
establish a GPS error: fast travel may be real. Investigate the trajectory and
road context before removing observations.

## Training trip 00000090: a large temporal break

GPS points remain spatially close to the supplied reference, but the largest
consecutive timestamp gap is 153,064 seconds (42.52 hours). Spatial agreement does
not establish that the vehicle followed the reference continuously through that
unobserved interval. Gap handling should allow a sequence restart or explicitly
mark an uncertain interval rather than force a confident connection.

## Training trip 00000054: sustained reference disagreement

The full-route plot largely follows the same broad corridor, while the zoom shows
a dense GPS cluster displaced from the supplied reference geometry. The distance
plot shows sustained deviations, rather than a few isolated outliers:
16,849 of 19,313 observations (87.24%) lie more than 50 m from every reference
segment; the 95th percentile distance is approximately 210.05 m.

This observation does not distinguish GPS bias, off-network movement, incomplete
road geometry, stationary activity, or reference-route limitations. Do not erase
these observations or assert a cause without further investigation. Candidate
radius choices and reference-derived label confidence need to account for this
case. A high point-weighted score could also be dominated by long trips or long
clusters, so retain per-trip evaluation alongside aggregate metrics.

## Cross-trip implications

- The supplied graphs cover multiple world regions. Use appropriate per-trip
  metric projections and treat graph IDs as local to their trip.
- Fourteen consecutive reference pairs have different endpoint node IDs; direct
  inspection also found their endpoint coordinates differ. They are reference
  sequence breaks, not proof that no connecting path exists elsewhere in the graph.
- The supplied graphs include zero-length segments and duplicate directed node
  pairs. Keep original arc IDs and handle zero-length geometry explicitly;
  deduplicating rows would change reference IDs.
- Reference proximity is not observation-level accuracy. Order-aware reference
  alignment remains necessary before training labels and pointwise evaluation
  are defined.

## Verification

The audit completed for all 80 training and 10 student-testing trips without
parsing failures. Partition totals, the 227,577 observation total, 1,260 gap events
above 5 seconds, and 14 reference endpoint breaks were checked against saved
tables. Static plots were rendered and visually inspected; all three Folium HTML
outputs were checked for GPS, road and reference layers. Interactive maps have
not been browser-rendered during this audit and require CDN/tile access.

No raw files were modified. No evaluator-testing partition was inspected. No
classifier or decoder has been trained or evaluated at this stage.
