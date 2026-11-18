# Development dataset

The supplied raw dataset is versioned in `Student folder/` relative to this
directory. The files were copied byte-for-byte from the originally supplied
student package; they have not been cleaned, filtered or renumbered.

| Partition | Trips | Files | GPS observations |
|---|---:|---:|---:|
| train | 80 | 320 | 208,417 |
| student_test | 10 | 40 | 19,160 |

Each trip directory contains four whitespace-separated, headerless files:

- `.track`: longitude, latitude, timestamp in seconds.
- `.nodes`: longitude, latitude; zero-based row position is the node ID.
- `.arcs`: from-node ID, to-node ID; zero-based row position is the arc ID.
- `.route`: ordered reference arc IDs, not one label per GPS observation.

Node and arc IDs are local to each trip's graph. Preserve file row ordering.
`manifest.csv` records each relative filename, byte count and SHA-256 digest.

Run the read-only audit from the repository root:

```powershell
python experiments/01_dataset_exploration.py
```

Use `--data-root PATH` to audit an alternative development-data location. The
script accesses only the `train` and `student_test` subdirectories. Reserved
evaluator data is not included. Development reference routes may be used for
label preparation and scoring, but must not become prediction input features.
