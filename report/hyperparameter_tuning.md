# LightGBM Bayesian hyperparameter tuning

## Model choice

The recorded five-model comparison selected LightGBM. It achieved mean CV
candidate Macro F1 of 0.928518 and student-testing Top-1 accuracy of 0.896775.
Its CV Macro F1 advantage over Random Forest (0.928185) is small relative to
fold-to-fold variation, so this comparison does not establish statistical
superiority. The tuning experiment investigates this selected model further.

## Objective and development partitions

The objective is the arithmetic mean of binary candidate Macro F1 at threshold
0.5 across the five committed trip-grouped folds. This matches the primary CV
metric used in the model comparison. Each fold holds out 16 of the 80 training
trips; all sequences and candidate rows belonging to one trip remain together.
Top-1 road accuracy, log loss and direct route continuity are recorded as
secondary diagnostics. Candidate-level F1 and observation-level Top-1 accuracy
measure different behaviours and are reported separately.

The objective receives training data only. Student-testing tables are loaded
after the search finishes and the best parameters have been written to disk.
Student-testing results do not change the search, thresholds, parameters or
winner. Reserved evaluator-testing data is never loaded.

Reference-derived labels, candidate radius, K and all 21 features remain fixed.
Observations excluded by the existing label-validity rule remain excluded.
Consequently these are conditional results under that label policy, not
independently measured physical ground truth.

## Search strategy and configuration

Optuna's Tree-structured Parzen Estimator (TPE) sampler proposes configurations.
The run requests 20 total trials: the existing baseline is enqueued first, the
sampler uses five startup trials in total, and subsequent trials use TPE.
The random seed is 42. Trials run sequentially, with four LightGBM threads per
fit. No CV row cap, pruning or early stopping is used. Each completed trial
therefore evaluates all five folds at the sampled tree count.

| Parameter | Search space | Purpose |
|---|---|---|
| n_estimators | 200 to 700 in steps of 100 | Number of boosting trees |
| learning_rate | 0.02 to 0.15, logarithmic | Shrinkage applied to each tree |
| num_leaves | 15 to 95, integer | Tree complexity |
| min_child_samples | 20 to 250, logarithmic integer | Minimum observations in a leaf |
| subsample | 0.6 to 1.0 | Row sampling fraction; subsample_freq fixed at 1 |
| colsample_bytree | 0.6 to 1.0 | Feature sampling fraction |
| reg_alpha | 0.0 to 5.0 | L1 leaf-weight regularisation |
| reg_lambda | 0.0 to 10.0 | L2 leaf-weight regularisation |

The search explores complexity, shrinkage, sampling and regularisation while
keeping the computational cost practical. This finite search does not establish
a global optimum. Class weights, candidate generation and decision thresholds
are not tuned in this experiment.

The first trial uses the original settings: 400 trees, learning rate 0.05,
63 leaves, min_child_samples 50, subsample 0.8, colsample_bytree 0.8 and zero L1/L2
regularisation. Including this baseline ensures the best completed CV score
cannot be worse than the baseline score within this search.

## Fair baseline and final refit

The saved original model was trained on another machine with another LightGBM
version. The tuner reruns its configuration locally using the same data, folds,
library and thread settings as the search. The baseline CV trial and final
baseline fit are therefore the direct comparison, while the original five-model
results remain unchanged.

After parameter selection, both baseline and selected configurations are refitted
on all eligible candidate rows from the 80 training trips and evaluated on the
ten student-testing trips. Model-only inference timing excludes candidate
generation and feature extraction. Speed measurements are approximate single-run
timings on this machine, not deployment latency guarantees.

The selected model is stored separately as `lightgbm_tuned.joblib`. If the
baseline itself wins CV, the selected parameters are simply the baseline; the
word tuned describes the search procedure, not a guaranteed improvement.

## Evidence and reproducibility

Run `python experiments/03_bayesian_optimization.py --trials 20 --n-jobs 4`.
The script saves the following under `results/tuning/lightgbm/`:

- `search_config.json`: search ranges, objective, features, input/code hashes,
  seed, versions and fixed settings.
- `study.sqlite3`, `sampler_state.joblib`: persisted study and sampler state.
- `trials.csv`: sampled parameters, scores, states and timings.
- `trial_fold_metrics.csv`: metrics for every evaluated trial/fold pair.
- `best_params.json`, `best_cv_folds.csv`: CV-selected configuration and folds.
- `untuned_student_test_metrics.json`, `tuned_student_test_metrics.json`:
  final evaluation on the same student-testing observations.
- `comparison.csv`: baseline and selected-model metrics and timings.
- `student_test_predictions.parquet`: selected-model candidate probabilities.
- `feature_importance.csv`: selected-model gain importances.
- `run_info.json`: actual completed trial count, costs, provenance and reload check.
- `optimization_comparison.png`: actual search progression and final comparison.

The saved model is reloaded and checked against a sample of saved predictions.
Unit tests verify that folds keep trips together, reject invalid assignments,
and reproduce the baseline configuration inside the declared search space.

## Technical references

- [Optuna TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)

## Completed experiment results

All 20 trials completed, producing 100 fold evaluations without pruning or row
caps. The run took approximately 892 seconds (14.9 minutes), including the final
refits and evaluation. The best configuration was Optuna trial 9 (the tenth trial
in execution order).

| Metric | Untuned baseline | CV selected configuration |
|---|---:|---:|
| Mean CV candidate Macro F1 | 0.928518 | 0.929920 |
| CV Macro F1 standard deviation | 0.017735 | 0.017244 |
| Mean CV observation Top-1 accuracy | 0.883506 | 0.884508 |
| Student-test candidate Macro F1 | 0.934432 | 0.931910 |
| Student-test observation Top-1 accuracy | 0.896775 | 0.888988 |
| Student-test log loss | 0.095923 | 0.098418 |
| Direct route continuity | 0.946066 | 0.944756 |
| Model inference ms per 1000 observations | 36.857 | 22.500 |
| Full refit time in seconds | 9.445 | 7.011 |

The selected parameters are 200 trees, learning_rate 0.0330019874, num_leaves 90,
min_child_samples 79, subsample 0.7026156762, colsample_bytree 0.6630687540,
reg_alpha 4.6409411545 and reg_lambda 2.0354522375. Full precision values are
saved in `best_params.json`.

The CV Macro F1 gain is 0.001402, but student-testing Macro F1 decreased by
0.002522 and Top-1 accuracy decreased by approximately 0.779 percentage points.
The tuned model was faster in this timing run. Its CV advantage is small relative
to fold-to-fold variation and is not evidence of a statistically established
improvement. The search therefore did **not** improve held-out quality in this
experiment. Report this outcome rather than claiming tuning always helps.

Both final models were fitted on 1,058,904 eligible candidate rows from 80 trips.
Student testing covers 19,133 label-valid observations of 19,160 total, using
113,588 candidate rows. The original five-model baseline outputs are preserved;
the tuned model remains a separate artifact. No final deployment choice was
changed based on this comparison, and no additional search was run to chase the
student-testing score. HMM/Viterbi performance remains a separate experiment.

`results/tuning/lightgbm/comparison.csv` and `run_info.json` hold the actual
measurements. `validation.json` records independent checks of the complete
trial history, input/code hashes and saved-model predictions. The figure
`optimization_comparison.png` shows both search progression and the final trade-off.
