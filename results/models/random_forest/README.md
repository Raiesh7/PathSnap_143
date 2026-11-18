# Random Forest training results

Command: python experiments/02_train_random_forest.py

Configuration: n_estimators=150, min_samples_leaf=20, max_features=sqrt, max_samples=0.3. Five shared trip-grouped folds, seed 42, no CV row cap. Each fold holds out 16 of 80 training trips. The final model was refitted on all 1,058,904 label-valid candidate rows. Each tree uses its configured bootstrap sample; this is not a training-table row cap.

| Metric | Five-fold mean | Student testing |
|---|---:|---:|
| Candidate binary Macro F1 | 0.928185 | 0.926726 |
| Observation Top-1 accuracy | 0.881784 | 0.879737 |
| Candidate log loss | 0.106161 | 0.102361 |
| Direct arc continuity | 0.943023 | 0.940039 |

Student testing uses 113,588 candidate rows and 19,133 label-valid observations from 10 trips. The label-validity policy excludes 27 of 19,160 observations. Scores use reference-derived alignment labels, not independent pointwise GPS ground truth. Candidate misses are counted as Top-1 errors.

These are untuned classifier-only results before HMM/Viterbi. Report continuity as the existing direct-arc diagnostic; it is not a full network-path accuracy measure.

The saved feature-importance CSV describes original features only. The estimator also uses missing-value indicator columns; those additional importances are not included in that CSV.

Outputs: cv_folds.csv, student_test_metrics.json, student_test_predictions.parquet, feature_importance.csv, run_info.json, training_provenance.json and validation.json. Trained model: artifacts/models/random_forest.joblib.

Verification reloaded the saved model, reproduced all student-testing probabilities, and recomputed Macro F1 and Top-1 accuracy. All five folds finished.
