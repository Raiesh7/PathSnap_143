import unittest
import numpy as np
import pandas as pd
import optuna
from src.tuning import BASELINE_PARAMS,suggest_params,build_tuned_model,grouped_fold_indices


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame({'trip_id':np.repeat(['a','b','c','d','e'],2),'label':[0,1]*5})
        self.assignment = pd.DataFrame({'trip_id':list('abcde'),'fold':range(5)})

    def test_folds_keep_whole_trips_and_cover_rows_once(self):
        splits = grouped_fold_indices(self.candidates,self.assignment)
        seen = []
        for tr,va in splits:
            self.assertFalse(set(self.candidates.iloc[tr].trip_id)&set(self.candidates.iloc[va].trip_id))
            seen.extend(va)
        self.assertEqual(sorted(seen),list(range(10)))

    def test_missing_duplicate_and_fractional_assignments_rejected(self):
        bad = [self.assignment.iloc[:-1],pd.concat([self.assignment,self.assignment.iloc[:1]]),
               self.assignment.assign(fold=[0,1,2,3,3.5])]
        for assignment in bad:
            with self.assertRaises(ValueError):
                grouped_fold_indices(self.candidates,assignment)

    def test_baseline_is_inside_search_and_reproduces_configuration(self):
        params = suggest_params(optuna.trial.FixedTrial(BASELINE_PARAMS))
        self.assertEqual(params,BASELINE_PARAMS)
        model = build_tuned_model(params,n_jobs=1)
        self.assertEqual(model.n_estimators,400)
        self.assertEqual(model.subsample_freq,1)


if __name__=='__main__':
    unittest.main()
