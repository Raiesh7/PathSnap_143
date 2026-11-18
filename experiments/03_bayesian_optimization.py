"""Tune LightGBM on training trips, then compare on student testing once.

python experiments/03_bayesian_optimization.py --trials 20
Use --resume to continue a study with matching inputs and configuration.
"""
import argparse
from pathlib import Path
import sys
import json
import time
import hashlib
import platform
import subprocess
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from src.features import FEATURE_COLUMNS
from src.training import load_partition, arc_endpoints, evaluate, TABLES, FOLD_FILE
from src.tuning import (SEARCH_SPACE, BASELINE_PARAMS, suggest_params, build_tuned_model,
                        grouped_fold_indices, flatten_trial_folds)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_search(study, output):
    study.trials_dataframe().to_csv(output/'trials.csv',index=False)
    flatten_trial_folds(study).to_csv(output/'trial_fold_metrics.csv',index=False)
    joblib.dump(study.sampler,output/'sampler_state.joblib')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=20,help='Total trials, including baseline and startup trials')
    parser.add_argument('--n-jobs',type=int,default=4,help='LightGBM threads; trials run sequentially')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/tuning/lightgbm')
    args = parser.parse_args()
    if args.trials < 6:
        parser.error('At least 6 trials are needed to progress beyond five startup trials')
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    # Deliberately load TRAIN only until search and best-parameter selection finish.
    train, observations = load_partition('train')
    assignment = pd.read_csv(FOLD_FILE,dtype={'trip_id':str})
    folds = grouped_fold_indices(train,assignment)
    X = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = train.label.to_numpy()
    endpoints = arc_endpoints('train',train.trip_id.unique())
    config = dict(seed=args.seed,n_jobs=args.n_jobs,features=FEATURE_COLUMNS,
        search_space=SEARCH_SPACE,objective='mean five-fold candidate binary Macro F1 at threshold 0.5',
        direction='maximize',sampler='TPESampler',startup_trials=5,
        pruning=False,early_stopping=False,cv_row_cap=None,
        fold_file_sha256=digest(FOLD_FILE),
        train_candidates_sha256=digest(TABLES/'train_candidates.parquet'),
        train_observations_sha256=digest(TABLES/'train_observations.parquet'),
        implementation_sha256={f:digest(ROOT/f) for f in ['src/tuning.py','src/features.py','src/metrics.py',
            'src/training.py','experiments/03_bayesian_optimization.py']},
        versions={p:importlib.metadata.version(p) for p in ['optuna','lightgbm','numpy','pandas','scikit-learn']})
    configuration_file = out/'search_config.json'
    if configuration_file.exists():
        if not args.resume:
            raise SystemExit('Study already exists; use --resume or a different --output-dir')
        if json.loads(configuration_file.read_text()) != config:
            raise SystemExit('Study inputs/configuration changed; use a different output directory')
    else:
        configuration_file.write_text(json.dumps(config,indent=2))
    sampler_path = out/'sampler_state.joblib'
    sampler = joblib.load(sampler_path) if args.resume and sampler_path.exists() else TPESampler(seed=args.seed,n_startup_trials=5)
    study = optuna.create_study(study_name='lightgbm_trip_grouped_macro_f1',direction='maximize',
        sampler=sampler,pruner=optuna.pruners.NopPruner(),
        storage='sqlite:///'+(out/'study.sqlite3').as_posix(),load_if_exists=args.resume)
    if not study.trials:
        study.enqueue_trial(BASELINE_PARAMS,user_attrs={'role':'untuned_baseline'})

    def objective(trial):
        params = suggest_params(trial)
        results = []
        trial_start = time.perf_counter()
        for k,(tr,va) in enumerate(folds):
            model = build_tuned_model(params,args.seed,args.n_jobs)
            fit_start = time.perf_counter()
            model.fit(X[tr],y[tr])
            fit_seconds = time.perf_counter()-fit_start
            val_trips = train.iloc[va].trip_id.unique()
            metrics,_ = evaluate(model,train.iloc[va],observations[observations.trip_id.isin(val_trips)],endpoints)
            metrics.pop('confusion')
            row = dict(fold=k,train_rows=len(tr),validation_rows=len(va),validation_trips=len(val_trips),
                       fit_s=fit_seconds,**metrics)
            results.append(row)
            trial.set_user_attr('fold_metrics',results)
            flatten_trial_folds(study).to_csv(out/'trial_fold_metrics.csv',index=False)
            print(f'Trial {trial.number+1}/{args.trials}, fold {k+1}/5: F1={metrics["f1_macro"]:.5f}, '
                  f'Top1={metrics["top1_accuracy"]:.5f}, fit={fit_seconds:.1f}s',flush=True)
        trial.set_user_attr('cv_f1_std',float(np.std([r['f1_macro'] for r in results],ddof=1)))
        trial.set_user_attr('cv_top1_mean',float(np.mean([r['top1_accuracy'] for r in results])))
        trial.set_user_attr('runtime_s',time.perf_counter()-trial_start)
        return float(np.mean([r['f1_macro'] for r in results]))

    remaining = max(0,args.trials-sum(t.state==TrialState.COMPLETE for t in study.trials))
    study.optimize(objective,n_trials=remaining,n_jobs=1,callbacks=[lambda s,t:save_search(s,out)])
    save_search(study,out)
    best = study.best_trial
    best_record = dict(trial=best.number,value=best.value,params=best.params,objective=config['objective'])
    (out/'best_params.json').write_text(json.dumps(best_record,indent=2))
    pd.DataFrame(best.user_attrs['fold_metrics']).to_csv(out/'best_cv_folds.csv',index=False)
    print(f'Search finished. Best trial {best.number}: CV Macro F1={best.value:.6f}',flush=True)
    # Parameters are now fixed. Student testing cannot influence the search.
    test,test_obs = load_partition('student_test')
    test_endpoints = arc_endpoints('student_test',test.trip_id.unique())
    results,comparison = {},[]
    for name,params,trial in [('untuned',BASELINE_PARAMS,study.trials[0]),('tuned',best.params,best)]:
        model = build_tuned_model(params,args.seed,args.n_jobs)
        fit_start = time.perf_counter()
        model.fit(X,y)
        fit_seconds = time.perf_counter()-fit_start
        metrics,scored = evaluate(model,test,test_obs,test_endpoints)
        metrics.update(fit_s_full=fit_seconds,train_rows_full=len(train))
        results[name] = metrics
        (out/f'{name}_student_test_metrics.json').write_text(json.dumps(metrics,indent=2))
        comparison.append(dict(model=name,cv_macro_f1=trial.value,
            cv_macro_f1_std=trial.user_attrs['cv_f1_std'],cv_top1_accuracy=trial.user_attrs['cv_top1_mean'],
            student_test_macro_f1=metrics['f1_macro'],student_test_top1_accuracy=metrics['top1_accuracy'],
            student_test_log_loss=metrics['log_loss'],route_continuity=metrics['route_continuity'],
            inference_ms_per_1000_obs=metrics['inference_ms_per_1000_obs'],fit_s_full=fit_seconds))
        if name == 'tuned':
            artifact = ROOT/'artifacts/models/lightgbm_tuned.joblib'
            artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,artifact,compress=3)
            scored[['trip_id','observation_id','arc_id','sequence_id','label','prob']].to_parquet(out/'student_test_predictions.parquet',index=False)
            pd.DataFrame({'feature':FEATURE_COLUMNS,'importance':model.booster_.feature_importance('gain')}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
            restored = joblib.load(artifact)
            ix = np.linspace(0,len(test)-1,min(1024,len(test)),dtype=int)
            probabilities = restored.predict_proba(test.iloc[ix][FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:,1]
            np.testing.assert_allclose(probabilities,scored.prob.iloc[ix],rtol=1e-12,atol=1e-12)
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    info = dict(completed_trials=sum(t.state==TrialState.COMPLETE for t in study.trials),
        best_trial=best.number,best_params=best.params,baseline_cv_f1=study.trials[0].value,best_cv_f1=best.value,
        resumed=args.resume,current_invocation_runtime_s=time.perf_counter()-started,
        total_trial_runtime_s=sum(t.user_attrs.get('runtime_s',0) for t in study.trials),
        train_trips=train.trip_id.nunique(),train_candidate_rows=len(train),
        student_test_label_valid_observations=int(test_obs.label_valid.sum()),
        student_test_total_observations=len(test_obs),student_testing_used_for_search=False,
        artifact_reload_verified=True,machine=platform.node(),python=platform.python_version(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tuned_artifact_sha256=digest(ROOT/'artifacts/models/lightgbm_tuned.joblib'),
        results=results)
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    plot_results(study,pd.DataFrame(comparison),out)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)


def plot_results(study,comparison,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed = [t for t in study.trials if t.state==TrialState.COMPLETE]
    values = np.asarray([t.value for t in completed])
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot([t.number+1 for t in completed],values,'o',label='Trial score')
    axes[0].plot([t.number+1 for t in completed],np.maximum.accumulate(values),label='Best so far')
    axes[0].set(xlabel='Trial number',ylabel='Mean five-fold Macro F1',title='Training-only Bayesian search')
    axes[0].legend()
    x = np.arange(2)
    axes[1].bar(x-.18,comparison.student_test_top1_accuracy,.36,label='Top-1 accuracy')
    axes[1].bar(x+.18,comparison.student_test_macro_f1,.36,label='Candidate Macro F1')
    axes[1].set(xticks=x,xticklabels=comparison.model,ylim=(0.75,1),title='Student-testing comparison')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out/'optimization_comparison.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

