"""Independent fit/tune/activation/test benchmark; resumable raw predictions.

Use --pilot for the predeclared diagnostic (not paper-protocol evidence).
No planning values are read. Failed baselines remain explicit errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party')]
from src.graphcal.activation import activation_decision
from src.graphcal.augmenter import GraphCalAugmenter, voronoi_indices
from src.graphcal.benchmark_data import MAIN_DATASETS,load_revision_dataset
from src.graphcal.metrics import evaluate_frozen_regions
from src.graphcal.preprocessing import TrainingEncoder
from src.graphcal.protocol import PAIRED_SETTINGS, comparison_family_size
from src.graphcal.revision_baselines import cems_reference,smogn_reference,smoter
from src.graphcal.revision_models import fit_mlp

METHODS=('vanilla','smoter','smogn','lds','balanced_mse','conr','prime','cems','graphcal')


def graphcal_augmenter(args, *, kappa: float, seed: int) -> GraphCalAugmenter:
    return GraphCalAugmenter(
        kappa=kappa,
        random_state=seed,
        distance_scale=args.distance_scale,
        residual_screen=args.residual_screen,
        delta=args.graphcal_delta,
        gamma=args.graphcal_gamma,
        ridge_residual=args.graphcal_ridge_residual,
        ridge_alpha=args.graphcal_ridge_alpha,
        residual_sigma_mult=args.graphcal_residual_sigma,
    )


def partition(n):
    remaining,test=train_test_split(np.arange(n),test_size=.3,random_state=20261024)
    fitting_tuning,activation=train_test_split(remaining,test_size=1/7,random_state=20261025)
    fitting,tuning=train_test_split(fitting_tuning,test_size=1/6,random_state=20261026)
    parts={'fitting':fitting,'tuning':tuning,'activation':activation,'test':test}
    if len(np.unique(np.concatenate(list(parts.values()))))!=n:raise AssertionError('Partition overlap')
    return parts


def atomic_json(path,value):
    temporary=path.with_suffix('.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def evaluate_dataset(name,args,out):
    frame,y,meta=load_revision_dataset(name)
    parts=partition(len(y))
    if args.pilot:
        # Develop only inside the flagship fitting pool; its activation/test
        # slices must remain outside development decisions.
        pool=parts['fitting']
        parts={key:pool[value] for key,value in partition(len(pool)).items()}
    fit,tune,act,test=[parts[k] for k in ('fitting','tuning','activation','test')]
    if args.pilot:
        rng=np.random.default_rng(9001)
        fit=np.sort(rng.choice(fit,min(len(fit),800),replace=False))
    encoder=TrainingEncoder(meta['categorical'])
    # Fitting labels only. Activation and test labels stay unread until every finalist is frozen.
    xf=encoder.fit_transform(frame.iloc[fit],y[fit])
    xt,xa,xe=[encoder.transform(frame.iloc[idx]) for idx in (tune,act,test)]
    yf,yt=y[fit],y[tune]
    mcap=max(3*float(np.std(yf)),1e-8)
    split_path=out/(name+'_partition.npz')
    np.savez_compressed(split_path,**parts,effective_fitting=fit)
    source_hash=hashlib.sha256(split_path.read_bytes()).hexdigest()
    for seed in args.seeds:
        # Geometry is frozen from fitting rows. The mask is applied after labels open.
        region=graphcal_augmenter(args, kappa=PAIRED_SETTINGS[0]['kappa'], seed=seed)
        region_result=region.fit_transform(xf,yf)
        frozen={}
        for method in args.methods:
            checkpoint=out/f'{name}_{seed}_{method}_frozen'
            if checkpoint.with_suffix('.npz').exists() and checkpoint.with_suffix('.json').exists():
                metadata=json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8'))
                with np.load(checkpoint.with_suffix('.npz')) as saved:
                    frozen[method]={**metadata,**{key:saved[key] for key in saved.files}}
                print(f'{name} seed={seed} {method}: resumed frozen predictions',flush=True)
                continue
            start=perf_counter();best=None;best_mae=float('inf');diagnostic={}
            try:
                for configuration,setting in enumerate(PAIRED_SETTINGS):
                    objective=method if method in ('lds','balanced_mse','conr','prime') else 'mae'
                    if method=='graphcal':
                        if configuration==0:aug=region_result
                        else:aug=graphcal_augmenter(args, kappa=setting['kappa'], seed=seed).fit_transform(xf,yf)
                        ax,ay,diag=aug.X_aug,aug.y_aug,aug.diagnostics
                    elif method=='smoter':ax,ay,diag=smoter(xf,yf,seed)
                    elif method=='smogn':ax,ay,diag=smogn_reference(xf,yf,seed)
                    elif method=='cems':ax,ay,diag=cems_reference(xf,yf,seed,configuration)
                    else:ax,ay,diag=xf,yf,{}
                    model=fit_mlp(ax,ay,xt,yt,seed=seed,objective=objective,
                        configuration=configuration,learning_rate=setting['learning_rate'],max_epochs=args.epochs)
                    score=float(np.mean(np.abs(model.predict(xt)-yt)))
                    if score<best_mae:
                        best,best_mae=model,score
                        diagnostic={'configuration':configuration,'augmentation':diag,'training':model.history}
                frozen[method]={'activation_prediction':best.predict(xa),'test_prediction':best.predict(xe),
                    'tuning_mae':best_mae,'diagnostics':diagnostic,'seconds':perf_counter()-start}
                np.savez_compressed(checkpoint.with_suffix('.npz'),
                    activation_prediction=frozen[method]['activation_prediction'],
                    test_prediction=frozen[method]['test_prediction'])
                atomic_json(checkpoint.with_suffix('.json'),{key:value for key,value in frozen[method].items()
                    if key not in ('activation_prediction','test_prediction')})
                print(f'{name} seed={seed} {method} frozen; tuning MAE={best_mae:.6g}',flush=True)
            except Exception as exc:
                atomic_json(out/f'{name}_{seed}_{method}_error.json',{'error':repr(exc),'method':method,'dataset':name,'seed':seed})
                raise
        # Labels open only after every finalist has been fitted, tuned, and stored.
        # Nothing below refits a predictor.
        ya,ye=y[act],y[test]
        ztest=region.encode_joint(xe,ye)
        nearest=voronoi_indices(ztest,region.reference_z_)
        sparse=np.isin(nearest,region.sparse_indices_)
        reference=frozen['vanilla']
        family=comparison_family_size(args.methods)
        for method,model in frozen.items():
            baseline=method=='vanilla'
            decision=None if baseline else activation_decision(ya,reference['activation_prediction'],
                model['activation_prediction'],loss_cap=mcap,family_size=family)
            raw=model['test_prediction']
            empirically_on=baseline or decision.empirical_difference<0
            certified_on=baseline or decision.activated
            predictions={'ungated':raw,'empirical_gate':raw if empirically_on else reference['test_prediction'],
                         'certified_gate':raw if certified_on else reference['test_prediction']}
            row={'dataset':name,'seed':seed,'method':method,'pilot':args.pilot,'epochs':args.epochs,
                 'partition_sha256':source_hash,'sizes':{k:len(v) for k,v in parts.items()},
                 'effective_fitting_n':len(fit),'loss_cap':mcap,'activation':None if baseline else decision.to_dict(),
                 'tuning_mae':model['tuning_mae'],'seconds':model['seconds'],'diagnostics':model['diagnostics'],
                 'metrics':{mode:evaluate_frozen_regions(ye,pred,training_targets=yf,
                    sparse_mask=sparse,loss_cap=mcap) for mode,pred in predictions.items()}}
            stem=f'{name}_{seed}_{method}'
            np.savez_compressed(out/(stem+'.npz'),y_test=ye,test_indices=test,sparse_mask=sparse,
                y_activation=ya,activation_prediction=model['activation_prediction'],**predictions)
            atomic_json(out/(stem+'.json'),row)
        atomic_json(out/'status.json',{'last_completed':name,'seed':seed,'methods':list(frozen),'pilot':args.pilot})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=list(MAIN_DATASETS))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,456,789,2026])
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--distance-scale',choices=('global','local'),default='global')
    p.add_argument('--residual-screen',dest='residual_screen',action='store_true',default=True)
    p.add_argument('--no-residual-screen',dest='residual_screen',action='store_false')
    p.add_argument('--graphcal-delta',type=float,default=2.0)
    p.add_argument('--graphcal-gamma',type=float,default=1.5)
    p.add_argument('--graphcal-ridge-residual',action='store_true')
    p.add_argument('--graphcal-ridge-alpha',type=float,default=1.0)
    p.add_argument('--graphcal-residual-sigma',type=float,default=3.0)
    args=p.parse_args()
    if 'vanilla' not in args.methods:p.error('All comparisons require the common unaugmented reference')
    if len(set(args.methods))!=len(args.methods):p.error('Duplicate methods are not allowed')
    if len(PAIRED_SETTINGS)!=2:p.error('The paper protocol has two paired settings, not a grid')
    if [s['learning_rate'] for s in PAIRED_SETTINGS]!=[1e-3,3e-3]:p.error('Learning rates must stay paired at 1e-3 and 3e-3')
    if [s['kappa'] for s in PAIRED_SETTINGS]!=[2.0,1.0]:p.error('GraphCal kappa must stay paired as 2 then 1')
    source_paths=sorted([*ROOT.joinpath('src','graphcal').glob('*.py'),
                         *ROOT.joinpath('third_party','cems').glob('*.py'),
                         *ROOT.joinpath('third_party','smogn').glob('*.py'),Path(__file__)])
    fingerprint=hashlib.sha256(json.dumps(vars(args),sort_keys=True).encode())
    for source in source_paths:
        fingerprint.update(source.relative_to(ROOT).as_posix().encode());fingerprint.update(source.read_bytes())
    fingerprint.update((ROOT/'results'/'revision_2026-10-24'/'dataset_selection.json').read_bytes())
    run_id=fingerprint.hexdigest()[:16]
    out=ROOT/'results'/'revision_2026-10-24'/('pilot' if args.pilot else 'flagship')/run_id
    out.mkdir(parents=True,exist_ok=True)
    print(f'Run directory: {out}',flush=True)
    atomic_json(out/'protocol.json',{'arguments':vars(args),'partition_seed':20261024,
        'headline':'overall uncapped test MAE','secondary':['upper/lower training-quantile tail MAE','joint sparse MAE','ECDF SERA'],
        'paired_settings':[dict(s) for s in PAIRED_SETTINGS],
        'certificate':'clipped absolute loss only; fixed M=3 fitting target SD; eta=0.05 simultaneous over the K non-reference finalists of this dataset and seed, not across datasets or seeds',
        'rank_bar':'top two or mean rank <=2.5 on primary metric, secondary rank also reported; duplicate fallback rows excluded',
        'no_estimates':True})
    os.environ['LOKY_MAX_CPU_COUNT']='2';torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for name in args.datasets:evaluate_dataset(name,args,out)


if __name__=='__main__':main()
