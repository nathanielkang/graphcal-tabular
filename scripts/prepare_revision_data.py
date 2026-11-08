"""Prepare raw, pinned datasets without any model fitting or performance screen."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.datasets import fetch_california_housing, fetch_openml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
CACHE = ROOT/'data'/'revision_cache'
sys.path.insert(0,str(ROOT))
from src.graphcal.benchmark_data import MAIN_DATASETS, RESERVE_DATASETS


def save(name, frame, target, source, categorical=(), exclusions=()):
    frame=frame.copy()
    frame=frame.drop(columns=[c for c in frame if c in exclusions or str(c).lower().startswith(('unnamed:','rownames'))])
    y=pd.to_numeric(frame[target],errors='coerce')
    frame=frame.loc[np.isfinite(y)].reset_index(drop=True)
    frame[target]=y.loc[np.isfinite(y)].to_numpy(dtype=float)
    path=CACHE/(name+'.pkl');frame.to_pickle(path)
    meta={'dataset':name,'target':target,'source':source,'n':len(frame),
          'd':len(frame.columns)-1,'categorical':list(categorical),
          'excluded_columns':list(exclusions),'skewness':float(skew(frame[target])),
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (CACHE/(name+'_meta.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(name,meta['n'],meta['d'],target,flush=True)
    return meta


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    specs=[('abalone',183,('Sex',),()),('bike',44063,('season','year','month','hour','holiday','weekday','workingday','weather'),()),
           ('wine',287,(),()),('energy',1472,('V6','V8'),('y2',)),
           ('superconductor',44964,(),()),('cpu',573,(),())]
    metadata={}
    for name,data_id,cats,exclusions in specs:
        cached=CACHE/(name+'_openml.pkl')
        if cached.exists():
            frame=pd.read_pickle(cached)
            old=json.loads((CACHE/(name+'_openml_meta.json')).read_text(encoding='utf-8'))
            target=old['target'];source={'provider':'OpenML','data_id':data_id,'name':old['name']}
        else:
            data=fetch_openml(data_id=data_id,as_frame=True)
            frame=data.frame;target=data.target.name
            source={'provider':'OpenML','data_id':data_id,'name':data.details['name']}
        cats=[c for c in cats if c in frame and c not in exclusions and c != target]
        cats+= [c for c in frame if c not in exclusions and c!=target and str(frame[c].dtype)=='category' and c not in cats]
        metadata[name]=save(name,frame,target,source,cats,exclusions)
    california=fetch_california_housing(as_frame=True)
    metadata['california']=save('california',california.frame,california.target.name,{'provider':'sklearn California housing','target':'MedHouseVal'})
    shared=WORKSPACE/'6_2026_Spring_Conference/7_ICML_2026/3_Code/Datasets'
    frame=pd.read_csv(shared/'forestFires_clean.csv')
    metadata['forest_fires']=save('forest_fires',frame,'Area',{'provider':'UCI','dataset':'Forest fires','local_source':str(shared/'forestFires_clean.csv')},('Month','Day'))
    dat=WORKSPACE/'6_2026_Spring_Conference/8_SAC_2027/E_RareFed/2_Code/data_cache/airfoil_self_noise.dat'
    frame=pd.read_csv(dat,sep=r'\s+',header=None,names=['frequency','angle','chord','velocity','displacement','sound'])
    metadata['airfoil']=save('airfoil',frame,'sound',{'provider':'UCI','dataset':'Airfoil self noise','local_source':str(dat)})
    for name,file,target in [('concrete','concreteStrength.csv','target'),('yacht','yacht.csv','Residuary.resistance'),('auto_mpg','auto_mpg.csv','class')]:
        metadata[name]=save(name,pd.read_csv(shared/file),target,{'provider':'shared public benchmark cache','file':file})
    selection={'protocol':'data-only fixed priority: retain all nine reviewed dataset identities; three reserves do not replace losing datasets',
               'n_select_main':9,'pool':list(MAIN_DATASETS+RESERVE_DATASETS),
               'selected_main':list(MAIN_DATASETS),'reserve':list(RESERVE_DATASETS),
               'performance_used_for_selection':False,'datasets':metadata}
    output=ROOT/'results'/'revision_2026-10-24';output.mkdir(exist_ok=True,parents=True)
    (output/'dataset_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')


if __name__=='__main__':main()
