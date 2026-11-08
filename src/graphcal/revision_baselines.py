"""Training-only augmenters. Reference CEMS and SMOGN are vendored with licenses."""
from __future__ import annotations
import random
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def smoter(x,y,seed):
    """Declared oversampling-only SMOTER tabular port; no undersampling."""
    rng=np.random.default_rng(seed)
    lower,upper=np.quantile(y,[.1,.9]);rare=np.flatnonzero((y<=lower)|(y>=upper))
    scaler=StandardScaler().fit(x)
    z=scaler.transform(x)
    k=min(6,len(rare))
    neighbors=NearestNeighbors(n_neighbors=k).fit(z[rare]).kneighbors(z[rare],return_distance=False)
    count=int(np.ceil(.3*len(y)))
    source=rng.integers(len(rare),size=count)
    donor=np.array([rng.choice(neighbors[i][neighbors[i]!=i]) if k>1 else i for i in source])
    i,j=rare[source],rare[donor];lam=rng.uniform(size=count)
    xs=x[i]+lam[:,None]*(x[j]-x[i])
    # Inverse-distance target interpolation, identical convex weights here.
    ys=y[i]+lam*(y[j]-y[i])
    return np.vstack((x,xs)),np.r_[y,ys],{'generated':count,'adaptation':'oversampling only, two training-decile relevance regions'}


def smogn_reference(x,y,seed):
    import smogn
    random.seed(seed);np.random.seed(seed)
    xs=StandardScaler().fit(x);ys=StandardScaler().fit(np.asarray(y)[:,None])
    frame=pd.DataFrame(xs.transform(x),columns=[f'x{i}' for i in range(x.shape[1])]);frame['target']=ys.transform(np.asarray(y)[:,None]).ravel()
    # Auto boxplot relevance is identically zero for several bounded targets.
    # Fix the same fitting-only decile/median relevance construction for all
    # datasets before any comparison; never change it in response to test error.
    low,center,high=np.quantile(frame['target'],[.1,.5,.9])
    if not low<center<high:
        low,center,high=float(frame['target'].min()),float(frame['target'].median()),float(frame['target'].max())
    if not low<center<high:
        return x.copy(),y.copy(),{'reference':'smogn 0.1.2','generated':0,'reason':'degenerate relevance knots'}
    out=smogn.smoter(data=frame,y='target',k=min(5,len(y)-1),samp_method='balance',
                     rel_method='manual',rel_ctrl_pts_rg=[[low,1,0],[center,0,0],[high,1,0]],under_samp=False)
    return xs.inverse_transform(out.drop(columns='target').to_numpy(float)),ys.inverse_transform(out[['target']].to_numpy(float)).ravel(),{
        'reference':'smogn 0.1.2 with documented numerical corrections','generated':len(out)-len(y),'relevance_knots_standardized':[low,center,high],
        'adaptation':'fitting-only scaling and decile/median relevance; no undersampling; see third_party/PATCHES.md'}


def cems_reference(x,y,seed,configuration=0):
    from third_party.cems import get_batch_cems
    from third_party.cems.intrinsic_dimension import intrinsic_dimension
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    scaler=StandardScaler().fit(np.column_stack((x,y)))
    z=scaler.transform(np.column_stack((x,y)))
    sample=z[rng.choice(len(z),size=min(800,len(z)),replace=False)]
    dimension=int(np.clip(round(intrinsic_dimension(sample)),1,min(z.shape[1]-1,10)))
    k=min(max(30,2*(dimension+dimension*(dimension+1)//2)),len(z)-1)
    anchors=rng.integers(len(z),size=int(np.ceil(.3*len(z))))
    neighbor=NearestNeighbors(n_neighbors=k).fit(z)
    args=SimpleNamespace(id=dimension,cems_method=3,sigma=(.1,.25)[configuration])
    result=[]
    for batch in np.array_split(anchors,max(1,int(np.ceil(len(anchors)/64)))):
        idx=neighbor.kneighbors(z[batch],return_distance=False)
        bx=torch.tensor(z[batch,:-1],dtype=torch.float64)
        by=torch.tensor(z[batch,-1:],dtype=torch.float64)
        nx=torch.tensor(z[idx,:-1],dtype=torch.float64)
        ny=torch.tensor(z[idx,-1:],dtype=torch.float64)
        with torch.no_grad():xs,ys=get_batch_cems(args,bx,by,nx,ny)
        result.append(np.column_stack((xs.numpy(),ys.numpy())))
    synthetic=scaler.inverse_transform(np.vstack(result))
    if not np.isfinite(synthetic).all():raise FloatingPointError('CEMS produced non-finite rows')
    return np.vstack((x,synthetic[:,:-1])),np.r_[y,synthetic[:,-1]],{'reference':'CEMS official source f8b9301; relative imports, CPU SVD dispatch, dtype-preserving tangent noise','dimension':dimension,'generated':len(synthetic)}
