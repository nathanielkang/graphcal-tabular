from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    if np.sum(mask) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_frozen_regions(y_true, y_pred, *, training_targets,
                            sparse_mask=None, loss_cap=None):
    """Region cutoffs and relevance distribution use training targets only."""
    y=np.asarray(y_true,dtype=float).reshape(-1)
    p=np.asarray(y_pred,dtype=float).reshape(-1)
    train=np.sort(np.asarray(training_targets,dtype=float).reshape(-1))
    if y.shape!=p.shape or not y.size or not np.isfinite(np.r_[y,p,train]).all():
        raise ValueError('Evaluation requires finite, paired observations')
    absolute=np.abs(y-p)
    low,high=np.quantile(train,[0.05,0.95])
    masks={'upper_tail':y>=high,'lower_tail':y<=low}
    if sparse_mask is not None:masks['joint_sparse']=np.asarray(sparse_mask,dtype=bool)
    out={'mae':float(absolute.mean()),'mse':float(np.mean((y-p)**2)),
         'n_test':int(len(y)),'upper_cutoff':float(high),'lower_cutoff':float(low)}
    # Integral_0^1 E[e^2 1{phi>=t}]dt = E[e^2 phi], with frozen ECDF relevance.
    cdf=np.searchsorted(train,y,side='right')/len(train)
    relevance=np.abs(2*cdf-1)
    out['sera_ecdf']=float(np.mean((y-p)**2*relevance))
    if loss_cap is not None:out['capped_mae']=float(np.minimum(absolute,loss_cap).mean())
    for name,mask in masks.items():
        out[name+'_n']=int(mask.sum())
        out[name+'_mae']=float(absolute[mask].mean()) if mask.any() else None
    if sparse_mask is not None:
        out['upper_tail_sparse_overlap_n']=int((masks['upper_tail']&masks['joint_sparse']).sum())
    return out
