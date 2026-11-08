"""Shared-backbone tabular ports of published regression objectives.

PRIME equations (1)--(8), ICML 2025; ConR equations (1)--(4), ICLR 2024;
Balanced MSE's batch Monte Carlo loss, CVPR 2022; LDS, ICML 2021.
These are declared tabular ports, not the original image encoders.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler


def density_weights(y):
    counts,edges=np.histogram(y,bins=50)
    smooth=gaussian_filter1d(counts.astype(float),sigma=2,mode='constant')
    idx=np.clip(np.searchsorted(edges,y,side='right')-1,0,len(counts)-1)
    weights=1/np.maximum(smooth[idx],1)
    return weights/weights.mean()


class Network(nn.Module):
    def __init__(self,d,proxy_targets):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.head=nn.Linear(128,1)
        self.proxies=nn.Parameter(torch.randn(20,128)*.1)
        self.register_buffer('proxy_targets',proxy_targets)

    def forward(self,x):
        z=self.encoder(x)
        return self.head(z).squeeze(1),z


def prime_objective(z,y,network):
    proxies=network.proxies;targets=network.proxy_targets
    eye=torch.eye(len(proxies),device=z.device,dtype=torch.bool)
    target_distance=torch.cdist(targets[:,None],targets[:,None])
    feature_distance=torch.cdist(proxies,proxies)
    lp=F.log_softmax((-target_distance)[~eye],dim=0)
    lq=F.log_softmax((-feature_distance)[~eye],dim=0)
    divergence=(lp.exp()*(lp-lq)).sum()
    cosine=F.normalize(proxies,dim=1)@F.normalize(proxies,dim=1).T
    spreading=(.001*target_distance*(1-cosine).square())[~eye].sum()
    target_assoc=F.softmax(-torch.abs(y[:,None]-targets[None,:]),dim=1)
    log_assoc=F.log_softmax(-torch.cdist(z,proxies),dim=1)
    alignment=-(target_assoc*log_assoc).sum(1).mean()
    return divergence-spreading+alignment


def conr_objective(z,y,pred,weights):
    # Two independent numeric views replace image transformations in this port.
    label_dist=torch.abs(y[:,None]-y[None,:])
    pred_dist=torch.abs(pred.detach()[:,None]-pred.detach()[None,:])
    positive=label_dist<=.5
    positive.fill_diagonal_(False)
    negative=(label_dist>.5)&(pred_dist<=.5)
    valid=positive.any(1)&negative.any(1)
    if not valid.any():return z.sum()*0
    logits=F.normalize(z,dim=1)@F.normalize(z,dim=1).T/.07
    log_push=torch.log(weights.clamp_min(1e-8))[:,None]+.01*label_dist
    denominator=torch.logsumexp(torch.where(positive,logits,
        torch.where(negative,logits+log_push,torch.full_like(logits,-torch.inf))),dim=1)
    # Average the log-probability of each positive, rather than taking the
    # log of their summed probability. These are different objectives.
    positive_mean=(logits*positive).sum(1)/positive.sum(1).clamp_min(1)
    loss=torch.zeros_like(y)
    loss[valid]=denominator[valid]-positive_mean[valid]
    return (weights*loss).mean()


class FittedMLP:
    def __init__(self,network,scaler,y_mean,y_scale,history):
        self.network,self.scaler=network,scaler
        self.y_mean,self.y_scale,self.history=y_mean,y_scale,history

    def predict(self,x):
        self.network.eval()
        with torch.no_grad():
            tx=torch.as_tensor(self.scaler.transform(x),dtype=torch.float32)
            out=[]
            for chunk in tx.split(2048):out.append(self.network(chunk)[0].numpy())
        return np.concatenate(out)*self.y_scale+self.y_mean


def fit_mlp(x,y,x_tune,y_tune,*,seed,objective='mae',configuration=0,learning_rate=None,max_epochs=200):
    if configuration not in (0, 1):
        raise ValueError('Only the two paired configurations are defined')
    if learning_rate is None:
        learning_rate = (1e-3, 3e-3)[configuration]
    torch.manual_seed(seed)
    scaler=StandardScaler().fit(x)
    tx=torch.tensor(scaler.transform(x),dtype=torch.float32)
    tt=torch.tensor(scaler.transform(x_tune),dtype=torch.float32)
    ym=float(np.mean(y));ys=max(float(np.std(y)),1e-8)
    ty=torch.tensor((np.asarray(y)-ym)/ys,dtype=torch.float32)
    targets=torch.linspace(float(ty.min()),float(ty.max()),22)[1:-1]
    net=Network(x.shape[1],targets)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)
    weights=torch.tensor(density_weights(np.asarray(y)),dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed+7919)
    best_loss=float('inf');best=None;stale=0;history=[]
    for epoch in range(max_epochs):
        net.train();total=0.
        for indices in torch.randperm(len(ty),generator=generator).split(128):
            xb,yb,wb=tx[indices],ty[indices],weights[indices]
            pred,z=net(xb)
            if objective=='balanced_mse':
                variance=(.5,1.0)[configuration]**2
                logits=-(pred[:,None]-yb[None,:]).square()/(2*variance)
                loss=F.cross_entropy(logits,torch.arange(len(yb)))*2*variance
            elif objective=='lds':loss=(wb*torch.abs(pred-yb)).mean()
            else:
                loss=F.l1_loss(pred,yb)
                if objective=='prime':loss=loss+(.1,1.)[configuration]*prime_objective(z,yb,net)
                elif objective=='conr':
                    # Duplicate labels accompany small jittered feature views.
                    aug=torch.cat((xb,xb+.01*torch.randn_like(xb)))
                    p2,z2=net(aug)
                    loss=loss+(.01,.1)[configuration]*conr_objective(z2,yb.repeat(2),p2,wb.repeat(2))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        net.eval()
        with torch.no_grad():tune_prediction=net(tt)[0].numpy()*ys+ym
        tune_loss=float(np.mean(np.abs(tune_prediction-y_tune)))
        history.append({'epoch':epoch+1,'training_loss':total/len(ty),'tuning_mae':tune_loss})
        if tune_loss<best_loss:
            best_loss=tune_loss;best=copy.deepcopy(net.state_dict());stale=0
        else:stale+=1
        if stale>=20:break
    net.load_state_dict(best)
    return FittedMLP(net,scaler,ym,ys,history)
