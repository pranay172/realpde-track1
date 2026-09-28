"""Group-disjoint selection from additive, exact-SPS candidate statistics."""
import numpy as np


def select_crossfit(rewards, features, groups, conditional=False):
    """Return N x cells candidate indices; held-group rewards never select them."""
    rewards=np.asarray(rewards)
    features=np.asarray(features)
    groups=np.asarray(groups)
    if rewards.ndim != 3 or len(features) != len(rewards) or len(groups) != len(rewards):
        raise ValueError("incompatible candidate statistics")
    if not np.isfinite(rewards).all() or not np.isfinite(features).all():
        raise ValueError("nonfinite candidate statistics")
    if len(np.unique(groups)) < 2:
        raise ValueError("need multiple disjoint groups")
    chosen=np.zeros(rewards.shape[:2],dtype=int)
    fits=[]
    for group in np.unique(groups):
        train=groups != group
        test=~train
        threshold=float(np.median(features[train])) if conditional else None
        bins=(features > threshold).astype(int) if conditional else np.zeros(len(groups),dtype=int)
        fit={"held_group":int(group),"calibration_groups":[int(x) for x in np.unique(groups[train])],
             "threshold":threshold,"choices":{}}
        for bucket in range(2 if conditional else 1):
            subset=train & (bins == bucket)
            # No support: unchanged baseline (candidate zero), never test fitting.
            pick=rewards[subset].sum(axis=0).argmax(axis=-1) if subset.any() else np.zeros(rewards.shape[1],dtype=int)
            chosen[test & (bins == bucket)]=pick
            fit["choices"][str(bucket)]=pick.tolist()
        fits.append(fit)
    return chosen,fits


def gather_candidates(values, chosen):
    return np.take_along_axis(values,chosen[...,None],axis=-1)[...,0]
