#!/usr/bin/env python3
"""CPU-only exploratory interval screen; leaves saved point predictions unchanged."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/"src"),str(REPO/"realpde_t1_starting_kit_v9")]
import scoring
from realpde_t1.interval_crossfit import select_crossfit, gather_candidates
from realpde_t1.validation import persist_first_frames
from realpde_t1.uncertainty import interval_bounds_numpy
from e003_train_cno import sha256_file


def analyze(entry, config, return_statistics=False):
    folder=REPO/entry["artifacts"]
    cfg_path=REPO/entry["config"]
    cfg=json.loads(cfg_path.read_text())
    split_path=REPO/cfg["split_config"]
    split=json.loads(split_path.read_text())
    evaluation=json.loads((folder/"evaluation.json").read_text())
    pred_path=folder/"predictions.npy"
    checkpoint_path=folder/"final.pth"
    for key,path in [("config_sha256",cfg_path),("prediction_sha256",pred_path),("checkpoint_sha256",checkpoint_path)]:
        if evaluation[key] != sha256_file(path):
            raise ValueError(f"stale {entry['name']} {key}")
    ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    prov=ckpt["provenance"]
    if set(prov["training_files"]) != set(split["training_files"]) or prov["validation_field_values_accessed"] != []:
        raise ValueError("incompatible point-model provenance")
    del ckpt
    arrays=REPO/cfg["evaluation"]["arrays"]
    windows=json.loads((arrays/"windows.json").read_text())
    files={w["file"] for w in windows}
    if files != set(split["validation_files"]) or files & set(split["training_files"]):
        raise ValueError("holdout ancestry mismatch")
    groups=np.array([w["nominal_re"] for w in windows])
    pred=np.load(pred_path,mmap_mode="r")
    inputs=np.load(arrays/"inputs.npy",mmap_mode="r")
    target=np.load(arrays/"targets.npy",mmap_mode="r")
    if pred.shape != target.shape or inputs.shape != pred.shape or len(pred) != len(windows):
        raise ValueError("array alignment mismatch")
    multipliers=config["multipliers"]
    horizons=config["horizons"]
    shape=(len(pred),2*len(horizons),len(multipliers))
    reward=np.zeros(shape); covered=np.zeros(shape); widths=np.zeros(shape)
    counts=np.zeros(shape[:2]); features=np.zeros(len(pred))
    official_sum=0.
    for start in range(0,len(pred),8):
        end=min(start+8,len(pred)); sl=slice(start,end)
        x=np.asarray(inputs[sl]); t=np.asarray(target[sl]); p=persist_first_frames(pred[sl],x,4)
        if not all(np.isfinite(a).all() for a in (x,t,p)):
            raise ValueError("nonfinite arrays")
        features[sl]=np.std(x[...,:2],axis=1).mean(axis=(1,2,3))
        errors=np.stack([scoring.rel_l2_per_sample(p,t,2),scoring.tke_rel_l2_per_sample(p,t,2),scoring.mvpe_rel_l2_per_sample(p,t)],axis=1)
        accuracy=1-errors/(.5+errors)
        q=np.where(np.isfinite(accuracy),accuracy,0) @ np.array([.5,.3,.2])
        lo,hi=interval_bounds_numpy(p,.025,.15,scoring.SIGMA_GLOBAL)
        official,_=scoring.aggregate_sps(p,t,2,lo,hi)
        official_sum+=official*np.count_nonzero(t[...,:2])
        base_half=np.float32(.025)*np.abs(p)+np.float32(.15*scoring.SIGMA_GLOBAL)
        for h,(a,b) in enumerate(horizons):
            for c in range(2):
                cell=h*2+c; truth=t[:,a:b,:,:,c]; point=p[:,a:b,:,:,c]
                mask=truth!=0; axes=(1,2,3)
                counts[sl,cell]=mask.sum(axis=axes)
                for k,mult in enumerate(multipliers):
                    half=base_half[:,a:b,:,:,c]*np.float32(mult)
                    lower=point-half; upper=point+half
                    inside=(truth>=lower)&(truth<=upper)&mask
                    width=upper-lower
                    reward[sl,cell,k]=np.where(inside,np.exp(-width/scoring.SIGMA_GLOBAL),0).sum(axis=axes,dtype=np.float64)*q
                    covered[sl,cell,k]=inside.sum(axis=axes)
                    widths[sl,cell,k]=np.where(mask,width,0).sum(axis=axes,dtype=np.float64)
    baseline=100*reward[...,0].sum()/counts.sum()
    if abs(baseline-100*official_sum/counts.sum())>1e-5:
        raise AssertionError("official SPS mismatch")
    if abs(baseline-evaluation["persist_first_4"]["calibrated"]["scores"]["sps_score"])>1e-5:
        raise AssertionError("saved baseline mismatch")
    variants={}
    for variant in config["variants"]:
        chosen,fits=select_crossfit(reward,features,groups,variant.endswith("input_variance"))
        picked=gather_candidates(reward,chosen)
        per_group={str(g):{"sps":float(100*picked[groups==g].sum()/counts[groups==g].sum()),
                           "delta":float(100*(picked[groups==g].sum()-reward[groups==g,:,0].sum())/counts[groups==g].sum())} for g in np.unique(groups)}
        score=float(100*picked.sum()/counts.sum())
        variants[variant]={"sps":score,"delta":score-baseline,
             "coverage_percent":float(100*gather_candidates(covered,chosen).sum()/counts.sum()),
             "mean_width":float(gather_candidates(widths,chosen).sum()/counts.sum()),
             "by_re":per_group,"fits":fits,
             "screen_pass":bool(score-baseline>=config["screen_gate"]["minimum_sps_gain_each_fold"] and min(v["delta"] for v in per_group.values())>=-config["screen_gate"]["maximum_re_sps_regression"])}
    report = {"name":entry["name"],"baseline_sps":baseline,
            "baseline_coverage_percent":float(100*covered[...,0].sum()/counts.sum()),
            "baseline_mean_width":float(widths[...,0].sum()/counts.sum()),"variants":variants,
            "prediction_sha256":evaluation["prediction_sha256"],"point_predictions_unchanged":True,
            "data_hashes":{f:sha256_file(arrays/f) for f in ["inputs.npy","targets.npy","windows.json"]}}
    if return_statistics:
        return report, {"reward":reward,"covered":covered,"widths":widths,
                        "counts":counts,"features":features,"groups":groups}
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/e041_crossfit_intervals.json")
    ap.add_argument("--output",default="artifacts/e041_crossfit_intervals/report.json"); args=ap.parse_args()
    path=REPO/args.config; cfg=json.loads(path.read_text()); out=REPO/args.output
    if out.exists(): raise FileExistsError(out)
    results=[]
    for entry in cfg["models"]:
        result=analyze(entry,cfg); results.append(result)
        print(json.dumps({"name":result["name"],"baseline_sps":result["baseline_sps"],"variants":{k:{a:b for a,b in v.items() if a!="fits"} for k,v in result["variants"].items()}},indent=2),flush=True)
    report={"experiment":cfg["experiment"],"evidence_limit":cfg["evidence_limit"],"config_sha256":sha256_file(path),"results":results}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__": main()
