#!/usr/bin/env python3
"""Fit frozen E041 rule on Fold B and package E038 without retraining."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from e041_crossfit_intervals import analyze, REPO, scoring
from e003_train_cno import sha256_file
from realpde_t1.interval_crossfit import gather_candidates
from verify_existing_submission import inspect_zip


def main():
    root=REPO/"artifacts/e043_submission"
    if root.exists(): raise FileExistsError("preserve existing candidate directory")
    cfg=json.loads((REPO/"configs/e041_crossfit_intervals.json").read_text())
    entry=next(e for e in cfg["models"] if e["name"]=="E038")
    diagnostics,stats=analyze(entry,cfg,return_statistics=True)
    threshold=float(np.median(stats["features"]))
    bins=(stats["features"]>threshold).astype(int)
    indices=[]
    for bucket in (0,1):
        selected=stats["reward"][bins==bucket]
        indices.append(selected.sum(axis=0).argmax(axis=-1) if len(selected) else np.zeros(6,dtype=int))
    indices=np.asarray(indices)
    scales=np.asarray(cfg["multipliers"])[indices].reshape(2,3,2)
    cal={"experiment":"E043","threshold":threshold,"scales":scales.tolist(),
         "horizons":cfg["horizons"],"sigma_global":scoring.SIGMA_GLOBAL,
         "fit_role":"Fold B calibration, not independent validation",
         "data_hashes":diagnostics["data_hashes"],"prediction_sha256":diagnostics["prediction_sha256"]}
    folder=REPO/entry["artifacts"]
    checkpoint=torch.load(folder/"final.pth",map_location="cpu",weights_only=False)
    model_bytes=io.BytesIO()
    torch.save({"model_state_dict":checkpoint["model_state_dict"],"normalizer":checkpoint["normalizer"]},model_bytes)
    root.mkdir(parents=True)
    archive=root/"e043_candidate.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr("model.pth",model_bytes.getvalue())
        z.writestr("calibration.json",json.dumps(cal,indent=2)+"\n")
        z.writestr("metadata","model_type: cno\n")
        z.write(REPO/"submission/e043/submission.py","submission.py")
        kit=REPO/"realpde_t1_starting_kit_v9"
        z.write(kit/"load_baseline.py","load_baseline.py")
        for sub in ("rpde_baselines","_vendor"):
            for p in sorted((kit/sub).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".pyo"):
                    z.write(p,p.relative_to(kit))
    inspect_zip(archive)
    chosen=indices[bins]
    report={"experiment":"E043","archive_sha256":sha256_file(archive),
       "archive_bytes":archive.stat().st_size,"source_checkpoint_sha256":sha256_file(folder/"final.pth"),
       "source_config_sha256":sha256_file(REPO/entry["config"]),"calibration":cal,
       "calibration_sps_in_sample":float(100*gather_candidates(stats["reward"],chosen).sum()/stats["counts"].sum()),
       "prior_crossfit_evidence":diagnostics,
       "warning":"Fold B is now calibration. Local fitted SPS is optimistic; candidate is not a proven E037 upgrade.",
       "e037_sha256":sha256_file(REPO/"artifacts/e037_submission/e037_candidate.zip")}
    with zipfile.ZipFile(archive) as z: report["extracted_bytes"]=sum(x.file_size for x in z.infolist())
    (root/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("prior_crossfit_evidence","calibration")},indent=2))


if __name__=="__main__": main()
