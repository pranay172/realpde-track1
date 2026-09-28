#!/usr/bin/env python3
"""Validate existing zip bytes in two fresh, offline, read-only Docker runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

IMAGE = "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime"
WORKER = r'''
import json
import sys
import time
import numpy as np
sys.path.insert(0,"/package")
import submission
x = np.load("/inputs.npy")
keys = ("prediction","lower","upper")
def check(y,n):
    for key in keys:
        a = y[key]
        assert isinstance(a,np.ndarray) and a.shape == (n,20,32,64,3),key
        assert a.dtype == np.float32 and np.isfinite(a).all(),key
        assert np.all(a[...,2] == 0),key
    assert np.all(y["lower"] <= y["prediction"])
    assert np.all(y["prediction"] <= y["upper"])
start=time.perf_counter()
y=submission.predict(x)
check(y,len(x))
for i in range(len(x)):
    yi=submission.predict(x[i:i+1])
    check(yi,1)
    for key in keys:
        np.testing.assert_array_equal(y[key][i:i+1],yi[key])
again=submission.predict(x)
check(again,len(x))
for key in keys:
    np.testing.assert_array_equal(y[key],again[key])
saved={key:y[key] for key in keys}
stress=[]
for strength in (20,50,100):
    z=x[:1].copy()
    # Fixed positions in history, excluding last frame used by persistence.
    rng=np.random.default_rng(20260921)
    scales=np.std(z[...,:2],axis=(1,2,3),keepdims=True)
    for _ in range(8):
        t,h,w=int(rng.integers(19)),int(rng.integers(32)),int(rng.integers(64))
        z[0,t,h,w,:2] += strength*scales[0,0,0,0]*rng.normal(size=2)
    result=submission.predict(z)
    check(result,1)
    for key in keys:
        saved[f"stress_{strength}_{key}"]=result[key]
    stress.append({"sigma":strength,"input_max_abs":float(np.abs(z).max()),
                   "prediction_max_abs":float(np.abs(result["prediction"]).max())})
np.savez("/output/result.npz",**saved)
with open("/output/report.json","w") as f:
    json.dump({"seconds":time.perf_counter()-start,"stress":stress},f,indent=2)
'''


def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "submission.py" not in names:
            raise ValueError("duplicate entries or missing root submission.py")
        if sum(i.file_size for i in archive.infolist()) >= 256_000_000:
            raise ValueError("extracted package exceeds 256 MB")
        for info in archive.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
                raise ValueError("unsafe archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive symlinks are not supported")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--expected-sha256",required=True)
    ap.add_argument("--inputs",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--image",default=IMAGE)
    args=ap.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    before=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if before != args.expected_sha256:
        raise ValueError("archive hash mismatch")
    inspect_zip(args.archive)
    with tempfile.TemporaryDirectory(prefix="realpde-verify-") as temporary:
        root=Path(temporary)
        package=root/"package"
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(package)
        inputs=np.asarray(np.load(args.inputs,mmap_mode="r")[:2],dtype=np.float32)
        if inputs.shape != (2,20,32,64,3) or not np.isfinite(inputs).all():
            raise ValueError("two finite real input windows are required")
        np.save(root/"inputs.npy",inputs)
        (root/"worker.py").write_text(WORKER)
        reports=[]
        outputs=[]
        for run in range(2):
            output=root/f"run{run}"
            output.mkdir()
            command=["docker","run","--rm","--network=none","--read-only",
                     "--user",f"{os.getuid()}:{os.getgid()}",
                     "--cap-drop=ALL","--security-opt=no-new-privileges",
                     "--tmpfs","/tmp:rw,size=512m",
                     "-e","PYTHONDONTWRITEBYTECODE=1",
                     "-v",f"{package}:/package:ro",
                     "-v",f"{root/'inputs.npy'}:/inputs.npy:ro",
                     "-v",f"{root/'worker.py'}:/worker.py:ro",
                     "-v",f"{output}:/output:rw",args.image,"python","/worker.py"]
            subprocess.run(command,check=True,timeout=300)
            with np.load(output/"result.npz") as data:
                outputs.append({k:data[k] for k in data.files})
            reports.append(json.loads((output/"report.json").read_text()))
        for key in outputs[0]:
            np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
    after=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("archive changed during verification")
    report={"archive":str(args.archive),"archive_sha256":after,"image":args.image,
            "contract_passed":True,"fresh_process_exact":True,"runs":reports,
            "limitations":"Stress magnitudes are recorded, not a quality acceptance gate. Full holdout scoring and whole-run GPU timing are separate requirements."}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
