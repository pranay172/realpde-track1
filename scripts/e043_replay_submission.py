#!/usr/bin/env python3
"""Replay exact E043 zip on Fold B; quality here is calibration, not validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/"src"),str(REPO/"realpde_t1_starting_kit_v9")]
import scoring
from realpde_t1.validation import persist_first_frames
from realpde_t1.sps_diagnostics import sps_components, summarize_sps
from e003_train_cno import sha256_file


def main():
    if not torch.cuda.is_available(): raise RuntimeError("GPU timing requires CUDA")
    root=REPO/"artifacts/e043_submission"; output=root/"gpu_replay.json"
    if output.exists(): raise FileExistsError(output)
    archive=root/"e043_candidate.zip"; build=json.loads((root/"build_report.json").read_text())
    if sha256_file(archive)!=build["archive_sha256"]: raise ValueError("archive hash mismatch")
    arrays=REPO/"artifacts/e010_secondary_fold/arrays"
    x=np.load(arrays/"inputs.npy",mmap_mode="r"); target=np.load(arrays/"targets.npy",mmap_mode="r")
    raw=np.load(REPO/"artifacts/e038_fold_b_long_backbone/predictions.npy",mmap_mode="r")
    parts=[]; maximum=0.; point_chunks=[]
    with tempfile.TemporaryDirectory(prefix="e043-replay-") as tmp:
        with zipfile.ZipFile(archive) as z: z.extractall(tmp)
        spec=importlib.util.spec_from_file_location("e043_candidate",Path(tmp)/"submission.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        # Exclude one documented warmup; separately report cold initialization.
        start=time.perf_counter(); module.predict(x[:1]); torch.cuda.synchronize()
        cold=time.perf_counter()-start; elapsed=0.
        for start in range(0,len(x),4):
            end=min(start+4,len(x)); inputs=np.asarray(x[start:end])
            t0=time.perf_counter(); y=module.predict(inputs); torch.cuda.synchronize(); elapsed+=time.perf_counter()-t0
            p=y["prediction"]; expected=persist_first_frames(raw[start:end],inputs,4)
            maximum=max(maximum,float(np.abs(p-expected).max()))
            np.testing.assert_allclose(p,expected,rtol=1e-3,atol=1e-4)
            assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in y.values())
            parts.append(sps_components(p,target[start:end],y["lower"],y["upper"],scoring))
            point_chunks.append(p)
        # GPU strict batch/repeat and permutation invariance.
        original=module.predict(x[:2]); reverse=module.predict(np.asarray(x[:2])[::-1].copy())
        for k in original:
            np.testing.assert_array_equal(original[k],reverse[k][::-1])
            np.testing.assert_array_equal(original[k][:1],module.predict(x[:1])[k])
        zero=module.predict(np.zeros((1,20,32,64,3),dtype=np.float32))
        assert all(np.isfinite(v).all() for v in zero.values())
    merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    summary=summarize_sps(merged)
    if abs(summary["sps_score"]-build["calibration_sps_in_sample"])>.01:
        raise AssertionError("deployment/calibration SPS mismatch")
    if elapsed+cold>=300: raise AssertionError("local replay exceeded five minutes")
    report={"archive_sha256":sha256_file(archive),"evidence":"Fold B calibration replay, not independent validation",
            "samples":len(x),"cold_one_sample_seconds":cold,"warm_full_replay_seconds":elapsed,
            "maximum_point_difference_vs_saved_batch4":maximum,"batch_permutation_exact":True,
            "sps":summary,"limitations":"Local RTX 5050 proxy, not hidden evaluator/A800 runtime guarantee"}
    if report["archive_sha256"]!=build["archive_sha256"]: raise AssertionError("archive changed")
    output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
