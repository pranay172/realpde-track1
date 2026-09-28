#!/usr/bin/env python3
"""Validate E042 metadata without opening calibration/audit field arrays."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def validate_roles(protocol, split, universe):
    roles=[protocol[key] for key in ("point_training_files","calibration_files","audit_files")]
    if any(len(xs)!=len(set(xs)) or not xs for xs in roles):
        raise ValueError("empty/duplicate role membership")
    sets=[set(xs) for xs in roles]
    regimes=[{int(f.split("_")[0]) for f in xs} for xs in roles]
    for i in range(3):
        for j in range(i):
            if sets[i]&sets[j] or regimes[i]&regimes[j]:
                raise ValueError("overlapping trajectory or Reynolds group")
    if set.union(*sets)!=set(universe) or "7575_0.h5" in set.union(*sets):
        raise ValueError("invalid usable data partition")
    if set(split["training_files"])!=sets[0] or set(split["validation_files"])!=sets[1]|sets[2]:
        raise ValueError("trainer split does not enforce protocol roles")
    return {name:len(xs) for name,xs in zip(("point_training","calibration","audit"),roles)}


def main():
    paths=["configs/e042_locked_protocol.json","configs/e042_point_split.json",
           "configs/e042_clean_backbone.json"]
    protocol,split,training=[json.loads((REPO/p).read_text()) for p in paths]
    a=json.loads((REPO/"configs/e002_split.json").read_text())
    c=json.loads((REPO/"configs/e012_fold_c.json").read_text())
    counts=validate_roles(protocol,split,a["training_files"]+a["validation_files"])
    if protocol["calibration_files"]!=a["validation_files"] or protocol["audit_files"]!=c["validation_files"]:
        raise ValueError("source fold membership changed")
    baseline=json.loads((REPO/"configs/e038_fold_b_long_backbone.json").read_text())
    if training["training"]!=baseline["training"] or training["initial_checkpoint"]!=baseline["initial_checkpoint"]:
        raise ValueError("unregistered point training change")
    data=REPO.parent/"RealPDE-Competition-Data/train_real"
    if any(not (data/f).is_file() for f in a["training_files"]+a["validation_files"]):
        raise FileNotFoundError("missing released trajectory")
    output={"experiment":"E042","status":"metadata_preflight_passed_no_training_started",
            "roles":counts,"field_arrays_opened":False,
            "hashes":{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths},
            "estimated_gpu_hours":protocol["budget"]["estimated_gpu_hours"]}
    print(json.dumps(output,indent=2))


if __name__=="__main__": main()
