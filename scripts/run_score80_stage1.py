#!/usr/bin/env python3
"""Finish registered clean-model experiments, stopping on failures.

Adopts already-running E038/E039A training only with --adopt-running.
All results and child logs stay under ignored artifacts. No archive is built
or submitted. Later calibration/public-data stages require reviewing results.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
JOBS = {
    "E038": ("configs/e038_fold_b_long_backbone.json", "artifacts/e038_fold_b_long_backbone", "scripts/backbone_plateau.py"),
    "E039A": ("configs/e039a_long_backbone_rollout.json", "artifacts/e039a_long_backbone_rollout", "scripts/e027_train_rollout.py"),
    "E039B": ("configs/e039b_long_backbone_rollout.json", "artifacts/e039b_long_backbone_rollout", "scripts/e027_train_rollout.py"),
    "E040": ("configs/e040_fold_a_6000_backbone.json", "artifacts/e040_fold_a_6000_backbone", "scripts/backbone_plateau.py"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adopt-running", action="store_true")
    args = ap.parse_args()
    root = REPO / "artifacts/score80_stage1"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status":"running", "started_unix":time.time(), "pid":os.getpid(), "jobs":{}}

    def save():
        state["updated_unix"] = time.time()
        tmp = root / "state.tmp"
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(root / "state.json")

    def run(command, log):
        print("Running: " + " ".join(command), flush=True)
        with (root / log).open("a") as stream:
            with subprocess.Popen([sys.executable,"-u",*command],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT) as child:
                state["child_pid"] = child.pid
                while child.poll() is None:
                    save()
                    time.sleep(10)
                state.pop("child_pid",None)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode,command)

    def job(name, adopt=False):
        config, artifacts, trainer = JOBS[name]
        folder = REPO / artifacts
        if (folder / "paused_by_user.json").exists():
            raise RuntimeError(f"{name} is user-paused; do not restart automatically")
        cfg = json.loads((REPO / config).read_text())
        state["current_job"] = name
        state["jobs"][name] = {"status":"waiting_for_training" if adopt else "training"}
        save()
        report = folder / "train_report.json"
        if adopt:
            deadline = time.monotonic() + (cfg["training"]["maximum_wall_minutes"] + 30)*60
            while not report.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"adopted {name} did not produce its training report")
                time.sleep(20)
        elif not report.exists():
            run([trainer,"--config",config,"--artifacts",artifacts,"--device","cuda"],name+"_train.log")
        training = json.loads(report.read_text())
        early = training.get("status") == "early_stopped"
        valid_updates = training["updates"] == cfg["training"]["num_updates"]
        if early:
            valid_updates = (0 < training["updates"] < cfg["training"]["num_updates"]
                             and training.get("provenance",{}).get("early_stopping_policy",{}).get("validation_used") is False)
        if training["experiment"] != name or not valid_updates:
            raise ValueError(f"unexpected training report for {name}")
        digest = hashlib.sha256((REPO / config).read_bytes()).hexdigest()
        recorded = training.get("config_sha256",training.get("provenance",{}).get("config_sha256"))
        if recorded != digest:
            raise ValueError(f"training configuration hash mismatch for {name}")
        evaluation = folder / "evaluation.json"
        state["jobs"][name]["status"] = "evaluating"
        save()
        if not evaluation.exists():
            run(["scripts/evaluate_clean_candidate.py","--config",config,"--artifacts",artifacts,"--device","cuda"],name+"_evaluation.log")
        results = json.loads(evaluation.read_text())
        if results["experiment"] != name or results["config_sha256"] != digest:
            raise ValueError(f"stale evaluation for {name}")
        state["jobs"][name] = {"status":"complete", "report":str(evaluation.relative_to(REPO)),
                                "training_status":training.get("status"),"updates":training["updates"],
                                "scores":results["persist_first_4"]["calibrated"]["scores"],
                                "assessment":results["assessment"]}
        save()
        return results

    save()
    try:
        # E039A is cheap and already training; E038 can continue during its evaluation.
        job("E039A",args.adopt_running)
        b = job("E038",args.adopt_running)
        job("E039B")
        if b["assessment"]["passed"] and b.get("checkpoint_status") != "early_stopped_final":
            job("E040")
        else:
            state["jobs"]["E040"] = {"status":"not_run", "reason":"E038 missed replication gates or stopped early; review before longer training"}
        state["status"] = "complete_review_required"
        state["next_action"] = "Review clean A/B results and SPS headroom before registering conditional uncertainty or matched 6000-update Fold B confirmation. No submission candidate created."
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        raise
    finally:
        state["updated_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
