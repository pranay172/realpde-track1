#!/usr/bin/env python3
"""Monitor checkpoint-aligned training loss; preserve best recovery on plateau.

With --pid, attach to an existing trainer. Otherwise launch the backbone trainer.
The model state and normalizer are copied verbatim from a recovery checkpoint;
an early-stopped final is labelled explicitly, never fixed-budget-final.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from realpde_t1.early_stopping import PlateauMonitor


def write_json(path, value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def check_process(pid, config):
    try:
        tokens=(Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    decoded=[x.decode() for x in tokens if x]
    if not decoded:
        return False
    if not any(x.endswith("e003_train_cno.py") for x in decoded):
        raise RuntimeError("PID is not the expected backbone trainer")
    if "--config" not in decoded or Path(decoded[decoded.index("--config")+1]).resolve() != config.resolve():
        raise RuntimeError("PID config differs from requested experiment")
    return True


def trainer_running(child, pid, config):
    # Popen owns this exact child; /proc cmdline may be empty during exec.
    if child is not None:
        return child.poll() is None
    return check_process(pid, config)


def stop_owned_child(child):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)


def finalize_early(folder, config_path, config, decision):
    best=torch.load(folder/"best_plateau_recovery.pth",map_location="cpu",weights_only=False)
    split_path=REPO/config["split_config"]
    split=json.loads(split_path.read_text())
    policy=json.loads((folder/"early_stopping_policy.json").read_text())
    if best["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("best recovery config mismatch")
    if best["split_config_sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise ValueError("best recovery split mismatch")
    provenance={"config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "split_config_sha256":best["split_config_sha256"],
                "training_files":split["training_files"],"validation_field_values_accessed":[],
                "early_stopping_policy":policy,"early_stopping_decision":decision,
                "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
    final={"format_version":1,"experiment":config["experiment"],"status":"early_stopped_final",
           "model_kind":"cno","adapter_config":None,"model_state_dict":best["model_state_dict"],
           "normalizer":best["normalizer"],"provenance":provenance,
           "training":{"updates":best["update"],"budget_updates":config["training"]["num_updates"],
                       "stopped_at_update":decision["update"]}}
    if (folder/"final.pth").exists() or (folder/"train_report.json").exists():
        raise FileExistsError("will not replace a completed training result")
    torch.save(final,folder/"early_final.tmp")
    (folder/"early_final.tmp").replace(folder/"final.pth")
    report={"experiment":config["experiment"],"status":"early_stopped",
            "updates":best["update"],"stopped_at_update":decision["update"],
            "budget_updates":config["training"]["num_updates"],"seed":config["seed"],
            "config_sha256":provenance["config_sha256"],"provenance":provenance,
            "checkpoint_sha256":hashlib.sha256((folder/"final.pth").read_bytes()).hexdigest()}
    write_json(folder/"early_stop.json",decision)
    write_json(folder/"train_report.json",report)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid",type=int)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--artifacts",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    os.chdir(REPO)
    cfg=json.loads(args.config.read_text())
    folder=args.artifacts
    if (folder/"paused_by_user.json").exists():
        raise RuntimeError("user-paused experiment requires explicit recovery plan")
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"plateau_monitor.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/"final.pth").exists():
        raise FileExistsError("training already completed")
    policy={"window":200,"min_updates":1000,"patience":600,"relative_improvement":.01,
            "check_every_updates":100,"selection":"best materially improved 200-update trailing training-loss mean",
            "validation_used":False,"authorized_change":"User requested plateau stopping on 2026-09-21"}
    write_json(folder/"early_stopping_policy.json",policy)
    child=None
    if args.pid is None:
        child=subprocess.Popen([sys.executable,"-u","scripts/e003_train_cno.py","--config",str(args.config),
                                "--artifacts",str(folder),"--device",args.device],cwd=REPO)
        args.pid=child.pid
    monitor=PlateauMonitor()
    seen=0
    previous=folder/"plateau_monitor.json"
    if previous.exists():
        saved=json.loads(previous.read_text())
        if saved.get("ready") and saved.get("status") == "monitoring":
            monitor.best_mean=saved["best_mean_loss"]
            monitor.best_update=saved["best_update"]
            seen=saved["update"]
    try:
        while True:
            if (folder/"train_report.json").exists():
                write_json(folder/"plateau_monitor.json",{"status":"budget_completed","last_checkpoint_update":seen})
                return
            if child is not None and child.poll() is not None:
                raise RuntimeError(f"trainer exited with code {child.returncode} before report")
            if not trainer_running(child,args.pid,args.config):
                raise RuntimeError("trainer exited before publishing a report")
            write_json(folder/"plateau_heartbeat.json",{
                "status":"monitoring","pid":args.pid,"updated_unix":time.time(),
                "last_checkpoint_update":seen,"owned_child":child is not None})
            history=folder/"training.jsonl"
            latest=folder/"latest.pth"
            if history.exists() and latest.exists():
                rows=[]
                for line in history.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # writer may be between flushes
                if rows and rows[-1]["update"] >= seen+100:
                    recovery=torch.load(latest,map_location="cpu",weights_only=False)
                    update=int(recovery["update"])
                    if update > seen and update <= rows[-1]["update"]:
                        if recovery["config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest():
                            raise ValueError("recovery config hash mismatch")
                        losses=[r["loss"] for r in rows if r["update"] <= update]
                        decision=monitor.observe(losses,update)
                        seen=update
                        if decision["improved"]:
                            torch.save(recovery,folder/"best_plateau.tmp")
                            (folder/"best_plateau.tmp").replace(folder/"best_plateau_recovery.pth")
                        write_json(folder/"plateau_monitor.json",{"status":"monitoring","pid":args.pid,**decision})
                        print(json.dumps(decision),flush=True)
                        if decision["stop"] and update < cfg["training"]["num_updates"]:
                            if not trainer_running(child,args.pid,args.config):
                                raise RuntimeError("trainer ended at stop boundary")
                            os.kill(args.pid,signal.SIGINT)
                            for _ in range(60):
                                if child is not None:
                                    ended=child.poll() is not None
                                else:
                                    ended=not check_process(args.pid,args.config)
                                if ended:
                                    break
                                time.sleep(1)
                            else:
                                raise RuntimeError("trainer did not exit after SIGINT; no final was written")
                            completed=[json.loads(line)["update"] for line in history.read_text().splitlines()]
                            decision["last_logged_update_after_stop"]=max(completed)
                            finalize_early(folder,args.config,cfg,decision)
                            write_json(folder/"plateau_monitor.json",{"status":"early_stopped",**decision})
                            return
            time.sleep(10)
    except BaseException as error:
        write_json(folder/"plateau_monitor_error.json",{"error":repr(error),"pid":args.pid})
        stop_owned_child(child)
        raise


if __name__ == "__main__":
    main()
