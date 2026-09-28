#!/usr/bin/env python3
"""Run E001: Track 1 evaluator-container and golden-submission acceptance.

The script deliberately uses CPU containers because the competition's Torch
2.2.2 wheel has no kernels for the local RTX 5050 (sm_120). It produces a compact
ignored report under ``artifacts/e001_evaluator_acceptance/``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
DATA = PROJECT / "RealPDE-Competition-Data"
IMAGE = "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime"
IMAGE_DIGEST = "sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854"
ARTIFACTS = REPO / "artifacts" / "e001_evaluator_acceptance"
EXTRACTED_CAP_BYTES = 256_000_000

CHECKPOINTS = {
    "sim_cno.pth": DATA / "baseline_checkpoints/sim_pretrain/sim_cno.pth",
    "sim_fno.pth": DATA / "baseline_checkpoints/sim_pretrain/sim_fno.pth",
    "sim_fno_fp16.pth": DATA / "baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth",
    "sim_transolver.pth": DATA / "baseline_checkpoints/sim_pretrain/sim_transolver.pth",
    "sim_real_cno.pth": DATA / "baseline_checkpoints/sim_real_ft/sim_real_cno.pth",
    "sim_real_fno.pth": DATA / "baseline_checkpoints/sim_real_ft/sim_real_fno.pth",
    "sim_real_fno_fp16.pth": DATA / "baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth",
    "sim_real_transolver.pth": DATA / "baseline_checkpoints/sim_real_ft/sim_real_transolver.pth",
}


def run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def require_inputs() -> None:
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    example = DATA / "example_data/3750_0.h5"
    if not example.is_file():
        missing.append(str(example))
    if missing:
        raise FileNotFoundError("missing E001 inputs:\n" + "\n".join(missing))


def image_metadata() -> dict:
    raw = json.loads(run(["docker", "image", "inspect", IMAGE]).stdout)[0]
    repo_digests = raw.get("RepoDigests", [])
    if not any(value.endswith("@" + IMAGE_DIGEST) for value in repo_digests):
        raise AssertionError(f"unexpected image digests: {repo_digests}")
    return {
        "id": raw["Id"],
        "repo_digests": repo_digests,
        "created": raw["Created"],
        "architecture": raw["Architecture"],
        "os": raw["Os"],
        "size_bytes": raw["Size"],
    }


def audit_image_packages() -> dict:
    code = (
        "import importlib.util,json,platform,torch,numpy;"
        "names=['scipy','h5py','matplotlib','pandas','einops'];"
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,"
        "'numpy':numpy.__version__,'cuda_available':torch.cuda.is_available(),"
        "'optional_present':{n:importlib.util.find_spec(n) is not None for n in names}}))"
    )
    completed = run([
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--entrypoint", "python", IMAGE, "-B", "-c", code,
    ])
    audit = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_absent = {name: False for name in ("scipy", "h5py", "matplotlib", "pandas", "einops")}
    if audit["optional_present"] != expected_absent:
        raise AssertionError(f"unexpected evaluator packages: {audit['optional_present']}")
    if audit["torch"] != "2.2.2" or audit["numpy"] != "1.26.4":
        raise AssertionError(f"unexpected core versions: {audit}")
    if audit["cuda_available"]:
        raise AssertionError("E001 container should be isolated from the local GPU")
    return audit


def populate_flat_checkpoints(destination: Path) -> None:
    destination.mkdir()
    for name, source in CHECKPOINTS.items():
        target = destination / name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def run_kit_smoke(flat_checkpoints: Path) -> dict:
    start = time.perf_counter()
    completed = run([
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,size=1g",
        # The image defaults to MKL_THREADING_LAYER=INTEL. After the parent smoke
        # process loads libgomp, its torch-only packer subprocess needs GNU here.
        "-e", "MKL_THREADING_LAYER=GNU",
        "-e", "CKPT_DIR=/checkpoints", "-e", "EXAMPLE_H5=/missing/example.h5",
        "-v", f"{KIT}:/kit:ro", "-v", f"{flat_checkpoints}:/checkpoints:ro",
        "--entrypoint", "python", IMAGE, "-B", "/kit/smoke_test_kit.py",
    ], timeout=900)
    elapsed = time.perf_counter() - start
    output = completed.stdout + completed.stderr
    required = [
        "forwarded model types: ['cno', 'fno', 'transolver']",
        "weight round-trip:",
        "EVAL-IMAGE PARITY",
        "ALL KIT SMOKE CHECKS COMPLETE",
    ]
    missing = [marker for marker in required if marker not in output]
    if missing:
        raise AssertionError(f"kit smoke output is missing markers: {missing}")
    (ARTIFACTS / "kit_smoke.log").write_text(output, encoding="utf-8")
    return {"status": "passed", "elapsed_seconds": elapsed, "markers": required}


def copy_tree_without_caches(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def build_and_extract_submission(temp_root: Path) -> tuple[Path, dict]:
    staging = temp_root / "submission_staging"
    staging.mkdir()
    shutil.copy2(KIT / "submission_example_fno.py", staging / "submission.py")
    shutil.copy2(KIT / "load_baseline.py", staging / "load_baseline.py")
    shutil.copy2(CHECKPOINTS["sim_real_fno_fp16.pth"], staging / "sim_real_fno_fp16.pth")
    copy_tree_without_caches(KIT / "rpde_baselines", staging / "rpde_baselines")
    copy_tree_without_caches(KIT / "_vendor", staging / "_vendor")

    archive = temp_root / "sim_real_fno_fp16_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(staging))

    extracted = temp_root / "submission_extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if "submission.py" not in names or any(name.startswith("submission_staging/") for name in names):
            raise AssertionError("submission archive does not have the required root layout")
        if any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in names):
            raise AssertionError("submission archive contains Python cache files")
        bundle.extractall(extracted)

    extracted_bytes = tree_size(extracted)
    if extracted_bytes >= EXTRACTED_CAP_BYTES:
        raise AssertionError(
            f"submission extracts to {extracted_bytes} bytes, cap is {EXTRACTED_CAP_BYTES}"
        )
    return extracted, {
        "status": "passed",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "extracted_bytes": extracted_bytes,
        "cap_bytes": EXTRACTED_CAP_BYTES,
        "headroom_bytes": EXTRACTED_CAP_BYTES - extracted_bytes,
        "file_count": sum(1 for path in extracted.rglob("*") if path.is_file()),
        "root_entries": sorted(path.name for path in extracted.iterdir()),
        "checkpoint_sha256": sha256_file(extracted / "sim_real_fno_fp16.pth"),
    }


def make_windows(work: Path) -> tuple[np.ndarray, np.ndarray]:
    example = DATA / "example_data/3750_0.h5"
    inputs = []
    targets = []
    with h5py.File(example, "r") as handle:
        for start in (0, 40):
            fields = []
            future = []
            for key in ("u", "v"):
                fields.append(np.asarray(handle[key][start:start + 20, ::2, ::2], dtype=np.float32))
                future.append(np.asarray(handle[key][start + 20:start + 40, ::2, ::2], dtype=np.float32))
            zero = np.zeros_like(fields[0])
            inputs.append(np.stack([fields[0], fields[1], zero], axis=-1))
            targets.append(np.stack([future[0], future[1], zero], axis=-1))
    input_array = np.stack(inputs, axis=0)
    target_array = np.stack(targets, axis=0)
    if input_array.shape != (2, 20, 32, 64, 3) or target_array.shape != input_array.shape:
        raise AssertionError(f"bad example windows: {input_array.shape}/{target_array.shape}")
    np.save(work / "inputs.npy", input_array, allow_pickle=False)
    np.save(work / "targets.npy", target_array, allow_pickle=False)
    return input_array, target_array


def run_submission_container(
    extracted: Path,
    runner: Path,
    input_array: np.ndarray,
    destination: Path,
) -> tuple[np.ndarray, dict]:
    destination.mkdir()
    np.save(destination / "input.npy", input_array, allow_pickle=False)
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,size=1g", "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{extracted}:/submission:ro", "-v", f"{runner}:/runner.py:ro",
        "-v", f"{destination}:/work:rw", "--entrypoint", "python", IMAGE,
        "-B", "/runner.py", "--input", "/work/input.npy",
        "--output", "/work/output.npy", "--report", "/work/report.json",
    ]
    completed = run(command, timeout=600)
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "passed" or report["cuda_available"]:
        raise AssertionError(f"submission runner failed isolation checks: {report}")
    (destination / "container.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    return np.load(destination / "output.npy", allow_pickle=False), report


def run_contract_scoring(prediction: np.ndarray, target: np.ndarray, mean_seconds: float, root: Path) -> dict:
    scoring_input = root / "scoring_input"
    scoring_ref = scoring_input / "ref"
    scoring_output = root / "scoring_output"
    scoring_ref.mkdir(parents=True)
    np.savez(
        scoring_input / "predictions.npz",
        prediction=prediction,
        mean_t_neural_s=np.asarray([mean_seconds], dtype=np.float64),
    )
    np.savez(scoring_ref / "targets.npz", target=target)
    run([sys.executable, str(KIT / "scoring.py"), str(scoring_input), str(scoring_output)])
    scores = json.loads((scoring_output / "scores.json").read_text(encoding="utf-8"))
    if set(scores) != {"rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"}:
        raise AssertionError(f"unexpected scorer output: {scores}")
    if not all(np.isfinite(value) for value in scores.values()):
        raise AssertionError(f"non-finite scorer output: {scores}")
    return {
        "status": "passed",
        "scope": "contract-only; two windows from one public trajectory, not a validation result",
        "scores": scores,
    }


def main() -> None:
    require_inputs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "experiment": "E001",
        "image": IMAGE,
        "image_expected_digest": IMAGE_DIGEST,
        "host_python": sys.version.split()[0],
        "git_commit": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
    }

    with tempfile.TemporaryDirectory(prefix="realpde-e001-") as temp_name:
        temp_root = Path(temp_name)
        flat_checkpoints = temp_root / "flat_checkpoints"
        populate_flat_checkpoints(flat_checkpoints)

        report["image_metadata"] = image_metadata()
        report["package_audit"] = audit_image_packages()
        report["kit_smoke"] = run_kit_smoke(flat_checkpoints)
        extracted, package_report = build_and_extract_submission(temp_root)
        report["submission_package"] = package_report

        input_array, target_array = make_windows(temp_root)
        runner = REPO / "scripts/container_submission_check.py"
        output_a, batch1_a = run_submission_container(
            extracted, runner, input_array[:1], temp_root / "batch1_a"
        )
        output_b, batch1_b = run_submission_container(
            extracted, runner, input_array[:1], temp_root / "batch1_b"
        )
        output_batch2, batch2 = run_submission_container(
            extracted, runner, input_array, temp_root / "batch2"
        )

        fresh_max_abs = float(np.max(np.abs(output_a - output_b)))
        if not np.array_equal(output_a, output_b):
            raise AssertionError(f"fresh-process predictions differ: max_abs={fresh_max_abs}")
        batch_max_abs = float(np.max(np.abs(output_a - output_batch2[:1])))
        batch_exact = bool(np.array_equal(output_a, output_batch2[:1]))
        if not np.allclose(output_a, output_batch2[:1], rtol=1e-6, atol=1e-7):
            raise AssertionError(
                f"batch-size 1 prediction differs from batch-size 2: max_abs={batch_max_abs}"
            )

        report["submission_inference"] = {
            "status": "passed",
            "batch1_process_a": batch1_a,
            "batch1_process_b": batch1_b,
            "batch2": batch2,
            "fresh_process_exact_repeat": True,
            "fresh_process_max_abs": fresh_max_abs,
            "batch_invariance_exact": batch_exact,
            "batch_invariance_allclose": True,
            "batch_invariance_max_abs": batch_max_abs,
        }
        report["contract_scoring"] = run_contract_scoring(
            output_batch2,
            target_array,
            batch2["mean_first_predict_seconds_per_sample"],
            temp_root,
        )

    report_path = ARTIFACTS / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"E001 report: {report_path}")


if __name__ == "__main__":
    main()
