#!/usr/bin/env python3
"""Build and gate the frozen E006 Track 1 submission candidate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
DATA = PROJECT / "RealPDE-Competition-Data"
CONFIG = REPO / "configs" / "e007_submission_acceptance.json"
WRAPPER = REPO / "submission" / "e007" / "submission.py"
CHECKPOINT = REPO / "artifacts" / "e005_scale_balanced_uv" / "final.pth"
NORMALIZER = REPO / "artifacts" / "e005_scale_balanced_uv" / "normalizer.json"
ARTIFACTS = REPO / "artifacts" / "e007_submission"
ARCHIVE_NAME = "e006_candidate.zip"
REPORT_NAME = "report.json"
RUNNER = REPO / "scripts" / "container_submission_bounds_check.py"
IMAGE = "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime"
IMAGE_DIGEST = "sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854"
CHECKPOINT_SHA256 = "389f42f3509526b232b3064e168fd652c5668183f8739218cc8dc998d1c5edba"
NORMALIZER_SHA256 = "801b3aaf391f6499afdc171cb6afc4277402dd5ceb534b2a2a89d7d69514d6ae"
EXTRACTED_CAP_BYTES = 256_000_000
REQUIRED_KEYS = ("lower", "prediction", "upper")


def run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
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


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def copy_tree_without_caches(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_candidate_inputs(config: dict) -> dict:
    missing = [
        str(path)
        for path in (CONFIG, WRAPPER, CHECKPOINT, NORMALIZER, RUNNER)
        if not path.is_file()
    ]
    example = DATA / "example_data" / "3750_0.h5"
    if not example.is_file():
        missing.append(str(example))
    if missing:
        raise FileNotFoundError("missing E007 inputs:\n" + "\n".join(missing))

    checkpoint_hash = sha256_file(CHECKPOINT)
    normalizer_hash = sha256_file(NORMALIZER)
    if checkpoint_hash != CHECKPOINT_SHA256 or checkpoint_hash != config["candidate"]["checkpoint_sha256"]:
        raise AssertionError(f"unexpected checkpoint hash: {checkpoint_hash}")
    if normalizer_hash != NORMALIZER_SHA256:
        raise AssertionError(f"unexpected normalizer hash: {normalizer_hash}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    if checkpoint.get("experiment") != "E005" or checkpoint.get("status") != "fixed_budget_final":
        raise AssertionError("checkpoint is not the frozen E005 fixed-budget final")
    if checkpoint.get("normalizer") != normalizer:
        raise AssertionError("checkpoint and standalone normalizer disagree")

    wrapper = load_module(WRAPPER, "e007_source_audit")
    expected = {
        "mean_input": wrapper._MEAN_INPUT.tolist(),
        "std_input": wrapper._STD_INPUT.tolist(),
        "mean_target": wrapper._MEAN_TARGET.tolist(),
        "std_target": wrapper._STD_TARGET.tolist(),
    }
    for key, value in expected.items():
        if not np.array_equal(
            np.asarray(value, dtype=np.float32), np.asarray(normalizer[key], dtype=np.float32)
        ):
            raise AssertionError(f"wrapper {key} differs from the frozen normalizer")
    uncertainty = config["candidate"]["uncertainty"]
    expected_floor = np.float32(uncertainty["beta"] * uncertainty["sigma_global"])
    if wrapper._ALPHA != np.float32(uncertainty["alpha"]):
        raise AssertionError("wrapper alpha differs from E007 config")
    if wrapper._ADDITIVE_HALF_WIDTH != expected_floor:
        raise AssertionError("wrapper additive interval floor differs from E007 config")
    if wrapper._BATCH_SIZE != config["candidate"]["point_batch_size"]:
        raise AssertionError("wrapper batch size differs from E007 config")
    return {
        "status": "passed",
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_experiment": checkpoint["experiment"],
        "checkpoint_status": checkpoint["status"],
        "normalizer_sha256": normalizer_hash,
        "normalizer_exact_float32_match": True,
        "uncertainty_parameters_exact_float32_match": True,
        "wrapper_sha256": sha256_file(WRAPPER),
    }


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


def build_and_extract(temp_root: Path) -> tuple[Path, Path, dict]:
    staging = temp_root / "staging"
    staging.mkdir()
    shutil.copy2(WRAPPER, staging / "submission.py")
    shutil.copy2(CHECKPOINT, staging / "model.pth")
    shutil.copy2(KIT / "load_baseline.py", staging / "load_baseline.py")
    copy_tree_without_caches(KIT / "rpde_baselines", staging / "rpde_baselines")
    copy_tree_without_caches(KIT / "_vendor", staging / "_vendor")

    archive = temp_root / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(staging))

    extracted = temp_root / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for name in names:
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                raise AssertionError(f"unsafe archive member: {name}")
        if "submission.py" not in names or "model.pth" not in names:
            raise AssertionError("archive is missing required root-level files")
        if any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in names):
            raise AssertionError("archive contains Python cache files")
        bundle.extractall(extracted)

    root_entries = sorted(path.name for path in extracted.iterdir())
    required = set(json.loads(CONFIG.read_text(encoding="utf-8"))["package"]["required_root_entries"])
    if not required.issubset(root_entries):
        raise AssertionError(f"archive root entries are incomplete: {root_entries}")
    extracted_bytes = tree_size(extracted)
    if extracted_bytes >= EXTRACTED_CAP_BYTES:
        raise AssertionError(
            f"submission extracts to {extracted_bytes}, cap is {EXTRACTED_CAP_BYTES}"
        )
    checkpoint_hash = sha256_file(extracted / "model.pth")
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise AssertionError("packaged checkpoint hash changed")
    return archive, extracted, {
        "status": "passed",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "extracted_bytes": extracted_bytes,
        "cap_bytes": EXTRACTED_CAP_BYTES,
        "headroom_bytes": EXTRACTED_CAP_BYTES - extracted_bytes,
        "file_count": sum(1 for path in extracted.rglob("*") if path.is_file()),
        "root_entries": root_entries,
        "checkpoint_sha256": checkpoint_hash,
        "submission_sha256": sha256_file(extracted / "submission.py"),
        "cache_files_absent": True,
        "safe_member_paths": True,
    }


def make_public_windows(temp_root: Path) -> tuple[np.ndarray, np.ndarray]:
    source = DATA / "example_data" / "3750_0.h5"
    inputs = []
    targets = []
    with h5py.File(source, "r") as handle:
        for start in (0, 40):
            input_channels = [
                np.asarray(handle[key][start:start + 20, ::2, ::2], dtype=np.float32)
                for key in ("u", "v")
            ]
            target_channels = [
                np.asarray(handle[key][start + 20:start + 40, ::2, ::2], dtype=np.float32)
                for key in ("u", "v")
            ]
            inputs.append(
                np.stack([*input_channels, np.zeros_like(input_channels[0])], axis=-1)
            )
            targets.append(
                np.stack([*target_channels, np.zeros_like(target_channels[0])], axis=-1)
            )
    input_array = np.stack(inputs).astype(np.float32, copy=False)
    target_array = np.stack(targets).astype(np.float32, copy=False)
    expected = (2, 20, 32, 64, 3)
    if input_array.shape != expected or target_array.shape != expected:
        raise AssertionError(f"unexpected public windows: {input_array.shape}/{target_array.shape}")
    np.save(temp_root / "public_inputs.npy", input_array, allow_pickle=False)
    return input_array, target_array


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def run_submission_container(
    extracted: Path, input_array: np.ndarray, destination: Path
) -> tuple[dict[str, np.ndarray], dict]:
    destination.mkdir()
    np.save(destination / "input.npy", input_array, allow_pickle=False)
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,size=1g", "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "MKL_THREADING_LAYER=GNU",
        "-v", f"{extracted}:/submission:ro", "-v", f"{RUNNER}:/runner.py:ro",
        "-v", f"{destination}:/work:rw", "--entrypoint", "python", IMAGE,
        "-B", "/runner.py", "--input", "/work/input.npy",
        "--output", "/work/output.npz", "--report", "/work/report.json",
    ]
    started = time.perf_counter()
    completed = run(command, timeout=600)
    wall_seconds = time.perf_counter() - started
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    report["host_observed_container_wall_seconds"] = wall_seconds
    if report["status"] != "passed" or report["cuda_available"]:
        raise AssertionError(f"official-image isolation check failed: {report}")
    if report["python"] != "3.10.14" or report["torch"] != "2.2.2":
        raise AssertionError(f"unexpected official-image runtime: {report['python']}/{report['torch']}")
    if wall_seconds >= 300:
        raise AssertionError(f"representative container process exceeded 300 seconds: {wall_seconds}")
    (destination / "container.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    return load_npz(destination / "output.npz"), report


def validate_cross_process_outputs(
    batch1_a: dict[str, np.ndarray],
    batch1_b: dict[str, np.ndarray],
    batch2: dict[str, np.ndarray],
    rtol: float,
    atol: float,
) -> dict:
    for result in (batch1_a, batch1_b, batch2):
        if tuple(sorted(result)) != REQUIRED_KEYS:
            raise AssertionError(f"unexpected output keys: {sorted(result)}")
    fresh_max_abs = {}
    batch_max_abs = {}
    batch_exact = {}
    fresh_exact = {}
    batch_allclose = {}
    for key in REQUIRED_KEYS:
        fresh_max_abs[key] = float(np.max(np.abs(batch1_a[key] - batch1_b[key])))
        fresh_exact[key] = bool(np.array_equal(batch1_a[key], batch1_b[key]))
        batch_max_abs[key] = float(np.max(np.abs(batch1_a[key] - batch2[key][:1])))
        batch_exact[key] = bool(np.array_equal(batch1_a[key], batch2[key][:1]))
        batch_allclose[key] = bool(
            np.allclose(batch1_a[key], batch2[key][:1], rtol=rtol, atol=atol)
        )

    prediction = batch2["prediction"]
    half = np.float32(0.025) * np.abs(prediction) + np.float32(0.15 * 0.0563870259)
    half[..., 2] = 0.0
    expected_lower = prediction - half
    expected_upper = prediction + half
    expected_lower[..., 2] = 0.0
    expected_upper[..., 2] = 0.0
    if not np.array_equal(batch2["lower"], expected_lower):
        raise AssertionError("packaged lower bounds differ from frozen E006 formula")
    if not np.array_equal(batch2["upper"], expected_upper):
        raise AssertionError("packaged upper bounds differ from frozen E006 formula")
    fresh_pass = all(fresh_exact.values())
    batch_pass = all(batch_allclose.values())
    return {
        "status": "passed" if fresh_pass and batch_pass else "failed",
        "fresh_process_exact_repeat": fresh_pass,
        "fresh_process_exact_by_output": fresh_exact,
        "fresh_process_max_abs": fresh_max_abs,
        "batch_invariance_exact": batch_exact,
        "batch_invariance_allclose": batch_pass,
        "batch_invariance_allclose_by_output": batch_allclose,
        "batch_invariance_max_abs": batch_max_abs,
        "rtol": rtol,
        "atol": atol,
        "e006_interval_formula_exact": True,
    }


def run_contract_scoring(
    output: dict[str, np.ndarray], target: np.ndarray, mean_seconds: float, root: Path
) -> dict:
    score_sets = {}
    for name, include_bounds in (("custom_bounds", True), ("default_bounds", False)):
        score_input = root / f"scoring_{name}" / "input"
        score_reference = score_input / "ref"
        score_output = root / f"scoring_{name}" / "output"
        score_reference.mkdir(parents=True)
        arrays = {
            "prediction": output["prediction"],
            "mean_t_neural_s": np.asarray([mean_seconds], dtype=np.float64),
        }
        if include_bounds:
            arrays.update(lower=output["lower"], upper=output["upper"])
        np.savez(score_input / "predictions.npz", **arrays)
        np.savez(score_reference / "targets.npz", target=target)
        run([sys.executable, str(KIT / "scoring.py"), str(score_input), str(score_output)])
        scores = json.loads((score_output / "scores.json").read_text(encoding="utf-8"))
        expected = {"rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"}
        if set(scores) != expected or not all(np.isfinite(value) for value in scores.values()):
            raise AssertionError(f"unexpected scorer output: {scores}")
        score_sets[name] = scores
    if np.isclose(
        score_sets["custom_bounds"]["sps_score"],
        score_sets["default_bounds"]["sps_score"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("custom bounds did not change the official scorer's SPS")
    return {
        "status": "passed",
        "scope": "contract-only; two public windows, not model-selection evidence",
        "custom_bounds": score_sets["custom_bounds"],
        "default_bounds": score_sets["default_bounds"],
        "custom_bounds_path_exercised": True,
    }


def validate_host_result(raw, expected_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict) or tuple(sorted(raw)) != REQUIRED_KEYS:
        raise AssertionError("host GPU wrapper returned the wrong keys")
    result = {key: np.asarray(raw[key]) for key in REQUIRED_KEYS}
    for key, value in result.items():
        if value.shape != expected_shape or value.dtype != np.float32:
            raise AssertionError(f"bad host GPU {key} contract: {value.shape}/{value.dtype}")
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"host GPU {key} contains non-finite values")
        if not np.array_equal(value[..., 2], np.zeros_like(value[..., 2])):
            raise AssertionError(f"host GPU {key} pressure is not zero")
    if np.any(result["lower"] > result["prediction"]) or np.any(
        result["prediction"] > result["upper"]
    ):
        raise AssertionError("host GPU bounds do not enclose prediction")
    return result


def run_host_gpu_proxy(extracted: Path, public_inputs: np.ndarray, sample_count: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("E007 requires the approved local GPU proxy, but CUDA is unavailable")
    process_start = time.perf_counter()
    module = load_module(extracted / "submission.py", "e007_packaged_gpu")
    warm = validate_host_result(
        module.predict(public_inputs[:1], metadata={"gpu_warmup": True}),
        tuple(public_inputs[:1].shape),
    )
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeats = (sample_count + len(public_inputs) - 1) // len(public_inputs)
    benchmark_input = np.tile(public_inputs, (repeats, 1, 1, 1, 1))[:sample_count]
    started = time.perf_counter()
    raw = module.predict(benchmark_input, metadata={"gpu_whole_run_proxy": True})
    torch.cuda.synchronize()
    predict_seconds = time.perf_counter() - started
    result = validate_host_result(raw, tuple(benchmark_input.shape))
    whole_seconds = time.perf_counter() - process_start
    if whole_seconds >= 300:
        raise AssertionError(f"local GPU whole-run proxy exceeded 300 seconds: {whole_seconds}")
    report = {
        "status": "passed",
        "interpretation": "supplementary hardware proxy; the official A800 uses Torch 2.2.2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "sample_count": sample_count,
        "model_load_and_warmup_plus_proxy_seconds": whole_seconds,
        "proxy_predict_seconds": predict_seconds,
        "mean_proxy_seconds_per_sample": predict_seconds / sample_count,
        "below_five_minutes": True,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "output_sha256": {key: sha256_array(value) for key, value in result.items()},
    }
    del result, raw, benchmark_input
    torch.cuda.empty_cache()
    return report


def main(experiment: str = "E007") -> None:
    if ARTIFACTS.exists():
        raise FileExistsError(f"{experiment} artifacts already exist: {ARTIFACTS}")
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    if config.get("experiment") != experiment:
        raise AssertionError("wrong experiment config")
    report = {
        "experiment": experiment,
        "status": "running",
        "git_commit": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "host_python": platform.python_version(),
        "host_torch": torch.__version__,
        "host_numpy": np.__version__,
        "candidate_inputs": audit_candidate_inputs(config),
        "image": IMAGE,
        "image_expected_digest": IMAGE_DIGEST,
        "image_metadata": image_metadata(),
    }

    with tempfile.TemporaryDirectory(prefix=f"realpde-{experiment.lower()}-") as temp_name:
        temp_root = Path(temp_name)
        archive, extracted, package = build_and_extract(temp_root)
        report["submission_package"] = package
        public_inputs, public_targets = make_public_windows(temp_root)
        output_a, batch1_a = run_submission_container(
            extracted, public_inputs[:1], temp_root / "batch1_a"
        )
        output_b, batch1_b = run_submission_container(
            extracted, public_inputs[:1], temp_root / "batch1_b"
        )
        output_batch2, batch2 = run_submission_container(
            extracted, public_inputs, temp_root / "batch2"
        )
        acceptance = config["acceptance"]
        reference_hashes = acceptance.get("reference_public_batch1_sha256")
        reference_match = True
        if reference_hashes is not None:
            actual_hashes = {key: sha256_array(output_a[key]) for key in REQUIRED_KEYS}
            reference_match = actual_hashes == reference_hashes
            report["reference_public_batch1"] = {
                "status": "passed" if reference_match else "failed",
                "source": acceptance["reference_public_batch1_source"],
                "expected_sha256": reference_hashes,
                "actual_sha256": actual_hashes,
                "exact_match": reference_match,
            }
        cross_process = validate_cross_process_outputs(
            output_a,
            output_b,
            output_batch2,
            float(acceptance["batch_invariance_rtol"]),
            float(acceptance["batch_invariance_atol"]),
        )
        report["official_container"] = {
            "status": cross_process["status"],
            "isolation": {
                "network": "none",
                "root_filesystem": "read-only",
                "capabilities": "all dropped",
                "no_new_privileges": True,
                "gpu_exposed": False,
            },
            "batch1_process_a": batch1_a,
            "batch1_process_b": batch1_b,
            "batch2": batch2,
            "cross_process": cross_process,
        }
        report["contract_scoring"] = run_contract_scoring(
            output_batch2,
            public_targets,
            batch2["mean_second_predict_seconds_per_sample"],
            temp_root,
        )
        proxy = acceptance["local_gpu_whole_run_proxy"]
        report["local_gpu_proxy"] = run_host_gpu_proxy(
            extracted, public_inputs, int(proxy["samples"])
        )
        comparator_seconds = proxy.get("e007_proxy_predict_seconds")
        if comparator_seconds is not None:
            report["local_gpu_proxy"]["e007_proxy_predict_seconds"] = float(
                comparator_seconds
            )
            report["local_gpu_proxy"]["predict_slowdown_vs_e007"] = (
                report["local_gpu_proxy"]["proxy_predict_seconds"]
                / float(comparator_seconds)
            )
        report["checks"] = {
            "candidate_hashes_and_constants": True,
            "official_image_digest": True,
            "archive_root_layout": True,
            "archive_under_256_mb": True,
            "offline_read_only_container": True,
            "float32_finite_exact_shape": True,
            "bounds_enclose_prediction": True,
            "pressure_exactly_zero": True,
            "same_process_exact_repeat": True,
            "fresh_process_exact_repeat": cross_process["fresh_process_exact_repeat"],
            "batch_invariance": cross_process["batch_invariance_allclose"],
            "e006_interval_formula_exact": True,
            "official_scorer_custom_bounds": True,
            "representative_container_process_under_five_minutes": True,
            "local_gpu_357_sample_proxy_under_five_minutes": True,
        }
        if reference_hashes is not None:
            report["checks"]["public_batch1_exactly_matches_e007"] = reference_match
        report["status"] = "complete"
        report["strict_hypothesis_accepted"] = all(report["checks"].values())
        report["limitations"] = [
            "The official image cannot use the local RTX 5050 because Torch 2.2.2 lacks sm_120 kernels, so image compatibility is checked on CPU.",
            "The five-minute GPU proxy repeats public inputs on RTX 5050/Torch 2.7.1; only Codabench can measure the exact A800/Torch 2.2.2 whole-run time.",
            "The two public-window scorer result is a contract check, not validation or model-quality evidence."
        ]

        ARTIFACTS.mkdir(parents=True)
        if report["strict_hypothesis_accepted"]:
            shutil.copy2(archive, ARTIFACTS / ARCHIVE_NAME)
        report_path = ARTIFACTS / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["strict_hypothesis_accepted"]:
        print(f"{experiment} archive: {ARTIFACTS / ARCHIVE_NAME}")
    else:
        print(f"{experiment} archive withheld because the strict acceptance gate failed")
    print(f"{experiment} report: {ARTIFACTS / REPORT_NAME}")


if __name__ == "__main__":
    main()
