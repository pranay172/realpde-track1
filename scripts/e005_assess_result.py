#!/usr/bin/env python3
"""Mechanically assess E005 against its preregistered success criteria."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "e005_scale_balanced_uv.json"
E002_REPORT = REPO / "artifacts" / "e002_validation" / "report.json"
E003_EVALUATION = REPO / "artifacts" / "e003_cno_finetune" / "evaluation.json"
E004_REPORT = REPO / "artifacts" / "e004_postmortem" / "report.json"
E005_EVALUATION = REPO / "artifacts" / "e005_scale_balanced_uv" / "evaluation.json"
OUTPUT = REPO / "artifacts" / "e005_scale_balanced_uv" / "assessment.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace completed E005 assessment: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    e002 = json.loads(E002_REPORT.read_text(encoding="utf-8"))
    e003 = json.loads(E003_EVALUATION.read_text(encoding="utf-8"))
    e004 = json.loads(E004_REPORT.read_text(encoding="utf-8"))
    e005 = json.loads(E005_EVALUATION.read_text(encoding="utf-8"))
    criteria = config["evaluation"]["success_criteria"]

    e003_by_re = {
        re_value: row["e003"]["scores"]
        for re_value, row in e004["by_nominal_re"].items()
    }
    e005_by_re = {
        re_value: row["scores"]
        for re_value, row in e005["metrics_by_nominal_re"].items()
    }
    recovery_checks = {}
    recovery_gains = {}
    for metric, minimum in criteria["re_3750_min_score_gain_vs_e003"].items():
        gain = e005_by_re["3750"][metric] - e003_by_re["3750"][metric]
        recovery_gains[metric] = gain
        recovery_checks[metric] = gain >= float(minimum)

    non_regression_checks = {}
    non_regression_deltas = {}
    for re_value in ("11400", "20325", "26700"):
        non_regression_checks[re_value] = {}
        non_regression_deltas[re_value] = {}
        for metric, maximum_regression in criteria[
            "other_re_max_score_regression_vs_e003"
        ].items():
            delta = e005_by_re[re_value][metric] - e003_by_re[re_value][metric]
            non_regression_deltas[re_value][metric] = delta
            non_regression_checks[re_value][metric] = delta >= -float(maximum_regression)

    persistence_scores = e002["models"]["persistence"]["metrics"]["overall"]["scores"]
    e003_overall = e003["metrics"]["overall"]["scores"]
    e005_overall = e005["metrics"]["overall"]["scores"]
    persistence_checks = {
        metric: e005_overall[metric] > persistence_scores[metric]
        for metric in criteria["overall_must_exceed_persistence"]
    }
    mvpe_delta = e005_overall["mvpe_score"] - e003_overall["mvpe_score"]
    mvpe_check = mvpe_delta >= -float(criteria["overall_mvpe_max_regression_vs_e003"])
    failed = []
    for metric, passed in recovery_checks.items():
        if not passed:
            failed.append(f"re_3750.{metric}")
    for re_value, rows in non_regression_checks.items():
        for metric, passed in rows.items():
            if not passed:
                failed.append(f"re_{re_value}.{metric}")
    for metric, passed in persistence_checks.items():
        if not passed:
            failed.append(f"overall_vs_persistence.{metric}")
    if not mvpe_check:
        failed.append("overall_vs_e003.mvpe_score")

    report = {
        "experiment": "E005",
        "status": "complete",
        "strict_hypothesis_accepted": not failed,
        "failed_criteria": failed,
        "re_3750_score_gains_vs_e003": recovery_gains,
        "re_3750_checks": recovery_checks,
        "other_re_score_deltas_vs_e003": non_regression_deltas,
        "other_re_non_regression_checks": non_regression_checks,
        "overall_score_deltas_vs_persistence": {
            metric: e005_overall[metric] - persistence_scores[metric]
            for metric in e005_overall
            if metric in persistence_scores
        },
        "overall_persistence_checks": persistence_checks,
        "overall_mvpe_delta_vs_e003": mvpe_delta,
        "overall_mvpe_check": mvpe_check,
        "sps_status": criteria["sps"],
        "source_sha256": {
            "config": sha256_file(CONFIG),
            "e002_report": sha256_file(E002_REPORT),
            "e003_evaluation": sha256_file(E003_EVALUATION),
            "e004_report": sha256_file(E004_REPORT),
            "e005_evaluation": sha256_file(E005_EVALUATION),
        },
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
