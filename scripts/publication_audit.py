#!/usr/bin/env python3
"""Redacted release-tree and reachable-history checks; stdlib only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".npy", ".npz",
                      ".zip", ".tar", ".gz", ".tgz", ".pem", ".key", ".pdf"}
FORBIDDEN_NAMES = {"competition_docs", "ENVIRONMENT_SETUP_ARCHIVE.md",
                   "LOCAL_INVENTORY.md", "artifacts", "data", "datasets",
                   "checkpoints", "ckpts", ".env"}
PATTERNS = {
    "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "cloud_access_id": rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "service_key": rb"\bsk-[A-Za-z0-9_-]{24,}",
    "url_credentials": rb"https?://[^\s/:@]{2,}:[^\s/@]{4,}@",
    "personal_path": rb"/(?:home|Users)/[A-Za-z0-9_.-]+/",
}
EMAIL = re.compile(rb"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_CONTACTS = {b"realpde-competition@googlegroups.com"}
# Explicitly approved by the owner for public Git metadata on 2026-09-28.
# This is an exact-address exception, not permission for other personal emails.
APPROVED_PUBLIC_EMAILS = {b"pranayvandanapu2001@gmail.com"}


def public_email(value: bytes) -> bool:
    return value in APPROVED_PUBLIC_EMAILS or bool(
        re.fullmatch(rb"[A-Za-z0-9_.+-]+@users\.noreply\.github\.com", value)
    )


def forbidden_path(name: str) -> bool:
    p = Path(name)
    return (
        p.suffix.lower() in FORBIDDEN_SUFFIXES
        or p.parts[0] in FORBIDDEN_NAMES
        or any(part in {"competition_docs", "ENVIRONMENT_SETUP_ARCHIVE.md", "LOCAL_INVENTORY.md", ".env"}
               or part.startswith(("realpde_t1_starting_kit_", "realpde_t2_starting_kit_",
                                   ".venv", ".env."))
               for part in p.parts)
    )


def scan_blob(name: str, content: bytes) -> list[dict[str, str]]:
    categories = []
    if forbidden_path(name):
        categories.append("excluded_payload")
    if len(content) > 10_000_000:
        categories.append("large_file")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        categories.append("non_text_payload")
    if b"\0" in content:
        categories.append("binary_payload")
    for label, pattern in PATTERNS.items():
        if re.search(pattern, content):
            categories.append(label)
    if any(e not in ALLOWED_CONTACTS and not public_email(e)
           for e in EMAIL.findall(content)):
        categories.append("unreviewed_email")
    return [{"path": name, "category": category} for category in sorted(set(categories))]


def git(root: Path, *args: str, **kwargs) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], **kwargs)


def own_git(root: Path) -> bool:
    if not (root / ".git").exists():
        return False
    top = git(root, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top).resolve() != root.resolve():
        raise RuntimeError("Refusing to inspect an ancestor repository")
    return True


def candidate_paths(root: Path) -> list[Path]:
    if own_git(root):
        raw = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        return [root / p for p in sorted(set(raw.decode().split("\0"))) if p]
    paths = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            p = Path(parent) / name
            if p.is_symlink():
                paths.append(p)
                dirs.remove(name)
            elif name in {".git", "__pycache__", ".pytest_cache"}:
                dirs.remove(name)
        paths.extend(Path(parent) / name for name in files)
    return sorted(paths)


def audit(root: Path) -> dict:
    findings = []
    paths = candidate_paths(root)
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"path": name, "category": "symbolic_link"})
        elif not path.is_file():
            findings.append({"path": name, "category": "missing_indexed_file"})
        else:
            findings.extend(scan_blob(name, path.read_bytes()))
    history = {"state": "not_initialized", "commits": 0, "blobs_scanned": 0}
    if own_git(root):
        commits = git(root, "rev-list", "--all").decode().splitlines()
        history = {"state": "fresh_uncommitted" if not commits else "checked",
                   "commits": len(commits), "blobs_scanned": 0}
        # Check metadata without printing author names or email values.
        identities = git(root, "log", "--all", "--format=%ae%n%ce").splitlines() if commits else []
        for email in set(identities):
            if not public_email(email):
                findings.append({"path": "<Git metadata>", "category": "unreviewed_commit_email"})
        objects = git(root, "rev-list", "--objects", "--all").decode().splitlines()
        for line in objects:
            sha, _, name = line.partition(" ")
            if not name:
                continue
            kind = git(root, "cat-file", "-t", sha).strip()
            if kind != b"blob":
                continue
            history["blobs_scanned"] += 1
            findings.extend(dict(item, object=sha)
                            for item in scan_blob(name, git(root, "cat-file", "blob", sha)))
        # Preserve path/mode safety even if a blob is shared between commits.
        for commit in commits:
            for item in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
                if not item:
                    continue
                meta, name = item.split(b"\t", 1)
                mode, _, _ = meta.split()
                path = name.decode()
                if mode in {b"120000", b"160000"} or forbidden_path(path):
                    findings.append({"path": path, "category": "excluded_historical_entry",
                                     "object": commit})
    unique = {json.dumps(item, sort_keys=True): item for item in findings}
    return {"status": "passed" if not unique else "failed", "files_scanned": len(paths),
            "history": history, "findings": list(unique.values()),
            "limits": "Heuristic text patterns, selected exclusions, reachable local Git history only; "
                      "no remote LFS/unreachable-object/entropy/unknown-format scan; not legal clearance "
                      "or model validation. Not-initialized history is not a completed metadata audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
