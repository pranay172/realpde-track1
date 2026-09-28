#!/usr/bin/env python3
"""Data-free public snapshot checks, not model correctness tests."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote

from publication_audit import candidate_paths

ROOT = Path(__file__).resolve().parents[1]
READER_DOCS = (
    "README.md", "METHOD.md", "RESULTS.md", "REPRODUCING.md",
    "PUBLICATION_AUDIT.md", "LICENSING.md", "ACKNOWLEDGMENTS.md",
    "THIRD_PARTY_NOTICES.md", "EXTERNAL_DEPENDENCIES.md",
)


class HTMLTargets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs
                            if name in {"src", "href"} and value)


def check_reader_links(root, names):
    """Resolve local Markdown/HTML links relative to their containing document."""
    root = root.resolve()
    count = 0
    for name in names:
        doc = root / name
        text = doc.read_text()
        html = HTMLTargets()
        html.feed(text)
        targets = re.findall(r"\]\(([^)]+)\)", text) + html.targets
        for target in targets:
            if "://" in target or target.startswith(("#", "mailto:", "//")):
                continue
            target_path = unquote(target.split("#")[0])
            linked = (doc.parent / target_path).resolve()
            if not linked.is_relative_to(root) or not linked.exists():
                raise ValueError(f"Missing or out-of-repository reader-doc link: {name} -> {target}")
            count += 1
    return count


def main():
    counts = {"python": 0, "json": 0, "pins": 0, "checksums": 0, "doc_links": 0}
    for p in candidate_paths(ROOT):
        if p.is_symlink():
            raise ValueError("Symbolic links are not permitted in the snapshot")
        if p.suffix == ".py":
            ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
            counts["python"] += 1
        if p.suffix == ".json":
            json.loads(p.read_text())
            counts["json"] += 1
    counts["doc_links"] = check_reader_links(ROOT, READER_DOCS)
    for line in (ROOT / "environment-locks/SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Environment checksum mismatch: {name}")
        counts["checksums"] += 1
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    for env in ("evaluator", "gpu"):
        requirements = (ROOT / f"environment-locks/requirements-{env}.txt").read_text()
        pins = dict(line.split("==") for line in requirements.splitlines()
                    if line and not line.startswith("#"))
        snapshot = json.loads((ROOT / f"environment-locks/{env}-snapshot.json").read_text())
        if {canonical(k): v for k, v in pins.items()} != {
                canonical(k): v for k, v in snapshot["packages"].items()}:
            raise ValueError(f"Environment pins differ: {env}")
        if snapshot["python_version"] != "3.10.20":
            raise ValueError(f"Unexpected Python snapshot: {env}")
        counts["pins"] += len(pins)
    print(json.dumps({"status": "passed", "checks": counts,
                      "scope": "syntax, JSON, reader links, environment pins/checksums; no model tests"},
                     indent=2))


if __name__ == "__main__":
    main()
