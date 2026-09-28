# Shared RealPDE environment snapshots

Identical archival copies are versioned in both `RealPDE_T1/environment-locks/`
and `RealPDE_T2/environment-locks/`, because the shared project root is not a
working Git repository. When restoring from either track repository alone,
copy its `environment-locks/` folder into the parent RealPDE project root first;
then run the commands below from that root. Do not create per-track venvs.

Captured **2026-09-27**, immediately before optional environment removal.
These cover both tracks. The installed environments, not historical notes, are
the source of truth. Both passed `uv pip check` when captured.

| Environment | Requirements | Python | Torch | Packages |
|---|---|---|---|---:|
| `.venv` — evaluator-compatible local CPU | `requirements-evaluator.txt` | 3.10.20 | 2.2.2+cu121 | 34 |
| `.venv-gpu` — RTX 5050 development | `requirements-gpu.txt` | 3.10.20 | 2.7.1+cu128 | 36 |

The JSON snapshots include exact distribution versions and host details.
Every installed distribution is pinned, including NVIDIA libraries and other
transitive dependencies. No editable or direct/local-path packages were present.

## Recreate with uv

Run from the **RealPDE project root**, after the old environment directories
have been removed or moved aside. Do not recreate over an environment in use.
These commands download Python and packages and require internet access.
`uv 0.11.20` was used originally; the commands below were checked against that
installed version's help. If running in a restricted sandbox, set
`UV_CACHE_DIR=/tmp/realpde-uv-cache` to keep uv's cache writable.

```bash
uv python install 3.10.20

uv venv --python 3.10.20 .venv
uv pip sync --python .venv/bin/python \
  --default-index https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  environment-locks/requirements-evaluator.txt

uv venv --python 3.10.20 .venv-gpu
uv pip sync --python .venv-gpu/bin/python \
  --default-index https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  environment-locks/requirements-gpu.txt

uv pip check --python .venv/bin/python
uv pip check --python .venv-gpu/bin/python
```

The explicit index strategy allows exact pinned versions to be found across
PyPI and the official PyTorch index instead of stopping at the first index
containing a package name. Do not add untrusted indexes. `uv pip sync` installs
the listed environment exactly and removes unlisted packages from its target.

## Verify without competition data

```bash
.venv/bin/python -c 'import torch,numpy,h5py,scipy,einops,yaml,requests; assert torch.__version__ == "2.2.2+cu121"; print(torch.__version__, numpy.__version__); print(torch.ones(2).sum().item())'
.venv-gpu/bin/python -c 'import torch,numpy,h5py,scipy,einops,yaml,requests; assert torch.__version__ == "2.7.1+cu128"; print(torch.__version__, numpy.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(2, device="cuda").sum().item())'
```

The GPU check requires host GPU access and a compatible NVIDIA driver. Torch
2.2.2+cu121 is intentionally **not** usable on the local Blackwell sm_120 GPU;
keep `.venv` CPU-only locally. The official Docker image is a separate resource.

## Scope and validation limits

- Platform-specific: Linux x86-64, CPython 3.10.20. Do not treat these as portable
  macOS/Windows/ARM dependency sets.
- These are exact installed-version snapshots, **not hash-locked wheel archives**.
  Reconstruction depends on those versions remaining downloadable. Matching
  versions does not guarantee byte-identical environments or scientific results.
- Snapshot versions were compared against both live environments; dependency
  checks and offline sync dry-runs passed. A fresh downloaded installation has
  not been performed, to avoid duplicating multi-GB environments.
- Competition data, checkpoints, generated submissions, CUDA drivers, system
  packages, Docker images, and uv itself are not included. Deleted artifacts
  cannot be recovered from these files.
- Keep this entire folder somewhere backed up before deleting environments.
  No environment directories were deleted by this task.
