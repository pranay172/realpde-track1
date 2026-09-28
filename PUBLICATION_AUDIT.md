# Public snapshot audit — 2026-09-28

## Scope

This source-only snapshot was prepared for public research release from the
private archive commit in [PUBLIC_SNAPSHOT.json](PUBLIC_SNAPSHOT.json). No old
Git objects, refs, config, remotes, or commit identities were exported.
The private repositories remain separate; **publish this folder, not the
original development repository**.

Omitted: organizer kit v9, copied competition pages, local environment handoff,
example data, datasets, checkpoints, generated submissions, caches,
and all untracked source-workspace files. The source/configurations, results,
limitations, experiment notes, and dependency pins remain. Historical artifact
paths and private commit hashes are provenance only.

Upstream implementations are external dependencies, not relicensed; see
[LICENSING.md](LICENSING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Removing the kit does not make the external model/data stack unrestricted MIT.

## Re-run before publication

Local checks passed on 2026-09-28: 225 source files scanned; 143 Python / 51 JSON
parsed; 40 reader links; 70 dependency pins and 4 checksum records. Eight release
guard tests and four E042 protocol tests passed. No heuristic privacy/payload
findings. Fresh `main` history is initialized for the public snapshot. The owner
explicitly approved Pranay Vandanapu and the personal email recorded in
`PUBLIC_SNAPSHOT.json` for public author/committer metadata. Re-run the history
audit after committing and before publishing. CI has not run remotely.

200 retained files match the private archive byte-for-byte; all model/experiment
Python ASTs match. Changes are release docs/tools/CI, whitespace cleanup and
exclusions, not model logic.

```bash
python3 scripts/check_archive.py
python3 scripts/publication_audit.py
python3 -m unittest discover -s tests -p test_publication_checks.py
python3 -m unittest discover -s tests -p test_e042_protocol.py
```

The checks require only Python, plus Git if a local repository has been
initialized. They verify syntax/JSON, reader-facing local links, environment
pin/checksum consistency, forbidden payloads, symbolic links, personal paths,
and a limited set of credential patterns. An initialized snapshot additionally
checks all reachable Git blobs and author/committer emails. Unknown email
identities are flagged unless they use GitHub no-reply addresses or the owner's
exact explicitly approved personal email. Other personal emails remain blocked.
No source values matching secret patterns are printed.

These are limited heuristic checks, **not** comprehensive security scanning,
legal clearance, full scientific validation, or proof of reproducibility.
Historical internal-note links to deleted artifacts are not treated as files
that the public repository promises to provide. This audit does not inspect
remote LFS objects, unreachable Git objects, or unknown credential formats.

## Publication handoff

1. Preserve the owner-approved public identity in the local Git configuration.
   Its personal email will be publicly visible in the commit history; this was
   an explicit owner choice, not an inherited global configuration default.
2. Use the fresh `main` repository in **this folder**. Review `git status` and
   rerun the checks, including the reachable-history audit, before publishing.
3. Create/select an empty GitHub repository named `realpde-track1`. If that
   name already hosts the private development history, keep it private and
   choose a distinct public destination. Do not force-push over that archive.
4. Publish only this new `main` branch. Never use `--mirror`, import the old
   repository history, or change the original private repository's visibility.
5. Confirm the account, destination, visibility, public author attribution and
   GitHub Actions result. Add actual sibling links when the destinations are
   known; no account-specific URLs are invented here.

No push, remote/account mutation, or visibility change was performed by this
snapshot preparation. Only local snapshot history and configuration were created.

## Reproduction boundary

Environments and historical data/model artifacts were deleted; they were not
reinstalled or regenerated for this release. Full training, GPU/Docker checks,
model correctness tests and hidden evaluation were not rerun. Exact historical
kit download access could not be verified. See
[REPRODUCING.md](REPRODUCING.md) and [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).

E037 remains the best reported development score (77.307902); its historical
checkpoint assembly is not an automated end-to-end recipe. E043 failed official
evaluation with no supplied cause.
No final competition placement is claimed.
