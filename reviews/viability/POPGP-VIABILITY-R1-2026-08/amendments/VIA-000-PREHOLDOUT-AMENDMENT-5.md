# VIA-000 pre-holdout protocol amendment 5

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This fifth amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner output, reveal, final
label, hidden holdout content, or seed content entered the maintainer or falsifier
context. Scientific code, candidate and baseline identities, hypotheses, thresholds,
measurements, and mutation rules remain unchanged.

## Falsification evidence

The fifth clean falsifier produced `attacks/VIA-000-ATTACK-PLAN-5.md` at
artifact-only commit `07fb33c02bbf820650d74fcdfd47959d2ca87084`.
Its raw artifact SHA-256 is
`ae80a9888081b35431fdf622b5d16f8d2c4c0ef4dfeb4a21fbcdaec422553d8a`.

All ten frozen mutation families were executed and rejected under the amended
non-editable isolated boundary. The exact prior editable self-cleaning exploit was
reproduced under the old flow and eliminated under the new one. The clean control,
however, found that invoking the `pytest` console script from a non-editable isolated
environment omitted the repository root from `sys.path`; two tests that intentionally
import `scripts.*` failed collection.

## Amendment

The command is changed from
`uv run --isolated --frozen --no-editable pytest -q` to
`uv run --isolated --frozen --no-editable python -m pytest -q`.
Module invocation preserves the repository working directory on Python's import path
without restoring an editable package install or a project `.pth` startup surface.
No test, threshold, mutation, expected count, or scientific criterion changes. The
falsifier session ID is rotated again.

## Acceptance condition

Before `holdout_started` may become true, a sixth fresh falsifier session must start
from the fifth amended activated handoff, commit
`attacks/VIA-000-ATTACK-PLAN-6.md`, confirm that the exact clean test command collects
and passes all 187 tests, and independently retain all ten mutation outcomes as
rejected. Only Plan 6 may be attached as the packet's decisive attack-plan receipt.
Plans 1 through 5 remain immutable provenance for the protocol repairs.
