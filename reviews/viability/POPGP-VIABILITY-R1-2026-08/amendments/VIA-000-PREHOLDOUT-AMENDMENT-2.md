# VIA-000 pre-holdout protocol amendment 2

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This second amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner output, reveal, final
label, hidden holdout content, or seed content entered the maintainer or falsifier
context. The scientific candidate, baseline, hypotheses, thresholds, measurements,
uncertainty rules for scientific values, and source code remain unchanged.

## Falsification evidence

The second clean falsifier produced `attacks/VIA-000-ATTACK-PLAN-2.md` at
artifact-only commit `6f082c96d8a929a71cc6074fdfc3f9ed2d171b1e`.
Its raw artifact SHA-256 is
`840aa134fad43b2e5508f03dd9d1248450cb0a31271883806d0d6cd0b6b7a79e`.

All seven then-frozen mutation families were rejected, including the original
repository-root `sitecustomize.py` plus `PYTHONPATH` attack. Attack B08 demonstrated a
different boundary: an ignored `.venv/Lib/site-packages/sitecustomize.py` existed
before locked sync, survived `uv sync --frozen`, and executed under ordinary
`uv run python`. Normal porcelain status, Git diff, the artifact checker, and the
first amended checks still returned success. `git status --ignored` exposed it.

## Amendment

The second refreeze:

1. increases the required mutation count from seven to eight;
2. adds B08 as a required mutation family;
3. requires `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`, and
   `UV_PROJECT_ENVIRONMENT` to be unset;
4. requires a base interpreter rather than an active virtual-environment interpreter
   for both evidence-integrity checks;
5. makes preflight reject all tracked, untracked, and ignored repository state, which
   forces the locked environment to be created from an empty disposable clone;
6. makes postflight reject a missing or symlinked `.venv` and any
   `sitecustomize.py` or `usercustomize.py` inside it; and
7. rotates the falsifier session ID again.

The base interpreter prevents a candidate `.venv` startup hook from controlling the
checker that detects it. A clean preflight is expected to run before `uv sync`; the
postflight is expected to run after locked sync and all scientific commands.

## Acceptance condition

Before `holdout_started` may become true, a third fresh falsifier session must start
from the second amended activated handoff, commit
`attacks/VIA-000-ATTACK-PLAN-3.md`, independently exercise all eight frozen mutation
families, and confirm that B08 is rejected at both its pre-sync and post-sync
placements. Only that third plan may be attached as the packet's decisive attack-plan
receipt. Plans 1 and 2 remain immutable provenance for the two protocol repairs.
