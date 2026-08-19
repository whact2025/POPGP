# VIA-000 pre-holdout protocol amendment 3

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This third amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner output, reveal, final
label, hidden holdout content, or seed content entered the maintainer or falsifier
context. Scientific code, candidate and baseline identities, hypotheses, thresholds,
measurements, and scientific uncertainty rules remain unchanged.

## Falsification evidence

The third clean falsifier produced `attacks/VIA-000-ATTACK-PLAN-3.md` at
artifact-only commit `41422899c249620944c7206f7ecc46be48c261b9`.
Its raw artifact SHA-256 is
`667b3afd78b0cc1701d04e4c97293d2d85df1d3634f69ae35e498c661a8159f0`.

All eight then-frozen mutation families were executed. The named pre-sync and
post-sync `sitecustomize.py` and `usercustomize.py` placements were rejected. Attack
C09 then demonstrated an equivalent startup path: an ignored executable `.pth` file
added after locked sync executed under ordinary `uv run python`, while Git diff, the
artifact checker, and the second amended postflight returned success. Ignored Git
status exposed the file.

## Amendment

The third refreeze:

1. increases the required mutation count from eight to nine;
2. adds C09 as a required mutation family;
3. requires every `uv run` command to use explicit `--isolated --frozen`, preventing
   the repository `.venv` startup surface from controlling scientific tools or Python;
4. makes postflight enumerate every `.pth` file and require exactly the two startup
   files and contents produced by the locked editable POPGP sync;
5. retains rejection of customize hooks, environment substitution, pre-existing
   ignored state, and tracked/untracked residue; and
6. rotates the falsifier session ID again.

The base interpreter performs the startup-surface comparison. A clean control uses
the locked project environment only as a sync artifact; scientific commands execute
in fresh isolated lock-frozen environments.

## Acceptance condition

Before `holdout_started` may become true, a fourth fresh falsifier session must start
from the third amended activated handoff, commit
`attacks/VIA-000-ATTACK-PLAN-4.md`, independently exercise all nine frozen mutation
families, and confirm that customize and executable `.pth` hooks neither influence
isolated execution nor survive postflight. Only Plan 4 may be attached as the
packet's decisive attack-plan receipt. Plans 1 through 3 remain immutable provenance
for the protocol repairs.
