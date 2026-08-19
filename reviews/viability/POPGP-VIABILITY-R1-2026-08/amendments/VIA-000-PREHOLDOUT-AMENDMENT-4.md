# VIA-000 pre-holdout protocol amendment 4

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This fourth amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner output, reveal, final
label, hidden holdout content, or seed content entered the maintainer or falsifier
context. Scientific code, candidate and baseline identities, hypotheses, thresholds,
measurements, and scientific uncertainty rules remain unchanged.

## Falsification evidence

The fourth clean falsifier produced `attacks/VIA-000-ATTACK-PLAN-4.md` at
artifact-only commit `07dc8ef3012f6fa15334a1d9b676c8613037c531`.
Its raw artifact SHA-256 is
`e5ec37929331991989a289d0d14b2123ba32a63d4cf75793a71b958da7d9e58b`.

All nine then-frozen mutation families were executed. Extra `.pth`, customize, and
static allowed-file mutations were either isolated from execution or rejected by
postflight. Attack D10 then showed that uv copied the project environment's allowed
editable `_editable_impl_popgp.pth` into each temporary isolated environment. A
benign executable line set a marker and restored the project file before the command
body, so the exact checker, Git diff, and postflight all returned success.

## Amendment

The fourth refreeze:

1. increases the required mutation count from nine to ten;
2. adds D10 as a required mutation family;
3. changes locked sync to `uv sync --frozen --no-editable`;
4. requires every uv-run command to use `--isolated --frozen --no-editable`, so POPGP
   is built and installed as a wheel rather than propagated through editable `.pth`;
5. changes the valid project-environment startup surface to exactly uv's fixed
   `_virtualenv.pth` with content `import _virtualenv`;
6. retains rejection of all customize hooks, extra or modified `.pth`, environment
   substitution, pre-existing ignored state, and tracked/untracked residue; and
7. rotates the falsifier session ID again.

A clean non-editable sync removes `_editable_impl_popgp.pth`. The observed D10 hook
therefore has no allowed file to inhabit or to copy into isolated execution.

## Acceptance condition

Before `holdout_started` may become true, a fifth fresh falsifier session must start
from the fourth amended activated handoff, commit
`attacks/VIA-000-ATTACK-PLAN-5.md`, independently exercise all ten frozen mutation
families, and confirm that editable/self-cleaning startup variants cannot influence or
evade the non-editable isolated command sequence. Only Plan 5 may be attached as the
packet's decisive attack-plan receipt. Plans 1 through 4 remain immutable provenance
for the protocol repairs.
