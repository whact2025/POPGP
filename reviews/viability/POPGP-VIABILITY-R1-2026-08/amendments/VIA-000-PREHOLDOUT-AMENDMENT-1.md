# VIA-000 pre-holdout protocol amendment 1

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. The custodian verified that the two sealed commitment hashes
and packet IDs match, but did not reveal or summarize either sealed manifest. No runner
output, final label, hidden holdout content, or secret seed content existed in the
maintainer or falsifier context. The scientific candidate, baseline, hypotheses,
thresholds, measurements, uncertainty rules, and source code remain unchanged.

The amendment is therefore a pre-holdout repair to the evidence-integrity protocol,
not a post-result adjustment. The old activation must not be used to start the runner.

## Falsification evidence

The first clean falsifier produced
`attacks/VIA-000-ATTACK-PLAN-1.md` at artifact-only commit
`8a90f94da6c8476a9458dd44c214d4f8892ed2dd`. Its raw artifact SHA-256 is
`ad0bcd1a404a3c14c91f8ef36e79569ee9310ad06340149c493b5efdc22061ca`.

The original six frozen mutations were rejected. Attack A07 demonstrated that an
untracked repository-root `sitecustomize.py`, activated through an undeclared
`PYTHONPATH=.`, could influence ordinary Python commands while both
`git diff --exit-code` and the structured-artifact checker returned success. Full
`git status --porcelain=v1 --untracked-files=all` exposed the residue.

## Amendment

The refrozen protocol:

1. increases the required mutation count from six to seven;
2. adds A07 as a required mutation family;
3. requires `PYTHONPATH` to be unset;
4. adds isolated preflight and postflight commands that reject any tracked or
   untracked repository residue and any declared `PYTHONPATH`; and
5. rotates the falsifier session ID so a fresh clean seat must challenge the amended
   handoff.

The isolated command uses Python `-I` so repository `sitecustomize.py` and
`PYTHONPATH` cannot influence the checker itself. A clean control returns zero. The
A07 control prints `active` under ordinary Python, while the isolated checker reports
`?? sitecustomize.py`, reports that `PYTHONPATH` must be unset, and returns nonzero.

## Acceptance condition

Before `holdout_started` may become true, a new falsifier session must start from the
amended activated handoff, commit `VIA-000-ATTACK-PLAN-2.md`, independently exercise
all seven frozen mutation families, and confirm that A07 is rejected. Only that second
plan may be attached as the packet's decisive attack-plan receipt. The first plan is
retained as provenance for the protocol repair and must not be rewritten.
