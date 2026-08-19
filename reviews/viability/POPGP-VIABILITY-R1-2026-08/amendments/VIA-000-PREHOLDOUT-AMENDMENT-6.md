# VIA-000 pre-holdout protocol amendment 6

Status: proposed for refreeze before holdout start.

## Boundary and authorization

This sixth amendment was made while `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner output, reveal, final
label, hidden holdout content, or seed content entered the maintainer or falsifier
context. Scientific code, candidate and baseline identities, hypotheses, thresholds,
measurements, and mutation rules remain unchanged.

## Falsification evidence

The sixth clean falsifier produced `attacks/VIA-000-ATTACK-PLAN-6.md` at
artifact-only commit `a72f851becfe36ba89ef951ef8fd41b1ae4ace7e`.
Its raw artifact SHA-256 is
`20768450243687bd19e47b2097b20b5dd69ac3c25e42ef1a23ff5e5e8c87c883`.

The corrected isolated non-editable module test command passed exactly 187 tests.
Ruff, TeX source validation, all six examples, the semantic checker, and both
pdfTeX passes also succeeded. The exact clean postflight nevertheless failed because
pdfTeX wrote `framework.aux`, `framework.log`, `framework.out`, `framework.pdf`, and
`frameworkNotes.bib` into the repository root. Git diff did not expose those untracked
files; full porcelain did.

## Amendment

The protocol now creates a fixed, initially absent evidence directory adjacent to the
disposable clone and passes it to pdfTeX through its quoted `-output-directory`
argument. Both PDF passes write into that external directory, whose raw artifacts are
retained and hashed by the runner. Postflight continues to require the exact candidate
repository itself to have no tracked or untracked residue. No PDF engine, pass count,
source, test, threshold, mutation, or scientific criterion changes. The falsifier
session ID is rotated again.

## Acceptance condition

Before `holdout_started` may become true, a seventh fresh falsifier session must start
from the sixth amended activated handoff, commit
`attacks/VIA-000-ATTACK-PLAN-7.md`, run the exact full clean sequence including two
external PDF passes and postflight, and retain all ten mutation outcomes as rejected.
Only Plan 7 may be attached as the packet's decisive attack-plan receipt. Plans 1
through 6 remain immutable provenance for the protocol repairs.
