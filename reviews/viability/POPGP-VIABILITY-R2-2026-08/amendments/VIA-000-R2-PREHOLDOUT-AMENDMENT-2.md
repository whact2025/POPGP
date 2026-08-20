# VIA-000 R2 pre-holdout protocol amendment 2

Status: proposed for independent re-review and refreeze before holdout start.

## Boundary

R2 remains preregistered, `holdout_started: false`, and unrevealed. This amendment
does not change the scientific candidate, hypothesis, thresholds, calibrated visual
bound, or eighteen mutation families. No hidden holdout, secret seed, output
commitment, or runner result was accessed.

## Independent-review findings addressed

The first independent review of amendment 1 requested changes in five areas:

1. hash closure did not prove that retained evidence had the declared scientific
   meaning;
2. platform and frozen-contract identities were not unconditionally reconciled;
3. a raw-results author could self-declare blockage;
4. platform fragments lacked a frozen fail-closed assembler; and
5. the draft protocol manifest contained a worktree-byte hash instead of the exact
   Git-blob hash of the validator.

## Amendment

The raw-results schema and validator now require typed evidence roles, exact command
contract identities, both frozen platform identities, all eighteen generated
artifacts, all eighteen mutation-family records per platform, a real PDF envelope and
two retained eleven-page build logs, complete source-tree Git object bindings, and a
typed environment manifest. The validator independently compares every retained JSON
and visual artifact to the frozen scientific candidate and reruns the executable
semantic decision predicates; stored capability, count, and outcome Booleans are
recomputed rather than trusted.

R2 raw results now require `blocked: false`. A missing prerequisite, early command
failure, partial platform fragment, or incomplete mutation package is an invalid
attempt and cannot be assembled or committed. It is not an author-selected terminal
scientific outcome.

`VIA-000-ASSEMBLER.py` is a frozen protocol artifact. It requires exactly the two
platform fragments and two mutation packages, verifies their hashes and frozen
identity sets, prefixes all paths to avoid collisions, rejects any nonzero command or
accepted/missing mutation, validates the combined document against the frozen Draft
2020-12 schema, and atomically emits `raw-results.json` plus a distinct
`output-commitment.json`. On any error it removes its private temporary assembly and
leaves no output commitment.

The protocol-manifest correction is intentionally deferred until after the
implementation-fix commit exists. The refreeze step must compute every contract hash
from `git show <exact-commit>:<path>` bytes, not the Windows worktree, and an
independent re-review must verify those bindings before activation.

## Acceptance condition

This amendment cannot be refrozen until a fresh independent reviewer verifies all
five prior finding IDs and requested tests as resolved/satisfied with zero blockers.
After refreeze, a new clean falsifier must execute the complete protocol, the original
Plan-1 attacks, the review's synthetic-evidence/identity/blockage/partial-assembly
counterexamples, and broadened equivalents. Holdout authorization remains forbidden
until that falsifier returns an unequivocal safe recommendation.
