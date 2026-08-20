# VIA-000 R2 pre-holdout protocol amendment 1

Status: proposed for independent review and refreeze before holdout start.

## Boundary and authorization

This amendment was made while R2 `VIA-000` remained `preregistered` with
`holdout_started: false`. Custody remained sealed. No runner result, output
commitment, reveal, final label, hidden holdout content, or secret-seed content
entered the builder or falsifier context. The scientific candidate, comparison
baseline, hypothesis, measurements, thresholds, visual bound, mutation families,
and outcome rules are unchanged.

## Falsification evidence

The first R2 clean falsifier produced
`attacks/VIA-000-R2-ATTACK-PLAN-1.md` at artifact-only commit
`66eac5c7aede6595bd7402fc9196ea7d29203e6a`. Its raw artifact SHA-256 is
`09cfd61ec238785c8b2d1dfea2c5b1979d08ca1ac94553a5de91f42b194197c4`.

The falsifier rejected the semantic, visual, provenance, startup, dependency,
bytecode, repository-boundary, protocol-freeze, custody, and outcome attacks it
executed. It nevertheless found two pre-holdout blockers:

1. The Windows supporting runner compared the complete `uv --version` banner to
   the literal string `uv 0.11.11`. The pinned executable correctly reported the
   semantic version followed by build metadata, so the clean control aborted before
   cloning the candidate.
2. The public campaign validator accepted a summary-only raw-results document. A
   result could therefore assert platform, capability, mutation, and outcome
   Booleans without supplying the required retained command, manifest, visual, and
   PDF evidence.

## Amendment

The supporting runner now parses the first two whitespace-delimited fields of the
`uv --version` banner and requires the executable name `uv` and exact semantic
version `0.11.11`; build metadata is retained but does not create a false mismatch.

The protocol now freezes a Draft 2020-12 raw-results schema and an executable
`raw_results_contract`. For reproduced and later packet states, the authoritative
validator requires the schema-bound raw-results receipt, both declared platforms,
the exact command-ID set, retained per-command stdout and stderr hashes, a complete
evidence manifest, candidate commit/tree identity, environment/source/PDF hashes,
test/example/visual/mutation counts, typed contract Booleans, and a mechanically
recomputed capability/failure outcome. Evidence paths must remain inside the
campaign and their byte counts and SHA-256 values must match the retained files.
Summary-only, missing-platform, missing-command, changed-byte, stale-count, stale
capability, and stale-outcome results fail closed.

The runner emits the platform evidence manifest, command-result map, and platform
summary needed by the reproduction runner to assemble the single pre-reveal raw
results and output commitment. The clean runner summary intentionally records zero
mutations and a nonpassing overall result until the independently executed frozen
mutation families are attached and aggregated.

## Acceptance condition

Before this amendment can be activated, a fresh independent reviewer must approve
the exact remediation commit with zero blockers. The protocol must then be refrozen
through new snapshot, activation, and handoff commits while custody remains sealed.
Before `holdout_started` may become true, a second fresh falsifier session must run
the exact clean control, replay all eighteen mutation families, reproduce the two
Plan-1 counterexamples, and commit a new attack-plan receipt concluding that the
amended protocol is safe. Plan 1 remains immutable provenance and cannot be replaced
or reinterpreted.
