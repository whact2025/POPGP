# VIA-000 R2 pre-holdout amendment 3

Status: proposed; not activated; holdout remains unstarted and custody remains unrevealed.

This amendment addresses the independent re-review counterexamples against protocol
handoff `e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a` and the failed hosted calibration
runs at that handoff. It does not authorize execution, reveal, adjudication, or a
scientific conclusion.

## Execution trust boundary

The assigned reproduction-runner seat, the GitHub Actions control plane used for the
exact-SHA platform witness, and the independent reviewer seat are trusted principals
for R2. Retained evidence bytes, transported fragments, assembler inputs, and declared
summary Booleans are untrusted and are checked fail closed. The validator proves typed
byte closure, exact candidate/protocol identity, and recomputed predicates for evidence
supplied by those trusted principals. It does not claim that hashes or self-contained
logs can prove honest execution by a malicious or colluding trusted principal. That
excluded threat is now stated in the primary protocol instead of being implied away.

## Evidence and assembly changes

- The frozen assembler invokes the same authoritative raw-evidence verifier used by
  campaign validation before atomically publishing either `raw-results.json` or the
  output commitment. Schema-valid but semantically invalid input produces no output
  directory or commitment.
- The commitment uses the canonical custody envelope: `packet_id`, `committed_by`,
  `committed_at`, `output_receipt_id`, and `output_sha256`.
- Mutation evidence is bound to a frozen family-specific oracle identifier and a typed
  execution record. A generic nonempty error string is not a sufficient oracle.
- PDF evidence is parsed independently with a strict PDF parser, must contain exactly
  eleven derived pages, and must be accompanied by a retained exact engine-version
  record. Framed or padded pseudo-PDF bytes are invalid.
- Corresponding Ubuntu and Windows rasters are compared directly after the existing
  canonicalization. Each channel must differ by at most four; opposed per-platform
  drifts cannot consume the tolerance twice.

## Runner and hosted-workflow changes

The runner now records the generated repository status, requires every changed path to
belong to the frozen artifact allowlist, copies the generated artifacts into evidence,
restores those paths from the exact candidate, and then requires literal normal and
ignored cleanliness. Campaign validation parses both retained status records. The
hosted workflow uploads only the immutable evidence and PDF directories, not the clone,
environment, or caches. Ordinary CI fetches full history because the evidence tests
bind the retained scientific candidate commit and tree.

## Gate

The replacement handoff requires green ordinary CI, green Ubuntu and Windows hosted
platform runs, a fresh independent re-review resolving every prior finding/test, and a
new clean falsifier recommendation of SAFE. Until all four conditions hold, VIA-000
remains preregistered and the R2 holdout must not start.
