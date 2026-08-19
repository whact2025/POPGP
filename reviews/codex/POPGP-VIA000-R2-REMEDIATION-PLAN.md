# VIA-000 R2 reproducibility remediation

Status: re-review-1 changes requested; second remediation implemented and awaiting
fresh cross-platform execution and independent re-review

Predecessor campaign: `POPGP-VIABILITY-R1-2026-08`

Predecessor outcome: valid / failed / `implementation-capability-failed`

## Why a new round is required

The R1 record is terminal and remains unchanged. Its Windows execution satisfied the
frozen protocol, while Linux regenerated contract-equivalent scientific JSON and
usable visuals but changed serialized JSON/PNG bytes. Linux also installed
`_cuda_bindings_redirector.pth` from the locked environment, contradicting R1's
Windows-specific hard-coded startup allow-list. Those observations validly failed the
tested R1 implementation capability; they did not test a new scientific observable.

R2 changes the reproducibility mechanism and therefore requires a fresh campaign and
fresh preregistration before any new holdout execution.

## Implemented mechanism

1. Structured results retain exact keys, JSON types, shapes, configuration values,
   check identities, criteria, Boolean outcomes, and finite-number requirements. In
   addition, the checker independently recomputes every registered decision Boolean
   from its typed retained operands. Numeric portability drift can never preserve a
   stale Boolean after crossing a scientific threshold.
2. Raster outputs compare format, geometry, mode, frame count, and every decoded
   channel. Retained honest Windows/Linux artifacts differ by at most four channel
   levels after a near-zero chain legend is rendered canonically, so the fail-closed
   visual bound is 4. Encoded bytes and metadata are not treated as scientific
   observables. Compact, one-pixel, dashed, text, and real-annotation mutations all
   exceed the bound.
3. Regeneration may modify only artifacts declared by validation documents. The
   runner compares literal working-tree bytes and modes with batched frozen Git blobs
   before and after every child, rejecting staged state, mutable index flags, clean-
   filter concealment, and all undeclared ordinary or ignored state. The environment,
   bytecode, uv, Ruff, Matplotlib, and general caches are fresh external directories.
4. A trusted base interpreter running with `-I -S` extracts the checker from the frozen
   Git object database and snapshots every file and symlink in the fresh external
   locked non-editable environment. The same wrapper verifies the complete environment
   and source boundary before and after every child. Python targets run through
   `scripts/run_without_startup_hooks.py` with `-I -S`; it adds dependency paths
   directly without evaluating `.pth` or customize modules and requires an external
   bytecode cache. Persistent, self-deleting, byte-restoring, installed-package, and
   ignored-bytecode carriers therefore cannot execute or erase their evidence.
5. Every decision-bearing pipeline alias is bound to the canonical raw measurement.
   Many-body precision and Richardson ratios are recomputed from their retained raw
   responses, amplitudes, and floors rather than trusted as serialized diagnostics.

## Frozen negative controls required before R2 holdout

- materially change a compact region, thin curve, displaced curve, and annotation
  while retaining filename and dimensions;
- cross every registered numerical decision margin or duplicated pipeline alias while
  retaining its stale Boolean, and alter nested precision floors independently of raw
  responses;
- hide tracked byte changes with assume-unchanged/skip-worktree and exercise staged,
  deletion, rename, symlink, ordinary-untracked, and ignored-executable state;
- inject persistent, self-deleting, and byte-restoring `.pth`, `sitecustomize.py`, and
  `usercustomize.py` carriers after the locked sync;
- alter the external startup manifest without the runner-held digest;
- replay the historical self-cleaning editable `.pth` carrier and show that isolated
  non-editable execution neither imports nor copies it; and
- modify installed dependency bytes, inject ignored timestamp-valid bytecode, execute
  and restore tracked source, and conceal source through a repository-local clean
  filter; and
- rerun on Windows and Linux in fresh external environments/caches, requiring semantic
  and calibrated per-channel visual equivalence, exact literal-source cleanliness,
  and zero unexpected complete-environment drift on both platforms.

The new gates and their executable negative controls are registered in
`docs/scientific_hardening/GATE_TEST_REGISTRY.md`. No R2 pass or Tier-R promotion is
claimed by this implementation artifact.

Independent review 1 is preserved at
`reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1.md`. Its four
blocking counterexamples were accepted. Re-review 1 is preserved at
`reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1.md`;
its three remaining visual, semantic, and residue counterexamples are accepted and
addressed here. Only a fresh independent re-review may mark them resolved.
