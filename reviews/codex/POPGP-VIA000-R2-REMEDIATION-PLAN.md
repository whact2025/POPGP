# VIA-000 R2 reproducibility remediation

Status: review-1 remediation implemented; awaiting independent re-review

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
2. Raster outputs compare format, geometry, mode, frame count, normalized mean pixel
   error, the fraction of channels with a large error, a sliding 32x32 local-error
   maximum, and the largest connected high-error region. Encoded bytes and metadata
   are not treated as scientific observables.
   Plot fields already classified as zero at roundoff are rendered from canonical zero
   display data with fixed color limits, while their raw diagnostics remain unchanged.
3. Regeneration may modify only artifacts declared by the validation documents. A
   fresh temporary Git index loaded from the frozen tree compares actual working-tree
   bytes and modes, rejects staged state and mutable index flags, and enumerates both
   ordinary and ignored residue. Only the separately verified environment and known
   runtime caches are allowed as ignored state.
4. A trusted base interpreter running with `-I -S` snapshots every `.pth`,
   `sitecustomize.py`, and `usercustomize.py` immediately after a clean locked
   non-editable sync. The same trusted wrapper verifies the snapshot before and after
   every child. Python targets themselves run through
   `scripts/run_without_startup_hooks.py` with `-I -S`; it adds dependency paths
   directly without evaluating `.pth` or customize modules. Persistent,
   self-deleting, and byte-restoring hooks therefore cannot execute or erase their
   evidence before the preflight.

## Frozen negative controls required before R2 holdout

- materially change a compact region, thin curve, displaced curve, and annotation
  while retaining filename and dimensions;
- cross every registered numerical decision margin while retaining its stale Boolean;
- hide tracked byte changes with assume-unchanged/skip-worktree and exercise staged,
  deletion, rename, symlink, ordinary-untracked, and ignored-executable state;
- inject persistent, self-deleting, and byte-restoring `.pth`, `sitecustomize.py`, and
  `usercustomize.py` carriers after the locked sync;
- alter the external startup manifest without the runner-held digest;
- replay the historical self-cleaning editable `.pth` carrier and show that isolated
  non-editable execution neither imports nor copies it; and
- rerun on Windows and Linux, requiring semantic equivalence, exact source cleanliness,
  and zero unexpected startup-surface drift on both platforms.

The new gates and their executable negative controls are registered in
`docs/scientific_hardening/GATE_TEST_REGISTRY.md`. No R2 pass or Tier-R promotion is
claimed by this implementation artifact.

Independent review 1 is preserved at
`reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1.md`. Its four
blocking counterexamples are accepted and addressed here; only a fresh independent
re-review may mark them resolved.
