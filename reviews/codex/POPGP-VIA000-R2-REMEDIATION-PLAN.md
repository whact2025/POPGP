# VIA-000 R2 reproducibility remediation

Status: implemented candidate awaiting independent falsification  
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
   check identities, criteria, Boolean outcomes, and finite-number requirements. Only
   named bounded diagnostic policies accept numerical drift.
2. Raster outputs compare format, geometry, mode, frame count, normalized mean pixel
   error, and the fraction of channels with a large error. Encoded bytes and metadata
   are not treated as scientific observables.
3. Regeneration may modify only artifacts declared by the validation documents. Any
   source/configuration change or untracked repository residue fails closed.
4. A base interpreter snapshots every `.pth`, `sitecustomize.py`, and
   `usercustomize.py` immediately after a clean locked non-editable sync. Postflight
   requires byte-identical equality to that hash-bound platform-local snapshot. This
   admits files installed by the lock while rejecting later injection or mutation.

## Frozen negative controls required before R2 holdout

- materially change a visual while retaining its filename and dimensions;
- change a non-artifact tracked file and add an untracked file;
- inject an executable `.pth` after the locked sync;
- alter the external startup manifest without the runner-held digest;
- replay the historical self-cleaning editable `.pth` carrier and show that isolated
  non-editable execution neither imports nor copies it; and
- rerun on Windows and Linux, requiring semantic equivalence, exact source cleanliness,
  and zero unexpected startup-surface drift on both platforms.

The new gates and their executable negative controls are registered in
`docs/scientific_hardening/GATE_TEST_REGISTRY.md`. No R2 pass or Tier-R promotion is
claimed by this implementation artifact.
