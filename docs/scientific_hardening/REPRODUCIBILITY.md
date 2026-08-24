# Reproducibility record

Audit date: 2026-08-09 (America/New_York).

Review-protocol remediation verified: 2026-08-10 (America/New_York).

Blackwell native-build calibration verified: 2026-08-13 (America/New_York).

Blackwell review remediation verified: 2026-08-17 (America/New_York).

## Environment

- Windows / PowerShell
- Python 3.11.15
- uv 0.11.11
- CMake 4.3.2
- Locked environment from `uv.lock`
- NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, compute capability 12.0,
  driver 595.79, 12,227 MiB
- CUDA 13.3.73 development tools in the isolated local extraction
  `C:\src\POPGP-cuda-toolkit-13.3\local`; the installer URL and SHA-256 are pinned
  in `popgp_engine/kernel/README.md`; the locked Python environment remains CPU-only
  (`torch 2.10.0+cpu`)
- pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
- GitHub CLI 2.97.0; authenticated as `fuocor`

GitHub CLI access was verified against pull request #1. `docs/framework.tex` is
authoritative for the revised manuscript. A two-pass fresh PDF build produced an
11-page, 535,368-byte artifact with SHA-256
`8fd4be5c2003dde42f1ff390a0d3e4a955167e58524c16f5c7114944c9759323`;
layout warnings and the missing bibliography remain recorded limitations.

## Commands

```text
uv sync --frozen --no-editable
uv run --frozen --no-editable ruff check .
uv run --frozen --no-editable python scripts/check_tex.py
uv run --frozen --no-editable python -m pytest -q -p no:cacheprovider
uv run --frozen --no-editable python -m examples.physics_qg.chain_1d
uv run --frozen --no-editable python -m examples.physics_qg.grid_2d
uv run --frozen --no-editable python -m examples.physics_qg.gravity_well
uv run --frozen --no-editable python -m examples.physics_qg.source_law
uv run --frozen --no-editable python -m examples.physics_qg.source_law_many_body
uv run --frozen --no-editable python -m examples.physics_qg.ca_model
uv run --frozen --no-editable python -m scripts.check_validation_artifacts --enforce-change-boundary
```

Additional controlled checks exercised five common-probe seeds for chain partition
selection and swept k-NN values before adopting the blind adaptive-gap inference.

## Hardened results

| Command | Approx. runtime | Result |
|---|---:|---|
| `pytest -q` | about 1200 s | 449 passed |
| chain example | 15 s | contiguous blocks, D*=1, finite spectral peak≈0.84 |
| grid example | 6 s | 12/12 edges, P=R=1, D*=2; singleton Pi_res inadmissible |
| gravity diagnostic | 7 s | Green-function checks pass; singleton Pi_res inadmissible |
| source-law example | 7 s | RE slope≈1.9997; modular slope=1.0; negative result retained |
| many-body source-law example | 9 s | direct quadratic/floor gates and Kubo--Mori/Richardson susceptibility pass through β=3, including β=2.5 |
| CA analogy | 9 s | population declined 33 to 27; overall check fails; PNG/GIF/JSON regenerated |

The original 2026-08-13 native receipt is superseded as verification evidence because
its CTest invocation could succeed with zero discovered tests and its benchmark did
not validate kernel execution or output. The 2026-08-17 remediation build used MSVC
19.50, CUDA 13.3.73, the pinned vcpkg commit, and explicit architecture 120. The build
gate independently enumerated exactly seven CTest cases, all of which passed: three
phase-flow controls, boundary-cut calculation, pruning transitions, explicit clock
not-implemented behavior, and the self-validating benchmark. `cuobjdump --list-elf`
reported three `sm_120` cubins.

The hardened one-million-cell benchmark performed a warmup outside the timed region,
checked every launch and synchronization, read the final state back, and rejected
nonfinite or non-unit-norm output. A repeat completed 100 two-color steps in 448.57 ms
(`2.23e8` edge-updates/s), emitted FNV-1a-64 state checksum
`48e6ef8f40cb137c`, and reported maximum norm error
`1.1435297153639112e-14`. A separate CUDA-enabled PyTorch 2.10 probe exercised the
Python `GPUBackend` through the rebuilt DLL on 64 cells; the step was finite and
nontrivial, with maximum per-cell norm error `5.55e-16`. These remain implementation
and throughput calibrations, not evidence that the mean-field backend reproduces exact
MI/QCMI or the full projection pipeline.

All examples regenerated their committed `validation.json` artifacts. Wall-clock
timestamps have been removed, and every result artifact was byte-identical across two
consecutive runs in the locked Windows environment. Linux CI exposed bounded LAPACK
and floating-point drift despite unchanged scientific gates. CI therefore compares a
declared validation contract rather than serialized bytes: JSON schema, types, array
shapes, metadata, stable configuration, check identities, and pass/fail outcomes stay
strict; all numbers must be finite; ordinary diagnostics use narrow tolerances; and
the small set of sensitive fit/error fields has an explicit bounded allowlist in
`scripts/check_validation_artifacts.py`. Every registered check Boolean is also
recomputed from its retained typed operands. Required raster outputs must preserve
format/geometry. After canonicalizing the one declared near-zero chain legend value,
retained Windows/Linux evidence shows at most four intensity levels of per-channel
drift, so every raster channel is bounded by 4. This calibrated maximum rejects
compact features, one-pixel and dashed curves, rendered text, and actual plot-
annotation removal; encoded PNG bytes and metadata are not compared across
environments.

The prospective VIA-000 R2 protocol adds stricter evidence-integrity execution beyond
ordinary CI. The workflow creates a fresh locked non-editable environment and every
runtime cache outside the checkout. A base interpreter invoked with `-I -S` extracts
the boundary checker from the frozen Git object database and creates complete external
environment and literal-source manifests. `scripts/check_reproduction_boundary.py
run` hashes every environment file/symlink and compares worktree bytes/modes directly
to batched frozen Git blobs before and after each child; it does not trust index stat
flags or clean filters, and it permits no ignored checkout state. Python children use
`scripts/run_without_startup_hooks.py` under `-I -S` with an external bytecode cache;
the bootstrap inserts dependency directories directly and never evaluates `.pth`,
`sitecustomize.py`, or `usercustomize.py`. Every registered decision also binds
duplicated pipeline/check fields and recomputes its fits, quadratic/Richardson
assessments, first-law and static/evolved local-global identities, global-energy and
endpoint statistics, controls, sensitivity cases, and derived precision ratios from
the lowest-level retained raw values. Retained clock solves additionally carry their
complete mutual-information matrix, inferred-edge support, unmodified source,
removed constant mode, zero-mode policy, gauge choice, and configured mass. The
checker reconstructs the sparse symmetric nonnegative zero-diagonal weight matrix
from MI plus edges, reconstructs the effective source from the raw source and policy,
and only then recomputes the finite-graph equation. The localized gravity diagnostic
also derives its raw source from the declared center and point strength. The
checker rejects malformed, non-finite, mis-shaped, unordered, or zero-denominator
operands rather than accepting a stale serialized summary.
These mechanisms await a fresh preregistered two-platform R2 run; they are not a new
viability result.

## Negative and sensitivity results

- Relative entropy and its induced potential have fitted perturbative slope ≈2.
- Under an affine mixture, modular energy, physical energy, and the linear clock solve
  are exactly proportional to ε. Their unit slopes are analytic identities and are
  excluded from the robustness claim.
- In an exact five-site non-affine KMS family `ρ(ε)∝exp[-β(H+εV)]`, relative
  coefficient, absolute slope, normalized residual, and precision-floor gates verify
  `D=O(ε²)` and reject a synthetic first-order control. Signed Richardson estimates
  give a nonzero modular susceptibility and match exact Kubo--Mori values across the
  declared Heisenberg/Ising, β≤3 (including 2.5), and three-size finite sweep. Energy
  response is not independently gated because it is fixed by the KMS identity. An
  isospectral unitary
  control instead has `ΔS=0` and quadratic `D=Δ⟨K⟩=βΔ⟨H⟩`. A separate quench
  conserves the global Hamiltonian expectation while the audited profile spreads
  under Heisenberg dynamics; the commuting
  Ising profile remains stationary. This is a family-dependent feasibility result.
- The one-site reduced modular-energy mode is numerically blind in the symmetric KMS
  chain. The separately named exact-backend `−βΔ⟨h_i⟩` candidate reproduces the
  audited local profile. Its runtime requires the supplied reference to match the
  backend Gibbs state at the same β within trace distance `1e-10`; under that premise
  it sums to minus the global modular-energy change. Its microscopic decomposition
  dependence is retained as a limitation.
- Equal-energy states relative to a KMS reference have different entropy and raw
  relative entropy, exposing an entropy confound for a mass-source interpretation.
- The old fixed 3-nearest-neighbor grid inference had precision 0.75 and recall 1.0
  (16 inferred edges versus 12 reference edges).
- The new MI-gap rule recovers the selected chain/grid and passes label permutation
  equivariance. The four-cell chain is MST-degenerate. Uniform and disjoint-Bell
  correlations are marked non-separable, but the Bell control still receives a false
  D*=1 `geometric_candidate` declaration.
- A Petersen expander selects D*=4 at the weak default penalty, but λ=1 can force
  D*=2 with more than twice the stress; the embedding gate reports that result as
  `poor_fit` rather than a geometric candidate.
- Local metric fits use a Frobenius-orthonormal symmetric basis, report an explicit
  underdetermined flag with fixed rank tolerance, and are invariant under rotations
  of the MDS frame. The canonical MDS frame fixes the full represented subspace,
  including rank-deficient degenerate embeddings, and removes arbitrary orientation
  changes from serialized tensors. Earlier regularized scalar diagnostics could shift by
  roughly 30–60% under a rotation of the degenerate 2D eigenspace. The 3×3 grid still
  has underdetermined boundary fits; its 2D Delaunay deficits are proxy-only.
- Chain partition selection is unchanged for probe seeds 0, 1, 2, 42, and 99, while
  leakage estimates vary by roughly 19% across that small sweep.
- The 3×3 grid provides only two nonzero radial shells; its log-distance fit cannot
  validate a continuum potential law.

## Baseline mismatches corrected

Before hardening, the public grid constructor used an incompatible two-qubit default,
the gravity example used a positive source and reversed redshift interpretation,
the grid visualization overlaid reference edges without edge-recovery metrics, and
the geometry report called a penalized objective “stress.” Those outputs should not
be compared as evidence-equivalent to the regenerated artifacts.

## R3 stage-isolated hosted evidence

The drafted VIA-000 R3 hosted protocol treats candidate code as untrusted relative to
later evidence production. Candidate tests/generators, PDF production, and mutation
verification therefore run in separate fresh GitHub-hosted jobs on each platform.
No job consumes another job's writable environment, cache, temporary directory, tool
tree, configuration, or process state. The PDF job checks out the exact candidate
without running candidate Python, sanitizes TeX/kpathsea/font/native-loader selectors,
disables shell escape, and retains identical before/after manifests of the complete
pinned TeX tree. Every stage attests its own identity-bound summary and evidence
manifest. The assembler accepts only declared non-executable evidence bytes and
requires the complete, same-run three-stage set before emitting any commitment.

Each R3 stage additionally treats candidate and frozen-mutation execution as an
untrusted process tree. The Windows path starts a restricted low-integrity process
suspended, assigns it to a kill-on-close Job Object, and resumes it only after
assignment. The Ubuntu path creates a fresh unprivileged system account per command,
runs it in a systemd transient service, proves empty control group and UID process set,
creates evidence while the UID remains allocated, rechecks the UID process set, then
deletes the account. Retaining the allocation across both checks prevents UID reuse
from invalidating quiescence. Mutable staging is isolated from non-writable tool/config and
trusted-evidence roots, and child temp/home/cache/loader variables cannot name trusted
paths. Evidence creation and attestation-subject capture occur only after the complete
tree is terminated and zero descendants are proven. A production-hostile gate on both
hosted platforms launches a detached delayed writer against the live evidence and tool
paths and requires containment plus byte identity; the Windows primitive also has a
local executable regression.

The separate non-scientific proof workflow uses six explicit jobs. Each post-quiescence
canonical envelope is capped at 131072 bytes and staged as the sole regular file at a
fixed cell-specific workspace-relative path. A separate trusted outer step emits only
its lowercase SHA-256. Pinned cache v6.1.0 steps use an exact key binding repository,
workflow/source SHA, run/attempt, platform, stage, namespace, and digest; pre-existing
keys, prefix restore, cache misses, and primary/matched-key differences fail. Ubuntu
validates the six exact restored bytes in memory, retains exactly six envelopes plus
one aggregate manifest in one artifact, then downloads and revalidates that seven-file
set. Empty stdout/stderr retain the standard SHA-256 of the empty byte string. Cache
save warnings cannot establish success; missing, duplicate, nonhex, stale, fallback,
oversized, corrupt, cross-cell, or retained link/file/hash substitutions fail closed.
On Windows the fresh post-teardown cell export root has a protected DACL containing
exactly one current-runner SID `FullControl` ACE, exact owner SID, and an exact medium
`S-1-16-8192` mandatory label with `NO_WRITE_UP`; the inherited envelope descriptor
and hash are re-queried by separate trusted steps before digest output and cache save.
Ubuntu constructs its canonical expected empty token list as a non-null object array.

The hosted Ubuntu proof explicitly disables `PrivateTmp`, `ProtectSystem`, and
`ProtectHome` because that runner rejects the corresponding mount namespace. It makes
no namespace-isolation claim: the enforced boundary is the ephemeral unprivileged
account plus runner-owned mode-0700 protected roots, a dedicated mutable root,
no-new-privileges/SUID controls, closure hashes, empty-cgroup and empty-UID-process
proof, and account removal.
RR18 replaces the Windows managed export-root ACL setter with native owner/DACL
application. The fresh workspace-relative root must have the exact current runner
owner, protected non-defaulted non-null DACL, one explicit runner FullControl CI/OI
ACE, control mask 37892, and medium NO_WRITE_UP label. After the canonical envelope
is closed, its owner is set natively without replacing its one inherited runner ACE;
its exact control mask is 33796. Creation, digest, and pre-cache-save boundaries
independently requery native and managed facts. The RR17 experiment remains
non-authoritative and is not part of the campaign tree.

RR19 binds the proof-only Windows cache transport to the hosted image's exact
`C:\tools\zstd\zstd.exe` installation and `C:\tools\zstd` PATH entry. Each Windows
cell rejects missing, alternate, reparse, multi-link, alternate-stream, PATH-shadowed,
or non-1.5.7 zstd identities, captures its SHA-256, and repeats the complete identity
and hash check immediately before and after the pinned cache-save action. The failed
RR18 replay `32763190366` produced no retained aggregate or artifact and is superseded
only by a fresh exact six-cell RR19 replay.

RR20 replaces the platform-text proof/result JSON pipelines with one hash-bound
PowerShell byte writer shared by the production proof runner and containment helper.
It serializes one compact object, emits strict UTF-8 without BOM or raw control bytes,
appends exactly one `0x0A`, uses exclusive create or explicit replacement with
write-through flush, and verifies identical read-back bytes and SHA-256. The aggregate
continues to reject CR/BOM and now also rejects missing, embedded, or doubled LF in
decoded inner JSON. Run `32767703776` remains a failed, artifact-free predecessor.

RR21 treats the uploader's digest output as one exact representation boundary. A
trusted post-upload built-in-pwsh step accepts only a bare lowercase 64-hex action
value, an exact positive decimal artifact ID, and the current repository/run/ID URL;
it constructs `sha256:<hex>` exactly once and exports only those normalized values.
The verifier retains its single prefixed grammar and exact URL binding. Hosted run
`32771982270` is superseded: its six producers, caches, aggregate, upload, and retained
bytes passed, but its final verifier rejected the unnormalized bare action digest.

RR22 corrects only that normalizer's Ubuntu launch contract. Hosted run
`32777599858` proved all six cells, caches, aggregation, upload, and retained bytes,
but GitHub's built-in shell launched `/usr/bin/pwsh` while the unchanged identity gate
required `/opt/microsoft/powershell/7/pwsh`. The normalizer now uses the exact literal
`-NoLogo -NoProfile -NonInteractive -File {0}` shell already proven elsewhere in the
same job; all digest, URL, output-control, and verifier predicates remain unchanged.
