# VIA-000 R2 reproducibility-remediation independent review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-review-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "ad26a33d6183e6855cbbc64564cd8afc9734f7da"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "cdf9c86c98ee9fc46bf144e15f6141748d5416de"
context_hash_method: "git rev-parse \"ad26a33d6183e6855cbbc64564cd8afc9734f7da^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - ".github/workflows/via000-linux-reproduction.yml"
  - ".gitignore"
  - "README.md"
  - "pyproject.toml"
  - "uv.lock"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/independent-review-v2.schema.json"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/adjudication.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/mutation-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/amendments/VIA-000-PREHOLDOUT-AMENDMENT-4.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "examples/physics_qg/gravity_well/__main__.py"
  - "examples/physics_qg/gravity_well/results/source_comparison.png"
  - "examples/physics_qg/grid_2d/__main__.py"
  - "examples/physics_qg/grid_2d/results/clock_potential.png"
  - "examples/physics_qg/grid_2d/results/validation.json"
  - "tests/unit/test_reproduction_boundary.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_validation_artifact_contract.py"
access_level: "public-repository-only plus public GitHub Actions run metadata and logs"
independence_statement: |-
  This is a fresh independent-reviewer task and isolated worktree at the exact frozen
  candidate. It is separate from the builder session and did not edit implementation,
  tests, scientific documentation, thresholds, or candidate artifacts. The same human
  operator and Codex Desktop orchestrator are shared with the builder, so this is
  internal adversarial process separation rather than external scientific validation.
  The runtime exposes neither an exact served-model identifier nor a model snapshot;
  both reviewer model fields are therefore unknown. The builder model is also recorded
  as unknown, so model separation cannot be established and is declared false. The
  reviewer received the predecessor R1 terminal outcome and the builder's R2 scope;
  all builder assertions and green CI evidence were treated as hypotheses and tested
  independently. No R2 final label, secret seed, private evaluator logic, credential,
  or unrestricted private hardware profile was accessed.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-via000-r2-remediation-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with four blocking findings. The candidate correctly preserves the
  terminal R1 valid/failed record, introduces a platform-local hash-bound startup
  manifest, compares raster content rather than encoded PNG bytes, limits accepted Git
  changes to declared artifacts, and renders only display copies of the two
  roundoff-degenerate placeholder fields as canonical zero. Exact GitHub Actions run
  32286876656 is green at ad26a33 on Ubuntu 24.04, and the focused 29-test reviewer
  suite plus static validation passes.

  Those positive controls do not establish the claimed fail-closed boundary. Executed
  counterexamples show that a conspicuous localized image corruption passes both
  global pixel statistics; a near-zero numeric mutation can cross its explicit
  1e-12 scientific criterion while the semantic checker accepts it; a transient
  sitecustomize hook can execute and self-delete so the startup postflight returns no
  error; and a tracked source mutation hidden with Git's assume-unchanged bit is absent
  from the residue check. Ignored untracked residue is also omitted. These are direct
  violations of the new gates, not requests for stronger evidence after an otherwise
  correct implementation.

  Verification retained no implementation edits. `git diff --check
  d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4..ad26a33d6183e6855cbbc64564cd8afc9734f7da`
  returned zero; `uv run --isolated --frozen --no-editable ruff check` on every changed
  Python file returned `All checks passed`; `python -m pytest -q` over the three
  affected unit modules returned `29 passed in 25.22s`; and
  `python scripts/check_validation_artifacts.py --enforce-change-boundary` passed on
  the unchanged checkout. The GitHub API reports run 32286876656 completed success at
  the exact candidate, including the full test and example regeneration sequence.
  The current ordinary CI does not invoke the startup-manifest tool or supply current
  Windows evidence, so it cannot resolve the executed counterexamples.

findings:
  - id: "VIA000-R2-VISUAL-001"
    severity: high
    category: code
    location: "scripts/check_validation_artifacts.py:61-63,388-457; tests/unit/test_validation_artifact_contract.py:214-223"
    evidence: |-
      The visual oracle aggregates error over every RGBA channel using normalized mean
      error <= 2/255 and large-channel fraction <= 0.02. The registered negative
      control changes the whole image and does not exercise localized structure. In a
      temporary file, the reviewer loaded the committed 1800x750
      gravity_well/results/source_comparison.png, replaced a centered 180x71 rectangle
      (0.9467% of pixels) with opaque red, saved the same PNG geometry/mode/frame count,
      and called compare_visual_artifact(reference_bytes, mutated_path). The mutation
      produced normalized MAE 0.004465260 (<0.007843137) and large-channel fraction
      0.004932222 (<0.02); the function returned an empty error list. The candidate
      repository was not modified.
    finding: |-
      Global average/fraction thresholds permit conspicuous localized and thin
      structured corruption. A plot line, annotation, localized source, or compact
      feature can be removed or overwritten while occupying too few pixels to cross
      either aggregate threshold.
    failure_scenario: |-
      Cross-platform regeneration omits or displaces a scientifically meaningful thin
      curve or compact feature, or a mutation deliberately overwrites a small plot
      region. Format, dimensions, and aggregate pixel statistics remain within the
      current envelope, so VIA-000 records visual equivalence and mutation rejection
      even though the diagnostic content materially changed.
    consequence: |-
      GATE-PORTABLE-ARTIFACT-CONTRACT does not satisfy its declared material-visual-
      mutation boundary, and a future R2 packet could pass on visibly corrupted raster
      evidence.
    required_action: |-
      Add a locality/structure-aware visual contract whose thresholds are justified by
      retained honest Windows/Linux drift, such as bounded tile/region maxima plus
      edge, curve, feature, or perceptual correspondence appropriate to each plot.
      Demonstrate rejection of compact-region overwrite, thin-curve removal or
      displacement, annotation removal, and near-limit structured mutations while
      retaining acceptance of honest platform re-encoding.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-SEMANTIC-001"
    severity: high
    category: science
    location: "scripts/check_validation_artifacts.py:40-41,222-259; examples/physics_qg/grid_2d/results/validation.json:364-453"
    evidence: |-
      The default diagnostic absolute tolerance is 5e-9, while the registered
      placeholder-source criterion is `effective_source_norm < 1e-12 and phi_range <
      1e-12`. The reviewer deep-copied the committed grid validation document, changed
      both canonical copies of phi_range from 2.7514450339362562e-16 to 4e-9, retained
      the exact criterion and `passed: true`, and invoked compare_validation_documents
      plus check_validation_semantics. The comparator reported two accepted numeric
      drifts and zero errors; semantic validation also returned zero errors. The
      mutated 4e-9 value is 4000 times the criterion limit. GitHub run 32286876656
      independently shows that the Linux checker accepts many near-zero relative
      drifts (reported maximum relative drift up to 2), confirming this is an active
      policy rather than dead code.
    finding: |-
      Numeric portability tolerances are not constrained by the scientific criteria
      they accompany, and semantic validation does not recompute those criteria from
      the retained values. An artifact can therefore be accepted with exact gate text
      and Boolean identity but measurements that contradict the Boolean.
    failure_scenario: |-
      A regenerated or tampered near-degenerate diagnostic crosses a pass/fail
      threshold by less than the generic 5e-9 absolute allowance. Because the stale
      Boolean remains exact, both document comparison and semantic validation accept
      an internally false scientific gate.
    consequence: |-
      The machine-readable raw evidence is not self-consistent at the acceptance
      boundary; exact Boolean preservation cannot prove contract-equivalent science.
    required_action: |-
      Recompute every decision-relevant check from typed retained values or attach an
      executable predicate with explicit operands. At minimum, cap each accepted
      diagnostic drift strictly inside its frozen decision margin and reject any
      candidate value that changes the predicate outcome. Add a negative control using
      the exact 4e-9 mutation (and analogous threshold-adjacent cases).
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-STARTUP-001"
    severity: high
    category: code
    location: "scripts/check_reproduction_boundary.py:91-155; tests/unit/test_reproduction_boundary.py:42-79"
    evidence: |-
      In a temporary repository, the reviewer created a clean venv and used
      write_snapshot to commit its startup surface to an external manifest/digest.
      After the snapshot, the reviewer injected site-packages/sitecustomize.py. Its
      module body wrote an external marker and unlinked its own file, then a venv
      interpreter executed `-c pass`. The child returned zero, the marker existed, and
      the hook no longer existed. A base-process call to verify_snapshot against the
      original digest returned `[]`. This directly exercises the same customize-file
      surface the implementation inventories. The unit negative control covers only a
      persistent attack.pth and never starts an interpreter. In addition,
      write_snapshot itself does not enforce the base-interpreter or blocked-
      environment rules that verify_snapshot applies only later.
    finding: |-
      Pre/post filesystem equality is vulnerable to a transient self-cleaning startup
      hook: arbitrary code can execute and restore the measured surface before
      postflight. The new mechanism therefore repeats the persistence assumption that
      the R1 self-cleaning attack disproved.
    failure_scenario: |-
      A .pth/customize carrier is placed after the snapshot, runs before the candidate
      command, alters evidence or imports, and deletes itself. Final surface equality
      and the runner-held manifest digest both remain valid, so the command is accepted
      despite confirmed startup code execution.
    consequence: |-
      GATE-STARTUP-SURFACE-INTEGRITY cannot establish that candidate commands ran
      without undeclared startup state; all downstream tests and artifacts can be
      controlled while the final oracle is clean.
    required_action: |-
      Make unmeasured startup execution impossible or attest execution with a base-
      owned mechanism that cannot be erased by the child. For example, launch the
      target with site processing disabled and add only verified dependency paths
      without evaluating .pth/customize code, or use a separately monitored sandbox
      that records startup opens/execution. Enforce the base interpreter and blocked
      environment at snapshot creation as well as verification. Add both transient
      .pth and sitecustomize/usercustomize attacks that execute markers, self-delete,
      persist, or restore bytes; the gate must reject every execution, not merely a
      dirty postflight.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-RESIDUE-001"
    severity: high
    category: code
    location: "scripts/check_reproduction_boundary.py:159-180; scripts/check_validation_artifacts.py:580-607; tests/unit/test_reproduction_boundary.py:82-112"
    evidence: |-
      Both residue implementations trust `git diff --name-only HEAD` and enumerate
      untracked paths with `git ls-files --others --exclude-standard`. In a temporary
      Git repository, the reviewer committed source.py, ran `git update-index
      --assume-unchanged source.py`, and changed its contents. `git diff --name-only
      HEAD` returned an empty string and check_repository_residue(root, set()) returned
      `[]`; `git ls-files -v source.py` showed `h source.py`. Separately, an ignored
      untracked file also produced `[]`. The existing negative control changes an
      ordinary tracked file and a non-ignored untracked file, so neither bypass is
      tested.
    finding: |-
      The generated-path/source-cleanliness gate trusts mutable Git index flags and
      deliberately omits ignored working-tree state. It therefore does not compare
      actual source/configuration bytes to the frozen tree and does not enforce its
      stated untracked-residue boundary.
    failure_scenario: |-
      A candidate command, startup hook, or adversarial mutation marks a tracked source
      path assume-unchanged or skip-worktree before changing it, or places executable
      state under an ignored pattern. The scientific run consumes changed bytes while
      both residue checkers report clean.
    consequence: |-
      R2 can accept undeclared source/configuration/runtime state, so regenerated
      artifacts are not securely bound to the frozen candidate tree.
    required_action: |-
      Reject assume-unchanged and skip-worktree entries and hash actual working-tree
      bytes/modes for every tracked path against the frozen tree without trusting index
      lstat optimizations. Enumerate ignored state and allow only a narrow, separately
      verified runtime set (for example the exact environment and known caches), while
      rejecting ignored executable/package/source residue. Apply the same primitive to
      both checkers and add mutations for assume-unchanged, skip-worktree, ignored
      executable state, staged changes, deletions, renames, and symlinks.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    description: |-
      Calibrate the visual envelope with retained honest Windows/Linux outputs, then
      show that compact-region overwrite, thin-curve removal/displacement, annotation
      deletion, structured low-area corruption, and just-below-limit mutations are all
      rejected for every declared raster type.
    rationale: |-
      The current all-image color mutation does not cross the gate at the smallest
      scientifically meaningful spatial scale; the executed 0.9467%-area corruption
      survives.
    blocking: true

  - id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    description: |-
      Mutate every decision-relevant numeric operand to both sides of its declared
      criterion while preserving the serialized Boolean and criterion text; require
      semantic rejection whenever the recomputed outcome differs. Include grid
      phi_range=4e-9 against the 1e-12 criterion.
    rationale: |-
      A generic numeric tolerance must never be wider than a scientific decision
      margin or permit retained values to contradict a passing gate.
    blocking: true

  - id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    description: |-
      After the locked snapshot, inject persistent, self-deleting, and byte-restoring
      .pth, sitecustomize.py, and usercustomize.py carriers. Execute the actual R2
      command wrapper on Windows and Linux and require both that no marker executes and
      that the boundary rejects every attempted carrier. Also prove snapshot creation
      fails under a virtual interpreter or any blocked environment variable.
    rationale: |-
      Final manifest equality alone cannot observe code that executes and removes or
      restores its carrier before postflight.
    blocking: true

  - id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    description: |-
      Mutate tracked source/configuration bytes under assume-unchanged and skip-worktree
      flags and create ignored executable/package residue, plus staged, deletion,
      rename, and symlink variants. Require the actual common source-boundary primitive
      to reject each on Windows and Linux while allowing only explicitly verified
      generated artifacts and runtime directories.
    rationale: |-
      Git porcelain/diff output is not a byte-level oracle when index flags and ignored
      patterns can suppress paths.
    blocking: true

  - id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    description: |-
      After remediation and before any R2 holdout, run one immutable preregistered
      protocol at the exact new candidate on clean Windows and Linux runners. Retain
      startup snapshots/digests, raw JSON, regenerated visuals, source-byte manifests,
      command logs, and every negative-control result. Verify that canonical-zero
      rendering changes only display copies and preserves the raw phi/source
      diagnostics, while a just-above-threshold nondegenerate fixture is not zeroed.
    rationale: |-
      Run 32286876656 is useful Linux calibration evidence but does not invoke the
      startup boundary, does not close the four bypasses, and supplies no current
      Windows execution for the R2 candidate. R1 evidence belongs to a different
      frozen candidate and rule.
    blocking: true

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-001"
  predicted_outcome: |-
    Without remediation, localized low-area raster corruptions, threshold-crossing
    near-zero diagnostics with stale Booleans, transient self-cleaning startup hooks,
    and source mutations hidden by Git index flags will continue to pass their current
    component oracles despite ordinary CI success.
  predicted_failure_mode: |-
    A future R2 campaign may report contract-equivalent visuals, exact scientific gate
    identities, unchanged startup surfaces, and clean source state even though a
    material visual feature changed, retained values contradict a criterion, startup
    code executed, or source bytes differed from the frozen tree.
  confidence_statement: |-
    High confidence because each predicted failure class was independently reproduced
    against the exact frozen implementation. No conclusion is drawn about POPGP's
    physical mechanism; this review addresses only evidence-integrity implementation.

recommendation:
  approve: false
  blocking_findings: 4
  rationale: |-
    Changes requested. Four executed counterexamples violate the exact gates this R2
    remediation introduces. The terminal R1 failure remains valid and immutable, and
    the R2 candidate makes useful progress, but it is not ready to seed a new
    preregistered viability round until every blocker and requested test is addressed
    and independently re-reviewed at a new frozen commit.
```
