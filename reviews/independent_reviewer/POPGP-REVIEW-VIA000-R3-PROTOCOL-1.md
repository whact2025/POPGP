# VIA-000 R3 recovery-protocol independent review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-review-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "46f33080eae18f4692624d3172027635d72aff97"
baseline_commit: "6e0e0c8ebaecef6d129c68666f113fbd47af4ce7"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "ccc398d73a6d7b1c61d933ec15c13644d719bba0"
context_hash_method: 'git rev-parse "46f33080eae18f4692624d3172027635d72aff97^{tree}"'
files_reviewed:
  - ".gitattributes"
  - ".github/workflows/ci.yml"
  - ".github/workflows/via000-r3-protocol.yml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-010.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-100.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-150.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-200.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-300.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-400.json"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attempts/VIA-000-R2-INVALID-ATTEMPT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-2.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-3.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-4.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/amendments/VIA-000-R3-RECOVERY-DESIGN-1.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-010.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-100.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-150.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-200.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-300.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-400.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/dispatch-guard.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/mutation-runner.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/raw-results.schema.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-010/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-100/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-150/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-200/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-300/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-400/protocol.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
access_level: "public-repository-only plus local Windows execution; no custody, reveal, or external invalid-package access"
independence_statement: |-
  This was a fresh independent-reviewer task in a new isolated worktree at the exact
  R3 handoff. The reviewer edited no implementation, protocol, packet, campaign,
  lifecycle, custody, threshold, result, or reveal material. The same human operator
  and Codex Desktop orchestrator are shared with the builder, so this is internal
  adversarial process separation rather than external scientific validation. The
  reviewer session is distinct. Builder model identity was not supplied and remains
  unknown, so model separation is not asserted. Builder tests and claims were treated
  as hypotheses. No custody directory, sealed manifest, secret seed, hidden label,
  reveal material, external R2 invalid package, credential, or untracked handoff file
  was accessed.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with three blocking findings. The R3 draft preserves the R1 and
  R2 records, leaves every R3 packet drafted, holdout-unstarted, unrevealed, pending,
  and carries forward only fourteen public R2 URI/SHA-256 commitment pairs behind an
  explicit fresh-custodian verification gate. The supplied handoff, content snapshot,
  and R2 parent commit/tree identities all matched. The two-commit draft construction
  is internally coherent, all six declared execution-artifact SHA-256 values match the
  exact `8aa548ab4b24373634e114606f055b42da570bcd` Git blobs, protocol receipt copies
  reconcile, the R2 tree is unchanged, and the drafted campaign validator passes.

  The central recovery claim is nevertheless false. A content-addressed name proves
  that a selected tag resolves to the hash written in that tag's own name; it does not
  prove that the selected commit is the campaign-authorized protocol snapshot. In an
  independent temporary-repository probe, a later lifecycle commit with its own exact
  content-addressed R3 tag passed the dispatch guard. The assembler accepts the same
  caller-selected identity, does not read the campaign packet, and its purported
  pre-commit public validator constructs the expected packet protocol commit from the
  raw document itself. A later lifecycle run can therefore again produce an output
  commitment that only the subsequent full campaign validator rejects, reproducing
  the decisive R2 defect rather than closing it.

  The assembler's semantic gate is also not source-frozen. It verifies six protocol
  artifact hashes from Git, but imports `scripts/check_viability_campaign.py` and its
  validation dependencies from caller-provided mutable worktree bytes. The packet
  explicitly labels assembler inputs untrusted, yet the protocol-manifest validator
  digest is never consumed by the assembler before commitment.

  Independent quick checks were otherwise green: the campaign validator passed,
  changed-Python Ruff passed, TeX source validation passed, review-guidance tests
  passed, Git-blob hashes and receipt-copy bindings passed, and `git diff --check`
  passed. The mandatory R3 positive identity test fails on this ordinary Windows
  checkout: the workflow path lacks an LF attribute, its worktree copy is CRLF, and
  the test commits those bytes into a `core.autocrlf=false` fixture before comparing
  them with the declared LF Git-blob hash. The exact Git blob is correct; the claimed
  Windows/Linux happy-path control is not. The complete suite correspondingly had
  388 passes and 1 failure in 1899.07 seconds, with the same
  `test_r3_assembler_exact_snapshot_ref_happy_path` failure. This review authorizes
  no tag, refreeze, activation,
  preregistration, custody transition, falsifier run, holdout, assembly, commitment,
  reveal, scientific outcome, or merge.

findings:
  - id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    severity: critical
    category: governance
    location: ".github/workflows/via000-r3-protocol.yml:40-50; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py:14-73; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py:91-141,330-460; scripts/check_viability_campaign.py:2501-2544; reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml:8,302-308,332-356; tests/unit/test_via000_r3_identity.py:260-299"
    evidence: |-
      The workflow passes the dispatcher-controlled
      `inputs.protocol_snapshot_commit` to the guard and runners. The guard derives
      the expected tag name from that same value and checks only that the named tag,
      `github.sha`, and HEAD agree with it. It has no campaign-authorized snapshot
      input. The reviewer created a temporary repository at a valid snapshot, added
      one later lifecycle commit, and created
      `refs/tags/popgp-via000-r3-protocol-<later-commit>` at that later commit. Calling
      `verify_workflow_dispatch` with the exact later ref/SHA/input returned normally
      and printed `ACCEPTED_CONTENT_ADDRESSED_LIFECYCLE`. The existing lifecycle test
      instead advances HEAD while keeping the old tag/input, so it does not exercise
      this counterexample.

      The packet binds `protocol_commit` to the campaign snapshot, but the assembler
      receives no campaign or packet argument. It builds its expected identity solely
      from caller arguments. Its pre-commit call to
      `validate_via000_raw_results` cannot repair that gap: the wrapper at line 2523
      sets the synthetic packet's `protocol_commit` from
      `raw_document['protocol_source_commit']`. The full campaign validator will later
      compare against the real packet, but only after the assembler has atomically
      written raw results and an output commitment. This is the same ordering defect
      recorded for the invalid R2 attempt.
    finding: |-
      Content addressing enforces internal equality but supplies no authorization of
      which content-addressed commit is the frozen R3 protocol snapshot. An exactly
      tagged branch, activation, handoff, or later lifecycle commit is accepted by the
      guard, workflow, attestations, assembler, and pre-commit validator.
    failure_scenario: |-
      After final snapshot activation, a later lifecycle commit is tagged with its own
      hash suffix and manually dispatched with that suffix as input. Both signed jobs
      and the assembler consistently bind the later commit. The assembler emits a
      commitment; only attaching the package to the campaign exposes that it differs
      from the packet's frozen `protocol_commit`, exactly as in R2.
    consequence: |-
      R3 does not establish the claimed single-valued identity or fail-before-output
      guarantee. A committed invalid attempt can consume the one-shot holdout and
      force yet another round without producing scientific evidence.
    required_action: |-
      Add a non-self-authored authorization binding for the one permitted protocol
      snapshot, such as a frozen/signed activation record consumed by the dispatch
      guard and assembler. The assembler's pre-commit validation must load the real
      campaign packet (or an equivalent immutable authorization object), prove its
      `protocol_commit` equals the selected source, and reject before creating output
      for every other correctly content-addressed commit. Freeze an exact assembly
      invocation and packet/authorization input rather than relying on caller-supplied
      source identity. If tag kind or signature carries authority, specify and verify
      it explicitly; name syntax alone is insufficient.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py:91-131,437-452,467-488; reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json:1-65; reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml:320-338"
    evidence: |-
      `_verify_protocol_identity` checks the selected Git blobs and declared hashes
      for runner, raw schema, assembler, mutation runner, dispatch guard, and workflow.
      It does not verify or extract `scripts/check_viability_campaign.py` or the
      validator modules that script imports. Later, the honest frozen assembler adds
      caller-provided `--repo-root` to `sys.path` and imports
      `validate_via000_raw_results` from the mutable worktree. The protocol manifest
      records a Git-blob digest for `scripts/check_viability_campaign.py`, but the
      assembler neither accepts nor validates that manifest. The primary packet
      explicitly lists `assembler-inputs` among untrusted surfaces.
    finding: |-
      The pre-commit semantic validator is a mutable, caller-selected dependency rather
      than code bound to the protocol-source commit. Modifying the validator worktree
      does not violate any assembler identity check.
    failure_scenario: |-
      The selected Git objects and tag are correct, but the assembly checkout contains
      an uncommitted or substituted `scripts/check_viability_campaign.py` (or imported
      validation dependency) that returns no errors for invalid evidence. The frozen
      assembler imports that file and writes the output commitment. A later clean full
      campaign validation may reject it, but the fail-before-commitment guarantee is
      already lost.
    consequence: |-
      Source substitution can bypass the assembler's authoritative semantic gate while
      retaining correct declared hashes for every artifact the assembler currently
      checks. The claim that untrusted assembler inputs are validated before output is
      not met.
    required_action: |-
      Execute the pre-commit validator and all relevant imported validation modules
      from bytes extracted from the exact authorized Git object, under an isolated
      interpreter and a closed import path, or equivalently verify every executed byte
      against a complete frozen dependency manifest before import. Bind and validate
      the real campaign packet/manifest in the same operation. Add a negative control
      that mutates each validator dependency only in the worktree and proves no output
      directory, temporary residue, raw result, or commitment survives.
    verification: read-only
    blocking: true

  - id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    severity: high
    category: code
    location: ".gitattributes:1-2; tests/unit/test_via000_r3_identity.py:52-73,341-359; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py:115-131"
    evidence: |-
      The system Git configuration on the review machine is `core.autocrlf=true`.
      `.gitattributes` pins LF only under `reviews/viability/**` and `protocols/**`;
      `git check-attr` reports text/eol unspecified for
      `.github/workflows/via000-r3-protocol.yml`. Its reviewed Git blob and receipt
      have the correct declared SHA-256
      `70d1065721025bb2951b8f55be437e9c61e8911d6171a9b0b3e30dc8dd046c9a`,
      while this ordinary Windows worktree has CRLF bytes with SHA-256
      `42424df64fd8adeb3cee1b4f4c1bcf4c07aa93d0ec333ac1fd6e80df920d5b32`.

      `_snapshot_repo(protocol_tree=True)` copies those worktree bytes, configures the
      destination `core.autocrlf=false`, and commits them. The mandatory
      `test_r3_assembler_exact_snapshot_ref_happy_path` then fails at
      `_verify_protocol_identity` with `ValueError: frozen protocol artifact hash
      mismatch: .github/workflows/via000-r3-protocol.yml`. The exact focused command
      was `uv run --frozen python -m pytest -vv -p no:cacheprovider
      tests/unit/test_via000_r3_identity.py::test_r3_assembler_exact_snapshot_ref_happy_path`.
    finding: |-
      The registered exact-snapshot positive control is platform-dependent and fails
      on a normal Windows checkout. It does not establish the claimed Windows/Linux
      Git-normalized happy path, and the complete quality suite is red.
    failure_scenario: |-
      A reviewer or CI runner with the ordinary Git-for-Windows line-ending policy
      executes the frozen identity tests. The test manufactures a different workflow
      blob from converted worktree bytes and rejects the otherwise correct declared
      Git digest.
    consequence: |-
      The required positive control and full-suite gate fail on one of the two target
      platforms. This also masks whether later identity changes preserve true raw
      Git-byte portability.
    required_action: |-
      Make workflow byte normalization explicit (for example with a suitable
      `.gitattributes` rule) and/or construct the temporary snapshot fixture from exact
      Git blobs rather than converted source-worktree bytes. Re-run the exact positive
      and negative controls plus the complete suite on both Windows and Linux, and
      retain the Git-blob/worktree/receipt hashes that demonstrate the intended byte
      model.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    description: |-
      Starting from the final two-commit construction, exercise the authorized
      snapshot and then branch, activation, handoff, later lifecycle, moved, deleted,
      wrong-suffix, malformed, annotated, and lightweight ref cases. In particular,
      give each unauthorized lifecycle commit its own correct content-addressed tag
      and matching SHA/input. Only the single externally authorized snapshot may pass.
      Run each rejection through the real guard, hosted-dispatch-equivalent inputs,
      unmocked assembler identity/attestation boundary, and full campaign packet, and
      prove zero output directory, temporary assembly, raw result, or commitment.
    rationale: |-
      The current tests reject disagreement with a caller-selected commit but do not
      distinguish the authorized snapshot from another internally consistent commit.
    blocking: true

  - id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    description: |-
      Mutate `scripts/check_viability_campaign.py` and each imported semantic/visual
      validator dependency only in the assembler worktree while leaving the selected
      Git commit, tag, protocol, schema, and signed fragments intact. Require rejection
      before output. Demonstrate that the accepted path executes exact authorized Git
      blobs under a closed import surface and validates the real packet protocol
      commit. Repeat caller-controlled repository, protocol, schema, run ID, attempt,
      platform-root, and output-path substitutions without monkeypatching identity or
      attestation functions.
    rationale: |-
      The current assembler's final gate imports mutable caller-provided code, and the
      current assembler negative tests replace the functions under test with lambdas.
    blocking: true

  - id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    description: |-
      In fresh Windows and Linux clones with their ordinary Git configurations, assert
      exact equality among every declared execution-artifact Git-blob SHA-256 and its
      frozen receipt copy, then run the exact-snapshot happy path and all identity
      negatives. Include Windows `core.autocrlf=true` and `false` fixtures and the
      complete quality suite.
    rationale: |-
      The current positive fixture commits converted Windows workflow bytes and fails
      its own declared hash contract.
    blocking: true

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "TST-VIA000-R3-P1-END-TO-END-RECOVERY-001"
  predicted_outcome: |-
    Without remediation, at least one correctly content-addressed non-snapshot
    lifecycle commit will pass the R3 guard and reach commitment, a mutable assembler
    validator dependency will remain unchecked, and the exact-snapshot positive test
    will remain red under an ordinary Windows checkout.
  predicted_failure_mode: |-
    R3 will mistake internal identity consistency for campaign authorization, then
    perform its pre-commit semantic check with code not bound to that identity. A later
    full campaign check will discover the mismatch only after the one-shot commitment,
    while Windows cannot independently demonstrate the nominal happy path.
  confidence_statement: |-
    High confidence. The lifecycle-tag and Windows counterexamples were executed at
    the exact reviewed tree; the assembler-to-validator source gap and self-authored
    packet expectation follow directly from the frozen code paths. No conclusion is
    drawn about hidden holdouts or the physical mechanism.

recommendation:
  approve: false
  blocking_findings: 3
  rationale: |-
    Changes requested. R3 preserves predecessor science and improves internal run
    identity, but it does not authorize one unique protocol snapshot before output,
    does not source-freeze the assembler's final semantic validator, and does not have
    a passing Windows exact-snapshot control. Refreeze, tag creation, activation,
    preregistration, custody access, holdout execution, assembly, reveal, and merge
    remain prohibited until a fresh independent re-review verifies all three findings
    and requested tests with zero blockers.
```
