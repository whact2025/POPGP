# VIA-000 R2 pre-holdout protocol-amendment independent review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r2-protocol-amendment-independent-review-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-20"
commit_reviewed: "9d82ddabceee934be8036386af7867c93badc6f9"
baseline_commit: "66eac5c7aede6595bd7402fc9196ea7d29203e6a"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "e557a3014dfca8348efb8507c47c5d6d24fd5dd3"
context_hash_method: "git rev-parse \"9d82ddabceee934be8036386af7867c93badc6f9^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/DISAGREEMENT_LOG_TEMPLATE.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/independent-rereview-v1.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/independent-review-v1.schema.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/review-response-v1.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attacks/VIA-000-R2-ATTACK-PLAN-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/raw-results.schema.json"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
access_level: "public-repository-only plus public GitHub Actions metadata and local Windows execution"
independence_statement: |-
  This was a fresh independent-reviewer task in an isolated worktree at the exact
  remediation commit. The reviewer did not edit candidate, protocol, validator,
  test, campaign, threshold, lifecycle, custody, or outcome material. The same human
  operator and Codex Desktop orchestrator are shared with the builder, so this is
  internal adversarial process separation rather than external scientific validation.
  The reviewer session is distinct. The builder model is recorded as unknown; model
  separation therefore cannot be established and is declared false. The reviewer
  inherited the public scope, Plan-1 findings, and immutable implementation-review
  history, but treated builder claims and hosted CI as hypotheses and reproduced the
  important behavior independently. No hidden custody file, final label, secret seed,
  private evaluator logic, credential, unrestricted private hardware profile, or
  untracked handoff material was accessed.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r2-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with five blocking findings. The amendment fixes both literal
  Plan-1 demonstrations at their original boundary. The exact parent runner exited 1
  on the real `uv 0.11.11 (...)` banner, while the amended runner accepted that banner
  and completed the fresh Windows clean sequence in 1372.5 seconds: 366 tests, six
  generators, semantic/change-boundary validation, two pinned pdfTeX passes, and final
  environment verification all passed. The parent truth-table fixture still accepted
  summary-only results; the new schema rejects that minimal document, and focused raw-
  contract tests passed 4/4.

  Those two controls do not close the evidence contract. An executed positive fixture
  using only synthetic no-op command records, arbitrary text streams, plaintext bytes
  labeled as PDFs, no images, and no mutation records returned zero validator errors
  while claiming both-platform reproduction, twelve visuals, eighteen rejected
  mutations, and every capability. Further counterexamples showed that a wrong per-
  platform commit/tree, an arbitrary blocked Boolean, and a contract/platform-list
  inconsistency can also be accepted when summaries are made internally consistent.
  The runner emits per-platform fragments that do not validate directly against the
  frozen platform schema and whose two-platform relative paths collide, but no frozen
  assembler closes that transformation. Finally, the draft protocol manifest records
  a stale validator digest: `87d01ae47c5b493254c81489a62506203b2e8c6db9594313da6e497015c4520b`
  instead of the reviewed Git-blob SHA-256
  `535bfc82968fcb4376244f0fb036fbd56ae21c801129d63bbeb926a259cd82e3`.

  Independent quality evidence was otherwise green. Ruff passed; TeX source validation
  passed; review guidance passed 9/9; the complete reviewed-tree suite passed 370/370
  in 945.29 seconds; all six generators exited 0; the semantic/change-boundary checker
  passed after all review caches were redirected outside the worktree; schemas and the
  exact primary-protocol envelope validated; protocol/receipt copies and the packet-
  rule hash matched; and `git diff --check` reported no issue. GitHub Actions run
  32408119009 independently completed success at the exact reviewed SHA. No local
  Ubuntu R2 runner was available; hosted Ubuntu CI is supporting quality evidence, not
  a substitute for the frozen two-platform R2 runner.

  R2 remains `preregistered`, `holdout_started: false`, unrevealed, with no raw-result
  or output commitment. The current campaign validator reports the deliberate old
  executing-contract/protocol-snapshot mismatch expected before refreeze. The five
  findings below are additional defects. This review does not authorize refreeze,
  reveal, holdout start, lifecycle advancement, a scientific outcome, or merge.

findings:
  - id: "VIA000-R2-PA1-EVIDENCE-001"
    severity: critical
    category: code
    location: "scripts/check_viability_campaign.py:1454-1609; tests/unit/test_viability_raw_evidence_contract.py:30-151,224-229; protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json:69-163"
    evidence: |-
      The reviewer executed the new positive fixture through
      `_validate_raw_evidence_contract`; it returned `[]`. Every command was a
      synthetic `trusted-command <id>` with arbitrary text stdout/stderr and a
      self-authored result record. The two files marked `application/pdf` began with
      the ASCII text `retained ubuntu-latest-x...` and `retained windows-x86_64 ...`,
      not a PDF header. The manifest contained zero `image/*` entries and no mutation-
      result entries, yet platform records asserted `visual_count: 12`,
      `mutation_count: 18`, all semantic/visual/source/environment/PDF Booleans true,
      and all three packet capabilities true. The helper validates file closure and
      result-record self-consistency, but it neither binds command text to the frozen
      runner nor parses the retained scientific, visual, PDF, test, or mutation
      evidence before trusting those fields.
    finding: |-
      Hash closure proves only that the asserted bytes were retained. It does not prove
      that the frozen commands ran or that the retained bytes support any asserted
      count, gate, PDF, visual, mutation, or capability. A fully synthetic evidence
      package can therefore satisfy the amended passing raw-results contract.
    failure_scenario: |-
      A runner or postprocessor writes arbitrary hash-consistent files, labels text as
      environment/source/PDF evidence, fabricates sixteen successful no-op result
      records, and asserts the frozen counts and Booleans. The authoritative validator
      accepts a passing VIA-000 raw result even though no candidate test, generator,
      visual comparison, PDF build, or mutation executed.
    consequence: |-
      The Plan-1 summary-only channel is narrowed syntactically but remains open
      semantically. VIA-000 could pass without demonstrating evidence integrity,
      cross-platform reproduction, or mutation rejection.
    required_action: |-
      Bind every command ID to a frozen executable/argument contract and derive its
      result from retained runner-owned records. Define typed evidence roles and
      executable validators for pytest counts, generator outputs, semantic operands,
      required raster identities and comparisons, source/environment manifests, PDF
      magic/page/engine/build records, and all eighteen mutation outcomes. Recompute
      every capability from those typed results. Replace the synthetic positive test
      with a real runner-to-raw round trip and add a negative fixture reproducing the
      accepted plaintext/no-image/no-mutation package.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-PA1-IDENTITY-001"
    severity: high
    category: governance
    location: "scripts/check_viability_campaign.py:1293-1318,1430-1609; protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json:31-57"
    evidence: |-
      In a temporary copy of the positive fixture, the reviewer replaced the Windows
      platform commit and tree with unrelated 40-hex values, changed `overall_passed`
      and the cross-platform capability to false, and set `failed: true`. Validation
      returned zero errors. In a separate probe, the contract's required-platform list
      was reduced to Windows while the schema still required both platform records;
      validation again returned zero errors because the Ubuntu record was schema-
      present but ignored by semantic recomputation. Missing/extra platform records,
      stale summaries, and malformed paths were correctly rejected, so these are
      specific cross-binding gaps rather than parser failures.
    finding: |-
      Exact platform identity and agreement among the primary parameter list, raw-
      results contract, and schema are treated as capability inputs rather than
      protocol/evidence validity invariants. Self-consistent summary changes can turn
      wrong-candidate evidence or a one-platform semantic contract into an accepted
      valid failure instead of an invalid round.
    failure_scenario: |-
      A platform executes another commit/tree, or a pre-refreeze edit narrows only the
      contract list while leaving a dummy schema-required record. The raw result is
      accepted and can drive a terminal scientific failure even though the packet says
      candidate/tree identity is exact and malformed evidence invalidates the round.
    consequence: |-
      The adjudicated state can refer to the wrong implementation or an internally
      inconsistent platform contract, corrupting both positive and negative scientific
      provenance.
    required_action: |-
      Make every platform commit/tree an unconditional equality invariant with the
      packet and top-level raw identity. Cross-bind `required_platforms` exactly to
      `platform_families` and the raw schema's required/property set, and cross-bind
      command IDs/count to a frozen canonical command map. Identity or contract-list
      disagreement must be a validator error independent of capability, failed, or
      blocked Booleans.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-PA1-BLOCKED-001"
    severity: high
    category: governance
    location: "scripts/check_viability_campaign.py:1602-1615,3063-3066; reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml:375-421"
    evidence: |-
      Starting from the accepted all-capabilities-true positive fixture, the reviewer
      changed only `blocked` from false to true. The raw-evidence validator returned
      zero errors. `expected_failed` is explicitly computed from the unverified raw
      `blocked` value, and no `expected_blocked` value is derived. At full adjudication,
      the campaign validator requires only the presence of any receipt whose kind is
      `blockage-evidence`; the existing public fixture uses a generic hash-closed
      `{"evidence": true}` document without a blockage schema or causal binding.
    finding: |-
      A raw author can arbitrarily override an otherwise passing result into blocked.
      Neither the blocked Boolean nor the required external prerequisite and cause are
      mechanically derived from typed evidence.
    failure_scenario: |-
      Both platforms and all mutations pass, but the runner sets `blocked: true` and
      attaches any hash-consistent generic blockage receipt. The pass rule becomes
      false, the blocked rule true, and the campaign can record a terminal blockage
      without demonstrating an unavailable prerequisite.
    consequence: |-
      Negative or positive evidence can be suppressed by an unsupported blockage,
      defeating the plan's distinction between tested failure and external
      infrastructure unavailability.
    required_action: |-
      Define and freeze a typed blockage-evidence schema naming the prerequisite,
      observation, command or access attempt, responsible seat, cause code, and
      retained evidence. Derive `blocked` from that receipt and reject blocked when all
      tested capabilities passed or when the claimed prerequisite is part of the
      tested capability. Add pass-to-blocked, failed-to-blocked, dummy receipt, stale
      Boolean, and cause-code mismatch controls through the full campaign validator.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-PA1-ASSEMBLY-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1:278-360; protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json:69-163; reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-1.md:47-55"
    evidence: |-
      The exact amended Windows runner produced sixteen result records, 59 unique
      evidence entries, and a platform summary. Validating that summary against the
      frozen `$defs.platform` schema produced one error: additional properties
      `schema_version`, `campaign_id`, `packet_id`, and `completed_at` are forbidden.
      All emitted evidence paths are rooted as `evidence/...` or `pdf/...`; an Ubuntu
      execution of the same script necessarily emits the same relative names, while
      the combined raw validator rejects duplicate paths and requires one path to map
      to exactly one platform. Aggregation therefore requires stripping fields,
      rebasing every evidence/result/stdout/stderr path, and rewriting both summaries.
      No frozen assembler performs or tests that transformation. The runner also
      throws immediately after a nonzero retained command, before emitting its
      manifest and summary, leaving valid failure assembly unspecified.
    finding: |-
      The supporting runner does not emit schema-compatible platform input and the
      protocol freezes no deterministic two-platform/mutation assembler. The decisive
      raw result and output commitment still depend on an unreviewed manual
      transformation after execution.
    failure_scenario: |-
      The reproduction runner copies both platform directories without rewriting and
      receives duplicate-path/schema errors, or manually rewrites a path/field and
      accidentally or deliberately drops evidence. A clean-command failure emits no
      summary, so the operator improvises a terminal raw result. The final package is
      either unusable or depends on post-run logic absent from the frozen protocol.
    consequence: |-
      The amended protocol is not end-to-end executable as a single deterministic
      evidence contract, and its most security-sensitive transformation lacks both a
      frozen implementation and negative controls.
    required_action: |-
      Freeze an assembler that consumes immutable per-platform runner outputs and the
      committed eighteen-mutation results, assigns collision-free platform-qualified
      paths without changing source bytes, strips or maps metadata explicitly,
      validates the final document against the frozen schema and authoritative
      validator, and creates the output commitment. It must also assemble partial/non-
      zero command evidence deterministically without converting malformed evidence
      into a scientific outcome.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-PA1-FREEZE-001"
    severity: high
    category: governance
    location: "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json:51-54"
    evidence: |-
      `uv run ... check_viability_campaign.py --git-blob-sha256
      9d82ddabceee934be8036386af7867c93badc6f9 scripts/check_viability_campaign.py`
      returned `535bfc82968fcb4376244f0fb036fbd56ae21c801129d63bbeb926a259cd82e3`.
      The changed protocol manifest records
      `87d01ae47c5b493254c81489a62506203b2e8c6db9594313da6e497015c4520b`.
      The packet rule, runner, schema, primary protocol, and their receipt-copy hashes
      did match. The campaign's old protocol-commit mismatch is expected before
      refreeze; this stale digest is inside the proposed new manifest content and is a
      separate byte-binding defect.
    finding: |-
      The proposed protocol manifest does not bind the exact authoritative validator
      implementation reviewed at the handoff commit.
    failure_scenario: |-
      The manifest is snapshotted without correction and activation fails, or its
      recorded digest is trusted despite pointing to neither the reviewed validator
      bytes nor the future snapshot blob.
    consequence: |-
      The protocol cannot be validly refrozen from this manifest, and any later one-
      line repair would create new, previously unreviewed protocol content.
    required_action: |-
      Recompute every manifest contract-file digest from raw Git blobs of the exact
      new remediation candidate, update the manifest before the review handoff, and
      add an exact-SHA audit that compares all manifest entries with the reviewed
      commit. Obtain fresh independent review of the corrected candidate before
      snapshot activation.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    description: |-
      Submit the current synthetic package with no-op commands, plaintext PDF files,
      no image entries, no mutation records, and asserted counts/Booleans through the
      full authoritative campaign path. Require rejection for each missing typed
      evidence role, then demonstrate a real two-platform runner package passes.
    rationale: |-
      Hash/byte closure cannot substitute for semantic validation of the claimed
      evidence; the current positive fixture is an accepted counterexample.
    blocking: true

  - id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    description: |-
      Through the public campaign validator, mutate each platform commit/tree, remove
      or add a platform in each of the parameter/contract/schema locations, and alter
      command IDs/count/map independently. Require an invalid evidence/protocol error
      regardless of self-consistent capability or failed Booleans.
    rationale: |-
      Candidate identity and frozen contract lists are validity invariants, not
      scientific capability outcomes.
    blocking: true

  - id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    description: |-
      Exercise pass-to-blocked and failed-to-blocked flips, absent/dummy/malformed
      blockage receipts, every blockage cause code, and a typed genuine unavailable-
      prerequisite fixture through complete adjudication. Only the fully evidenced
      external prerequisite case may validate as blocked.
    rationale: |-
      The current raw Boolean and generic receipt-kind check allow unsupported
      blockage to override executed evidence.
    blocking: true

  - id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    description: |-
      Run both platform scripts into separate fresh roots, execute all eighteen frozen
      mutation families, assemble with the frozen tool, validate the final raw result
      and output commitment, and prove that duplicate/rebased paths, extra summary
      fields, missing files, changed bytes, failed commands, and partial output are
      handled deterministically and fail closed.
    rationale: |-
      The current emitted fragments require an unspecified manual transformation and
      cannot directly form one schema-valid two-platform package.
    blocking: true

  - id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    description: |-
      For the exact corrected review handoff, recompute every protocol-manifest
      contract-file and packet-rule digest from Git blobs and assert exact equality
      before permitting snapshot, activation, or handoff commits.
    rationale: |-
      The reviewed draft manifest currently records a stale validator digest.
    blocking: true

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "TST-VIA000-R2-PA1-END-TO-END-FALSIFIER-001"
  predicted_outcome: |-
    Without remediation, at least one synthetic hash-closed package, wrong-platform
    identity, unsupported blockage, or manually transformed two-platform package will
    remain accepted or ambiguously classified by the proposed protocol despite the
    literal Plan-1 summary-only and uv-banner controls passing.
  predicted_failure_mode: |-
    The campaign will attribute evidentiary meaning to retained bytes and self-authored
    summaries that are not mechanically tied to the frozen commands, scientific
    checks, visuals, PDF build, mutations, candidate identity, or blockage cause.
  confidence_statement: |-
    High confidence because the dummy-evidence, wrong-identity, arbitrary-blocked, and
    contract-list counterexamples were executed against the exact reviewed validator,
    while the assembly and manifest defects were reproduced from exact runner output
    and Git blobs. No conclusion is drawn about the physical mechanism or hidden
    holdouts.

recommendation:
  approve: false
  blocking_findings: 5
  rationale: |-
    Changes requested. The two Plan-1 blockers are repaired at their literal controls,
    but five independently confirmed defects still prevent the protocol from proving
    its frozen evidence, identity, blockage, assembly, and manifest claims. It is not
    safe to refreeze this commit, and this review grants no permission to reveal data,
    start holdout execution, advance lifecycle, assign a packet outcome, or merge.
```
