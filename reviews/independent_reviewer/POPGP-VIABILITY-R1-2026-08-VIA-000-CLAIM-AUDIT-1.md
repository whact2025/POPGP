# Independent claim audit: POPGP-VIABILITY-R1-2026-08 / VIA-000

```yaml
artifact_schema_version: 2
review_id: "POPGP-VIABILITY-R1-2026-08-VIA-000-CLAIM-AUDIT-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-viability-r1-2026-08-via000-claims-session"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "fe11cdd4e1d5ec1881cc1b224a3a9a62cbb1a617"
baseline_commit: "70c867552279b74d5ce1a7bc5c50d5a980cf81e6"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "419b46498d7267f8554692963e9a267592770737"
context_hash_method: "git rev-parse \"fe11cdd4e1d5ec1881cc1b224a3a9a62cbb1a617^{tree}\""
files_reviewed:
  - "README.md"
  - "docs/framework.md"
  - "docs/framework.tex"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/attacks/VIA-000-ATTACK-PLAN-7.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/manifests/VIA-000-HOLDOUT-MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/manifests/VIA-000-SEED-MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/environment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/mutation-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/output-commitment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/raw-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/reveal-record.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/run-log.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-1/environment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-1/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-2/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-mutation-continuation/mutation-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows/environment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows-mutation-continuation/mutation-results.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "scripts/check_viability_campaign.py"
access_level: "public-repository-post-reveal-receipts-only"
independence_statement: |-
  This is a fresh claim-auditor task and session, separate from the builder session,
  under the same human operator and the same Codex Desktop orchestrator. It is internal
  process separation, not external scientific validation. The exact runtime model
  identifier and version are not exposed, so both are recorded as `unknown`. The
  packet also records the builder model as `unknown`; model separation therefore
  cannot be established and is recorded as false.

  This audit used only the public repository at the exact post-reveal handoff. It did
  not access a private custody path, private evaluator logic, credentials, the local
  POPGP_Codex_Handoff.md file, or an unrevealed final label. While verifying the two
  public post-reveal manifest receipts, the now-public seed manifest value was exposed
  to this session. That post-reveal exposure is declared below; it did not precede or
  influence the runner's output commitment, but this seat must not be described as
  seed-blind. The candidate was not rerun after reveal. All execution statements below
  are receipt-level observations or hash/static validation, not a new scientific run.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r1-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: true
  private_evaluator_seen: false
summary: |-
  Scope and immutable identity. This receipt audits campaign snapshot
  fe11cdd4e1d5ec1881cc1b224a3a9a62cbb1a617 (tree
  419b46498d7267f8554692963e9a267592770737), scientific candidate
  9a29e05f803666bf0e3a28417ea399e3e26769fc (tree
  358fb1af6ca587b6c71ff2ef0fb87e335163eeaf), baseline
  70c867552279b74d5ce1a7bc5c50d5a980cf81e6, protocol snapshot
  72bfcfbb5ab5a0fee3449510f8868b8bb19be805, and runner commit
  bd1a8d28717e493cd62053d32bd38062f293c2cd (tree
  4d5ea64131a414a4dca0488bd8b248c6a6e8f5e0).

  Receipt integrity. The independently recomputed SHA-256 of raw-results.json is
  7a221ea1a11dc1033df3dd7d5732a3693ee620dcf6b429d243afa5be6bff2641,
  exactly the hash in output-commitment.json and VIA-000.yaml. All 296 files in the
  raw runner-evidence manifest exist and match both their declared byte counts and
  SHA-256 values; all aggregate receipt hashes also match. The output was generated at
  2026-08-19T16:22:43.110961Z and committed at
  2026-08-19T16:22:43.113962Z, before the custodian reveal at
  2026-08-19T16:35:04.810794Z. The public revealed-manifest hashes match their frozen
  commitments: holdout
  7ff9edcc032afe5ba38153a59d2d7e9c8268f652dd49f4c183b87e54d4cc06d5
  and seed
  13e5e65f49d74b620e47e235e170e7f828305aee0732aa307eda0d78eb4c6719.
  The unmodified post-reveal campaign passed the authoritative static validator.

  Platform result. The Windows exact-candidate run passed every frozen protocol
  command, collected 187 passing tests, completed all six examples and semantic
  validation, left the candidate clean, used the exact declared pdfTeX 1.40.29 / TeX
  Live 2026 engine, produced a nonempty 535368-byte PDF with SHA-256
  ff83b6c28eeacfb805e9261cee744a24f70b2a79fe11c6248de40e77f8356b4b,
  and the committed Windows continuation rejected all ten mutation families. The
  earlier Windows platform record's mutation-harness interruption is preserved rather
  than overwritten; the complete mutation evidence is in the immutable continuation.

  The Linux exact-candidate run also collected 187 passing tests, completed all six
  examples, passed the semantic checker, and completed both PDF passes with the exact
  declared engine. Its 388867-byte PDF was nonempty, but its SHA-256 differed from the
  Windows PDF and also differed between the retained Linux attempts. PDF byte identity
  was not a frozen cross-platform scientific equivalence criterion, so this drift is a
  reproducibility caveat rather than evidence of semantic or physical disagreement.
  More decisively, Linux regeneration changed 16 tracked JSON/PNG artifacts, making
  git diff exit 1, and the postflight found an additional lock-installed
  _cuda_bindings_redirector.pth alongside _virtualenv.pth. A restored Linux attempt
  removed the tracked diff but still failed that startup-surface oracle. Linux rejected
  mutation families G01 through G09. For G10 the editable carrier was removed and the
  isolated probe printed no injected value, but the frozen composite oracle still
  could not return a clean rejection because the unrelated lock-installed startup
  surface failed postflight. The frozen mutation-rejection capability therefore
  remains false; partial mechanism success cannot be substituted for the declared
  Boolean gate.

  Frozen capability mapping. raw-results.json records evidence-contract=false,
  cross-platform-reproduction=false, mutation-rejection=false, failed=true, and
  blocked=false. This is a tested-capability failure, not missing infrastructure. The
  observed facts support a Windows protocol pass and Linux functional/test/PDF passes,
  but they do not support the VIA-000 hypothesis or any cross-platform pass wording.
  No native/CUDA or scalable-backend result was in scope; accelerator_seconds was zero.

  Scientific-claim mapping. VIA-000 cannot promote any of its declared claims. C03
  remains a finite exhaustive stability-selection benchmark, and C04 remains an
  executable but physically unvalidated capacity diagnostic. C06 remains blind
  recovery of encoded locality only in selected finite benchmarks. C08 retains its
  parameter dependence and Bell-control false-positive limitation; C09 remains an
  embedding-coordinate metric prototype without intrinsic/refinement evidence. C10 is
  only a finite graph constraint without an independent operational clock. C11 retains
  the raw-relative-entropy and reduced-modular negative results and only a
  family/decomposition-dependent finite-chain KMS test object. C12 remains a
  Green-function/sign diagnostic rather than a Newtonian result. C13 remains an
  angle-deficit plumbing prototype with closure unimplemented. The claims matrix,
  framework status notes, and top-level README correctly withhold mechanism viability,
  General Relativity, Lorentz recovery, continuum behavior, native scalability, and
  external empirical validation. No scientific overclaim requiring a blocker was
  found in those claim definitions.

  Downstream gating and recommendation. VIA-000 is a necessary Tier-R packet, and
  VIA-010 and VIA-300 depend on it directly; the other Tier-R packets depend on it
  transitively. No dependent holdout may advance unless a VIA-000 round is adjudicated
  passed. Even a future VIA-000 pass would establish evidence integrity only and would
  leave all six other Tier-R packets to satisfy their own E3/E4 gates. For the frozen
  current bytes, the claim-audit recommendation is to retain every existing limitation,
  make no claim promotion, and have the adjudicator evaluate the frozen failed rule
  after all mandatory statistical-audit, claim-diff, review-chain, and adjudication
  receipts exist. Repair or rerun after reveal belongs to a new preregistered round,
  not this one.
findings:
  - id: "VIA000-CLAIM-001"
    severity: low
    category: governance
    location: "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md:20"
    evidence: |-
      At fe11cdd4e1d5ec1881cc1b224a3a9a62cbb1a617, README lines 20-23 say that
      custody reveal remains pending. In the same immutable tree, VIA-000.yaml records
      blind_custody.reveal.status=revealed and receipt reveal-record has SHA-256
      dfaa18a6b1e8c4e7b8a3313a379d2c1cd5177d8aa89e9a3d32ea095f96d8a409.
    finding: |-
      The campaign current-state prose underreports completion of the custody reveal.
      The scientific caveat language and failed Linux description remain accurate.
    failure_scenario: |-
      A downstream reviewer relying on the campaign README rather than the packet and
      hashed receipt could incorrectly treat the claim audit as pre-reveal work.
    consequence: |-
      This is an administrative provenance ambiguity, not a change to the frozen
      capabilities, scientific claims, or predicted packet outcome.
    required_action: |-
      In the normal post-review response/claim-diff lifecycle, update the current-state
      sentence to say the custody reveal is complete while the remaining audits and
      adjudication are pending. Do not alter the frozen protocol or raw results.
    verification: read-only
    blocking: false
requested_tests: []
prior_finding_results: []
prior_requested_test_results: []
predictions:
  experiment_id: "VIA-000-FROZEN-RULE-EVALUATION"
  predicted_outcome: |-
    If the adjudicator verifies the same immutable receipt bytes, the pass expression
    is false, the blocked expression is false, and the failed expression is true.
  predicted_failure_mode: |-
    Any pass or claim promotion would contradict the three false capability Booleans
    and failed=true; any post-reveal repair/rerun would require a new frozen round.
  confidence_statement: |-
    High confidence for the Boolean mapping because it is a read-only evaluation of
    hash-verified exact values; this is not adjudication and does not waive missing
    mandatory receipt kinds or review-chain reconciliation.
recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    Approve the candidate's existing narrowly scoped scientific wording and this
    claim audit, not a VIA-000 pass. There are no blocking claim defects. The one
    nonblocking provenance sentence should be corrected through the review chain.
    The current immutable evidence supports no claim promotion and predicts the
    frozen tested-capability failure outcome pending independent adjudication.
```
