# VIA-000 R2 pre-holdout protocol-amendment independent re-review 2

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-protocol-amendment-independent-rereview-session-2"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-20"
commit_reviewed: "71fc1fd6381a79a799ed1d4cc61dda640a22f382"
baseline_commit: "e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a"
prior_review_ref: "87d2d23168c8a34b3a849eaf7f0a01f30dd4a8a8:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-1.md"
builder_response_ref: "71fc1fd6381a79a799ed1d4cc61dda640a22f382:reviews/codex/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-1-RESPONSE-1.md"
context_hash: "9a57e34b2d199478488cfb0595f0bfaaec6ffaf4"
context_hash_method: "git rev-parse \"71fc1fd6381a79a799ed1d4cc61dda640a22f382^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - ".github/workflows/via000-r2-protocol.yml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-3.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attacks/VIA-000-R2-ATTACK-PLAN-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/raw-results.schema.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
  - "uv.lock"
access_level: "public repository plus public GitHub Actions metadata, logs, and downloaded artifacts"
independence_statement: |-
  This was a fresh independent-reviewer task in an isolated worktree and branch at
  exact response-containing handoff 71fc1fd6381a79a799ed1d4cc61dda640a22f382.
  HEAD, tree 9a57e34b2d199478488cfb0595f0bfaaec6ffaf4, branch, origin,
  baseline ancestry, immutable prior-review reference, builder-response reference,
  and clean normal and ignored state were verified before conclusions. Prior review
  commit 87d2d23168c8a34b3a849eaf7f0a01f30dd4a8a8 is preserved as an immutable
  reference and its candidate copy has the same Git blob, while it is intentionally
  not an ancestor because the remediation branch preserved it by cherry-pick.

  Builder claims, positive fixtures, trust-boundary wording, and hosted results were
  treated as hypotheses. All seven prior finding IDs and all seven prior requested-
  test IDs were reconciled exactly once. The coherent no-execution positive, invalid
  PDF, opposed raster, identity, blockage, dirty status, mutation, semantic assembly,
  custody, generated-copy/restore, final-status, Git-blob manifest, packet-rule, CI,
  and exact hosted-artifact paths were independently exercised. The same human
  operator and Codex Desktop orchestrator are shared with the builder. Reviewer task,
  session, worktree, and branch are distinct. Builder model identity remains unknown,
  so model separation is not established. This is internal adversarial separation,
  not external scientific validation.

  No custody file, hidden holdout, secret seed, private evaluator, output commitment,
  final label, restricted result, credential, or untracked handoff memo was accessed.
  R2 remained preregistered, holdout_started=false, and unrevealed throughout.
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
  Changes requested with two blocking scopes. The prior evidence-attribution finding
  remains unresolved under the amended trust model: the fully coherent no-execution
  package still validates and commits, and no authenticated binding distinguishes it
  from bytes supplied by a trusted execution principal. This is an in-scope untrusted-
  transport or assembler-input substitution, not the excluded malicious or colluding
  trusted-principal case.

  One distinct new blocker was found in the exact hosted bytes. The exact Ubuntu
  fragment from run 32429449628 is internally hash-closed and its runner is green, but
  the response-handoff validator rejects its real Linux symlink manifest representation
  and, on Windows, recomputes a retained potential moment outside the registered
  1e-18 absolute tolerance. The exact Windows job remained in pinned TeX
  provisioning and had not produced an artifact when the review was sealed, so an
  exact two-fragment assembly was not executed. That limitation does not weaken the
  counterexample: the unmodified exact Ubuntu fragment alone is rejected by the
  supported Windows-side authoritative verifier before it could participate in a
  valid two-platform commitment.

  The remediation otherwise resolves the six remaining prior findings and six of the
  seven prior requested tests. Invalid PDF bytes, opposed raster drift, contradictory
  identity/count/blocked fields, generic mutation evidence, dirty repository status,
  failed or partial fragments, and semantic invalidity now reject without a commitment.
  The canonical output commitment passes the public synthetic custody/reveal roundtrip.
  Exact ordinary CI run 32429449651 succeeds with 375 tests. Manifest Git-blob hashes,
  protocol receipt copies, and the VIA-000 packet-rule digest reconcile at the handoff.
  The currently activated campaign still points to its older protocol snapshot, so a
  refreeze remains a future post-approval action and is not authorized by this review.
findings:
  - id: "VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1; protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py; scripts/check_viability_campaign.py:_environment_manifest_errors and _validate_raw_evidence_contract; scripts/check_validation_artifacts.py:_bind_potential_summaries"
    evidence: |-
      Exact run 32429449628 Ubuntu artifact 9428747983 has GitHub archive digest
      sha256:03b605380f087b75baf9053a375cc589c85a5b402a05a3539cbf6ddec470c87c.
      It contains 79 unique retained-manifest entries whose sizes and hashes all
      reconcile, 560 source entries bound to candidate 5be3c38/tree 6ad387f, 16
      successful command records and streams, all 18 generated artifacts, 366 passed
      tests, six examples, twelve visuals, the pinned pdfTeX banner, an independently
      parsed and visually inspected eleven-page PDF, an allowlisted generated status,
      and a blank final status.

      Its environment manifest contains 22,215 ordinary five-field entries and four
      genuine Linux symlink entries encoded as {kind,path,target}; the current raw
      validator requires every entry, including a symlink, to have exactly
      {path,kind,mode,size_bytes,sha256}, so it reports `environment-manifest entry is
      malformed`. Independently applying the response-handoff semantic checker to the
      hosted 18 artifacts also reports chain_1d pipeline.pi_time.phi_index_moment
      observed -5.19173662304086e-16 versus Windows-recomputed
      -5.204170427930421e-16. Their 1.24345e-18 difference exceeds the registered
      1e-18 absolute tolerance. The exact Windows job was still provisioning TeX and
      its artifact was unavailable at seal time. Therefore the final two-fragment
      assembler invocation remains unexecuted, but the supported Windows verifier's
      rejection of the exact genuine Ubuntu fragment independently establishes the
      platform-portability failure.
    finding: |-
      A genuine supported-platform fragment that passed the frozen hosted runner is
      not accepted by the authoritative semantic precommit boundary on another
      supported assembler/reviewer platform. The fragment schema-only hosted wrapper
      does not exercise this boundary.
    failure_scenario: |-
      The assigned Linux and Windows runners both complete at the exact scientific
      candidate. Assembly or independent validation on Windows consumes their retained
      fragments. The genuine Linux symlink encoding and platform-sensitive derived
      floating-point statistic are rejected before output, despite an untampered and
      otherwise complete run.
    consequence: |-
      The registered two-platform reproduction cannot deterministically reach the
      pre-reveal output commitment. A green hosted fragment is not evidence that the
      final registered assembly path is feasible.
    required_action: |-
      Freeze one canonical environment-manifest schema that represents files and
      symlinks identically in runner output and raw validation, including a defined
      target/hash/mode policy. Replace the platform-sensitive derived comparison with
      a deterministic representation or independently calibrated cross-platform
      tolerance that still rejects the smallest registered mutation. Then run the
      exact genuine Ubuntu and Windows fragments through the complete raw validator
      and assembler on every supported assembler platform; require semantic success,
      direct visual success, a canonical custody-compatible commitment, and fail-closed
      no-commitment behavior for each malformed/tolerance-boundary control.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    description: |-
      Retain exact control-plane digests for genuine Ubuntu and Windows fragments.
      Assemble the exact extracted bytes with all eighteen executed mutation packages
      on both supported assembler platforms. Require both environment manifests to
      validate under one frozen schema, every lowest-level semantic predicate and all
      direct platform raster comparisons to pass, and the emitted commitment to pass
      custody validation. Perturb each symlink field variant and place the potential
      moment just inside and just outside the calibrated boundary; require invalid
      cases to return nonzero and create neither output nor commitment.
    rationale: |-
      Unit fixtures with a one-file environment and same-platform arithmetic do not
      exercise the real Linux symlink shape or cross-platform recomputation path.
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R2-PA1-EVIDENCE-001"
    outcome: unresolved
    evidence: |-
      The focused suite accepts the structural coherent no-execution fixture and then
      successfully assembles and commits it. Inspection found no producer/principal,
      workflow run, run attempt, job, artifact ID/digest, signature, or attestation in
      the platform fragment or raw-result schema. `--committed-by` is caller-supplied
      public text. A transport/input adversary can therefore replace every byte,
      recompute the self-contained manifest, claim the public runner identity, and
      obtain the same validator and assembler acceptance without collusion by any
      trusted principal.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Minimum remediation is an authenticated external binding from each accepted
      fragment and mutation package to a trusted producer/control-plane event. Bind a
      canonical extracted-manifest digest to the exact head SHA, workflow/run-attempt,
      job/artifact identity, issuer/principal, and immutable control-plane artifact
      digest using a verified signature or attestation; do not trust a package-local
      digest or caller-supplied identity. The assembler must verify that binding before
      semantic work and before output. A coherent replacement with all internal hashes
      recomputed but no valid binding must return nonzero and leave no commitment;
      the exact attested genuine packages must pass.
  - finding_id: "VIA000-R2-PA1-IDENTITY-001"
    outcome: verified-resolved
    evidence: |-
      Independent mutations of top-level and per-platform commit/tree, platform
      parameter/contract/schema sets, command map, and count continue to reject.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Candidate and registered protocol identity invariants remain fail closed."
  - finding_id: "VIA000-R2-PA1-BLOCKED-001"
    outcome: verified-resolved
    evidence: |-
      blocked=true is schema-invalid; nonzero, missing, partial, or unavailable
      fragments and mutations return nonzero and create no commitment. Incomplete
      attempts cannot become scientific pass/fail outcomes.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The packet correctly reserves blocked=false raw results for complete attempts."
  - finding_id: "VIA000-R2-PA1-ASSEMBLY-001"
    outcome: verified-resolved
    evidence: |-
      The assembler now applies the authoritative raw semantic validator before atomic
      rename. Invalid pseudo-PDF, opposed raster, generic mutation, dirty status,
      missing/duplicate/changed/nonzero/partial/unsafe evidence, and contradictory
      identity/count/Boolean cases reject without an output commitment. The canonical
      commitment fields pass the synthetic custody/reveal validator. Hosted Ubuntu
      generated paths are an allowed subset and final status is blank.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The original semantic/no-commitment and custody-shape scope is resolved. Genuine
      cross-platform fragment feasibility is the distinct PA3 portable-assembly finding.
  - finding_id: "VIA000-R2-PA1-FREEZE-001"
    outcome: verified-resolved
    evidence: |-
      At exact handoff 71fc1fd, raw Git-blob SHA-256 recomputation matches all ten
      contract-file entries and the requirements entry. The VIA-000 canonical packet
      rule equals babdbcd9404a3bd75fb776dbd5e676274ec5d8ecb8dd643d386d84760abd2544,
      and all four protocol receipt copies are byte-identical to their sources.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The activated CAMPAIGN still binds protocol commit da5dd8a and its older manifest
      digest. A new activation/refreeze is required only after all blockers are resolved
      and independently approved; this review does not authorize it.
  - finding_id: "VIA000-R2-PA2-VISUAL-XPLAT-001"
    outcome: verified-resolved
    evidence: |-
      The prior minus-four/plus-four opposed raster construction now rejects because
      the validator directly compares corresponding platform pixels; the registered
      maximum direct channel delta remains four. Exact hosted Ubuntu visuals all
      decode and pass the candidate-relative visual contract; the exact hosted
      Ubuntu/Windows pair was unavailable because the Windows artifact was pending.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The direct cross-platform relation, rather than two independent candidate-relative relations, is enforced."
  - finding_id: "VIA000-R2-PA2-CI-HISTORY-001"
    outcome: verified-resolved
    evidence: |-
      Exact-head ordinary CI run 32429449651 is terminal success at 71fc1fd. Its
      full-history checkout resolves historical candidate 5be3c38; lint, TeX source,
      375 tests, six generators, regeneration, and validation-artifact checks succeed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Ordinary exact-SHA quality authority is green."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    outcome: unresolved
    evidence: |-
      Dummy/hash-closed partial packages reject, but the complete coherent synthetic
      no-execution package still validates, assembles, and commits without an external
      authenticated producer binding.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Add a control that recomputes every internal hash in a coherent replacement but
      omits or forges the trusted external attestation; require pre-output rejection.
  - requested_test_id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    outcome: verified-satisfied
    evidence: "Commit/tree, platform-set, schema, command-map, executable/argument, and count contradictions reject independently."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No identity relaxation was found."
  - requested_test_id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    outcome: verified-satisfied
    evidence: "blocked=true and incomplete/unavailable attempts cannot validate, assemble, commit, or become scientific outcomes."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Fail-closed blockage semantics are defensible."
  - requested_test_id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    outcome: verified-satisfied
    evidence: |-
      Focused tests exercise valid assembly, canonical commitment, exact public
      custody/reveal compatibility, and semantic invalidity with no commitment.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The new PA3 test covers real cross-platform inputs rather than the resolved structural roundtrip scope."
  - requested_test_id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    outcome: verified-satisfied
    evidence: |-
      Eleven protocol-manifest Git-blob hashes, VIA-000 packet-rule digest, and four
      receipt copies independently reconcile at exact response handoff 71fc1fd.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "This is exact-handoff byte binding, not authorization to refreeze."
  - requested_test_id: "TST-VIA000-R2-PA2-VISUAL-XPLAT-001"
    outcome: verified-satisfied
    evidence: "The candidate-relative minus-four/plus-four pairwise delta-eight control rejects; the direct bound is recomputed."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Exact hosted raster comparison is retained in the evidence appendix."
  - requested_test_id: "TST-VIA000-R2-PA2-CI-HISTORY-001"
    outcome: verified-satisfied
    evidence: "Exact ordinary CI run 32429449651 succeeds from a full-history checkout with 375 passed tests and all downstream gates green."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The prior shallow-history regression is closed."
predictions:
  experiment_id: "VIA-000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-PREDICTION"
  predicted_outcome: |-
    Without further remediation, a coherent no-execution replacement with recomputed
    internal hashes will validate and commit, while genuine exact hosted two-platform
    fragments will fail the authoritative semantic precommit boundary.
  predicted_failure_mode: |-
    Self-contained hashes cannot authenticate the trusted principal across an untrusted
    transport, and Linux symlink representation plus platform-sensitive floating-point
    recomputation are inconsistent with the platform-neutral assembly claim.
  confidence_statement: |-
    High confidence because both paths were reproduced using the response's own focused
    tests and exact public hosted artifact bytes at the immutable reviewed handoff.
recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    Changes requested. VIA000-R2-PA1-EVIDENCE-001 remains unresolved and
    VIA000-R2-PA3-PORTABLE-ASSEMBLY-001 is a new blocker. The other six historical
    findings and six associated requested tests are independently resolved/satisfied.
    Do not merge or refreeze this protocol amendment, start holdout, access custody,
    reveal, commit a scientific result, advance lifecycle, or adjudicate an outcome.
```

## Executed evidence

| Command or probe | Observed result |
|---|---|
| Exact identity, ancestry, immutable refs, normal and ignored status | Commit `71fc1fd6381a79a799ed1d4cc61dda640a22f382`, tree `9a57e34b2d199478488cfb0595f0bfaaec6ffaf4`, isolated reviewer branch/worktree, expected ancestry/ref preservation, clean before review. |
| Focused raw-evidence, assembler, custody, and review-guidance suite | `18 passed in 514.82s`; includes coherent structural positive, PDF/raster/status negatives, semantic precommit no-output, and custody roundtrip. |
| Coherent replacement and producer-binding audit | Complete no-execution fixture validates and commits; no accepted input carries an authenticated external producer/control-plane binding. |
| Exact ordinary CI run `32429449651` | Terminal success at exact head; `375 passed` plus lint, TeX source, generators, regeneration, and artifact checks. |
| Exact frozen run `32429449628`, Ubuntu artifact `9428747983` | Hosted runner and upload succeeded; archive digest `sha256:03b605380f087b75baf9053a375cc589c85a5b402a05a3539cbf6ddec470c87c`; all retained hashes close, but response-handoff environment/semantic checks reject exact genuine bytes. |
| Exact frozen run `32429449628`, Windows and two-fragment assembly | Windows remained in pinned TeX provisioning and had not produced an artifact at seal time; exact two-fragment assembly was not run. The exact Ubuntu fragment already fails the supported Windows authoritative environment/semantic verifier. |
| PDF parse/render audit | Strict pypdf and Poppler report eleven letter-size pages; all eleven rendered pages are legible with no clipping, collision, missing page, or malformed-object symptom; pdfTeX banner and command logs reconcile. |
| Exact manifest/packet audit | Eleven raw Git-blob SHA-256 entries and four receipt copies match; VIA-000 packet rule is `babdbcd9404a3bd75fb776dbd5e676274ec5d8ecb8dd643d386d84760abd2544`. |

The Windows reviewer seat did not provide a local `pdflatex` or CUDA/nvcc execution
for this review. PDF execution was supplied by the two pinned hosted platform runners;
PDF parsing and rendering were independently repeated locally. VIA-000 registers no
accelerator use, so the available Blackwell processor is outside this protocol check.
