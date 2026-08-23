# VIA-000 R3 recovery-protocol independent re-review 2

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-2"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "7e194f89f789a89ec459dfb7504dbdf6a3f901ed"
baseline_commit: "e083dc79c204d40d218e10436e65ebfa06e79345"
prior_review_ref: "e083dc79c204d40d218e10436e65ebfa06e79345:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
builder_response_ref: "7e194f89f789a89ec459dfb7504dbdf6a3f901ed:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1-RESPONSE-1.md"
context_hash: "6fa1682d166813e6aba2fe0f3fddabfcb8074be6"
context_hash_method: 'git rev-parse "7e194f89f789a89ec459dfb7504dbdf6a3f901ed^{tree}"'
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
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-VALIDATOR-PACKAGE-INIT.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
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
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/VIA-000-AUTHORIZED-SIGNERS"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-010.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-100.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-150.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-200.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-300.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-400.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/authorization-signers"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/dispatch-guard.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/mutation-runner.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/raw-results.schema.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/validator-package-init.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
access_level: "public-repository-only plus local Windows execution; no custody, reveal, external invalid-package, or handoff-memo access"
independence_statement: |-
  This was a fresh independent re-review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-2 on dedicated branch
  review/via000-r3-protocol-rereview-2. Exact handoff, content, parent, tree,
  response, review, ancestry, and byte-immutability identities were verified before
  review. Builder claims and tests were treated as hypotheses. The reviewer changed
  no implementation, protocol, packet, campaign, signer, ref, lifecycle, evidence,
  commitment, custody, threshold, result, or reveal material.

  The same human operator and Codex Desktop orchestrator are shared with the builder.
  Reviewer task, session, branch, and worktree are distinct. Builder model identity
  remains unknown, so model separation is not asserted. This is internal adversarial
  process separation, not external scientific validation. No custody directory,
  sealed manifest, hidden label, secret seed, reveal material, external invalid
  output, credential, or C:\src\POPGP\POPGP_Codex_Handoff.md was accessed.
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
  CHANGES REQUESTED with one new critical blocker. The remediation correctly closes
  the ordinary mutable-ref race: a full authorization tag object ID is captured,
  compared with the ref, and used for type checking, message parsing, target peeling,
  SSH signature verification, returned identity, and a final ref-equality check.
  The invalid-captured-object/same-name valid-object swap, delete-after-parse,
  move-after-peel, swap-before-signature, and assembler post-capture swap all reject.
  The direct captured-object happy path passes. Packed refs pass; a loose ref shadowing
  the packed value, abbreviated object input, and nested annotated tag reject.

  Git object replacement remains enabled, however. Git transparently substitutes
  `refs/replace/<captured-oid>` when `cat-file`, peel, and `verify-tag` receive the full
  captured OID. A real independent repository mapped an invalidly signed canonical
  authorization tag object to the original valid signed tag without moving the
  authorization ref. The guard accepted and returned the invalid OID; direct
  verification of that exact object with replacement disabled exited 1. The
  assembler's independently extracted guard accepted the same construction. Full OID
  syntax therefore does not establish immutable-byte identity while Git replacement
  refs and replacement-control environment are inherited.

  The four prior findings are resolved in their original scopes and all four prior
  requested tests are satisfied in those scopes. R2 remains immutable. Seven
  execution-artifact Git hashes and receipt copies, the validator manifest closure,
  both response schemas, Ruff, TeX, campaign validation, and the 28-case R3 suite are
  green. Compatibility tests are reported in the verification ledger. The full suite
  collects 407 tests but was not executed after the decisive critical reproducer.
  The comment-only signer still blocks authorization, no local or origin R3 tag
  exists, all seven packets remain drafted with holdout_started=false, and the
  campaign remains pending and unrevealed. The public commitment carry-forward still
  requires a fresh private fourteen-of-fourteen custodian verification.

  This review authorizes neither merge nor a signer-key amendment. It authorizes no
  tag, snapshot/refreeze, activation, preregistration, custody transition, falsifier
  run, holdout, evidence assembly, commitment, reveal, or scientific outcome. After
  the blocker is fixed and independently re-reviewed, any approval must remain
  limited to the approved protocol bytes and a separately reviewed one-key signer
  amendment.
findings:
  - id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    severity: critical
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py:22-31,145-185,246-273,307-325; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py:98-107,110-207; .github/workflows/via000-r3-protocol.yml:40-59,78-101,135-159"
    evidence: |-
      Both `_git` and `_git_output` invoke ordinary `git -C ...` in the inherited
      process environment. The direct `verify-tag` subprocess does the same. None
      uses Git's global `--no-replace-objects` option, sets
      `GIT_NO_REPLACE_OBJECTS=1`, rejects `refs/replace/*`, or scrubs replacement and
      repository-control environment. Git replacement is therefore transparent even
      when the argument is a full object ID.

      In a real temporary sparse repository with the frozen Ed25519 signer, the
      reviewer created an invalidly signed tag object containing the correct canonical
      authorization record. The authorization ref remained exactly at invalid object
      `b4ad4c4168be726f902edf59e0fde7f2242a97af`. A Git replacement ref mapped that OID
      to the original valid signed object. Default `git cat-file tag <invalid-oid>`
      returned different bytes than the same command under
      `GIT_NO_REPLACE_OBJECTS=1`. The unmodified production guard printed
      `ACCEPTED b4ad4c4168be726f902edf59e0fde7f2242a97af`. Direct verification of that returned
      object with replacement disabled exited 1.

      A second real fixture exercised the actual assembler bootstrap and extracted
      guard. With the authorization ref fixed at invalid captured object
      `348a92b35155fa6e82b46c73be0b2369086e3b3f` and a replacement mapping to the valid
      signed object, `_verify_protocol_identity` printed
      `ASSEMBLER_ACCEPTED 348a92b35155fa6e82b46c73be0b2369086e3b3f True`.
      Thus workflow-shaped object pinning and the independent assembler boundary both
      authenticate substituted bytes while retaining the invalid OID as evidence.

      Adjacent controls behaved correctly without replacement: packed-ref exact happy
      path passed; abbreviated expected OID, nested annotated tag, and a loose ref
      shadowing a packed ref rejected. The five builder race cases and exact captured
      happy path passed in the 28-case suite. The defect is specifically Git's
      replacement-object layer and inherited Git control environment below the ref/OID
      equality logic.
    finding: |-
      A captured full Git OID does not bind the bytes parsed, peeled, or signature-
      verified because every Git operation honors replacement objects. The guard and
      assembler can return an invalid tag OID after authenticating a different object
      substituted through Git plumbing, without any authorization-ref movement for
      the final equality check to detect.
    failure_scenario: |-
      A caller-controlled assembler repository, Git environment, or contaminated
      hosted checkout installs a replacement mapping from an invalid canonical-record
      tag object to a valid signed tag object. Capture and both ref checks continue to
      observe the invalid OID. Git supplies the valid replacement bytes to parsing,
      peeling, and signature verification. The returned dispatch identity records the
      invalid object and otherwise valid fragments can proceed toward output.
      Replacement can analogously affect source-commit, contract, signer, manifest,
      and validator Git-object reads.
    consequence: |-
      The required equality among returned tag OID, parsed record bytes, target,
      signature, source Git objects, workflow identity, assembler identity, and
      evidence is false. Caller-controlled Git replacement state can bypass the
      campaign authorization root while all ref-stability checks remain green.
    required_action: |-
      Disable replacement objects for every security-critical Git invocation in the
      workflow, guard, and assembler, including signature verification and every
      source/contract/manifest/blob query. Use `git --no-replace-objects -C ...` and a
      deliberately scrubbed subprocess environment with
      `GIT_NO_REPLACE_OBJECTS=1`; remove or reject `GIT_REPLACE_REF_BASE`, repository-
      redirection, object-directory/alternates, namespace, and injected Git-config
      variables as appropriate to the declared trust boundary. Add real guard and
      extracted-assembler controls for a default `refs/replace` mapping, a custom
      replacement namespace selected through the environment, and source/tag object
      replacements. Require rejection before authorization JSON, platform execution,
      temporary assembly, raw result, or output commitment. Retain exact-object,
      packed/loose, nested-tag, abbreviated-OID, and late-ref controls.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    description: |-
      In real temporary signed repositories, hold the authorization ref fixed at an
      invalid canonical-record tag OID and map that OID to a separately valid signed
      tag using both the default replacement namespace and a custom environment-
      selected replacement namespace. Repeat with source-commit and tag-target
      replacement. Exercise the hosted guard command and extracted assembler guard
      with the production subprocess path. Every Git read and `verify-tag` must ignore
      replacements, the invalid original object must fail direct verification, and no
      authorization JSON, runner/mutation execution, temporary assembly, raw result,
      or commitment may survive. The ordinary exact signed-object and packed-ref
      controls must still pass.
    rationale: |-
      Full object IDs name stored objects, but Git replacement refs transparently
      change the bytes most plumbing and signature commands operate on. Ref/OID
      equality tests alone cannot detect this lower-layer substitution.
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: |-
      The separately SSH-signed canonical record continues to bind the exact source
      snapshot, authorization commit, campaign, preregistered packet, and manifest.
      Static later-lifecycle, branch/event/SHA/HEAD, tag-kind/suffix, campaign/packet/
      manifest, canonical-JSON, and signer-key substitutions reject under ordinary
      non-replacement Git semantics. The exact happy path passes.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new blocker is below that authorization design in Git object resolution, not a return of self-authorized snapshot selection."
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: |-
      The assembler still extracts the guard and declared validator closure from Git
      objects, checks both frozen authorities and worktree bytes, supplies the real
      packet, and runs isolated validation before commitment. Dependency substitution,
      missing sparse content, repo-root, protocol/schema, identity, attestation, and
      cross-run controls remain green under ordinary non-replacement Git semantics.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Replacement objects are a distinct Git plumbing input that can precede and falsify those otherwise correct blob checks."
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: |-
      Git-normalized source hashes match all seven execution-artifact declarations and
      receipt copies. Validator dependencies match primary and manifest hashes. The
      autocrlf true/false fixtures and exact signed-object path pass on Windows.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new blocker is platform independent and does not reopen LF normalization."
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    outcome: verified-resolved
    evidence: |-
      The remediation captures a full tag OID and uses that captured value for tag
      type, record bytes, parsed target, exact-object peel, `verify-tag`, returned
      identity, and final ref equality. Invalid-object/ref-swap, deletion after parse,
      movement after peel, signature-stage swap, and assembler post-capture swap all
      reject; the direct exact captured-object path passes. Packed and loose ref
      controls also reconcile correctly.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding does not move or re-resolve the ref; Git substitutes another object behind the same captured OID."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: |-
      The focused identity suite retains the signed happy path, self-consistent later
      commit, wrong event/ref/SHA/HEAD, annotated/lightweight/moved/deleted/wrong-
      suffix authorization, cross-run, and zero-output lifecycle cases. These static
      authority cases pass or reject as specified.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Replacement-object coverage is assigned a new requested-test ID."
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: |-
      Frozen validator bundle execution, real authorized packet validation, declared
      worktree-dependency substitutions, and output cleanup controls remain green in
      the focused and compatibility runs.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The new control must disable replacement before trusting any Git-extracted byte."
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: |-
      The exact Git blob, declared SHA-256, receipt, LF normalization, and Windows
      signed happy-path checks remain reconciled. Ruff, TeX, campaign, and focused
      suites are green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The complete suite collected 407 cases but was not rerun after the decisive blocker."
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    outcome: verified-satisfied
    evidence: |-
      The production decision logic rejects the invalid captured object after a
      same-name ref swap to a valid unrelated object, parse deletion, peel movement,
      signature-stage swap, and assembler post-capture movement. It accepts the
      unmoved exact signed captured object and returns the same OID.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The lower Git replacement layer is a genuinely distinct scope requiring the new test."
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment was run or inspected; this artifact assesses only pre-holdout protocol and governance machinery."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    CHANGES REQUESTED. Ordinary ref TOCTOU is fixed, but reproduced Git replacement
    makes the captured object ID name different authenticated bytes. Disable and test
    all replacement/environment paths before merge. This review does not authorize a
    signer-key amendment, tag, refreeze, activation, preregistration, custody
    transition, holdout, assembly, reveal, or scientific claim.
```

## Verification ledger

- Exact identity: handoff/tree `7e194f89f789a89ec459dfb7504dbdf6a3f901ed` / `6fa1682d166813e6aba2fe0f3fddabfcb8074be6`; content/tree `7a0cf295d874e6d181399d98ca77f4edd9d12a26` / `bccda159d89228589c3b36c97af12e545dcd8b34`; prior rereview/tree `e083dc79c204d40d218e10436e65ebfa06e79345` / `c66d8013c74643ec1ec227fca9fc24fab8d000da`.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py` — exit 0; 28 passed in 304.50 seconds.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r2_assembler.py tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_review_guidance.py` — exit 0; 22 passed in 663.57 seconds.
- Full collection — exit 0; 407 tests collected in 3.83 seconds. The complete execution was not run after the decisive critical blocker.
- Changed-Python Ruff — exit 0; all checks passed.
- `python scripts/check_tex.py` — exit 0; balanced and valid TeX source.
- `python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml` — exit 0; campaign contract valid.
- Both builder responses parsed against `review-response-v2.schema.json` — zero errors; all finding and requested-test responses reconcile one-for-one with immutable review history.
- Git replacement probe — production guard accepted invalid returned tag OID `b4ad4c4168be726f902edf59e0fde7f2242a97af`; direct no-replacement verification exited 1.
- Extracted assembler replacement probe — identity gate accepted invalid captured tag OID `348a92b35155fa6e82b46c73be0b2369086e3b3f`; no output was requested by the identity-only probe.
- Adjacent ref probes — packed-ref exact path passed; abbreviated OID, nested annotated tag, and loose-over-packed substitution rejected.
- Seven execution artifacts matched source-commit hashes and receipt copies; signer and all six validator-bundle entries matched their frozen hashes.
- R2 scoped diff was empty. Local and origin R3 tag counts were zero. Placeholder signer rejected with `exactly one frozen authorization signer is required`.
