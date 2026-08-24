# VIA-000 R3 recovery-protocol independent re-review 17

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-17"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-17"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "6ca4451bfcb79f16a838f0cbef2977d0f783eb83"
baseline_commit: "369c87731cfe01e8c4bec4ef24c524a36f8a75a0"
prior_review_ref: "369c87731cfe01e8c4bec4ef24c524a36f8a75a0:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16.md"
builder_response_ref: "6ca4451bfcb79f16a838f0cbef2977d0f783eb83:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16-RESPONSE-1.md"
context_hash: "c870ee28f97f75237403a79ca55edae669dcdacf"
context_hash_method: 'git rev-parse "6ca4451bfcb79f16a838f0cbef2977d0f783eb83^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32751391135 job, cache, and artifact metadata/logs; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-17 at the exact handoff. Origin,
  sole-parent ancestry, trees, prior review/response bytes, all eight hosted jobs,
  source and receipt copies, local Windows descriptor behavior, focused regressions,
  parser closure, and campaign validation were independently inspected. No campaign
  implementation or external lifecycle state was changed. Operator/orchestrator are
  shared; session/worktree/branch differ. Builder model is shared and external
  scientific validation is not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex (GPT-5)"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration: {final_labels_seen: false, secret_seed_seen: false, private_evaluator_seen: false}
summary: |-
  CHANGES REQUESTED with one high-severity, bounded Windows observability and hosted
  acceptance blocker. The architecture remains viable, but the exact retained 2x3
  proof is still absent and no lifecycle step is authorized.

  Run 32751391135 is exact push-event attempt 1 at handoff
  6ca4451bfcb79f16a838f0cbef2977d0f783eb83. Ubuntu pdf 97508923662,
  mutation 97508923935, and candidate 97508924093 passed every producer step,
  emitted distinct digests, and saved exact caches 6943044021, 6943045240, and
  6943042673. This confirms the RR16 Ubuntu empty-array fix on the hosted platform.

  Windows candidate 97508923876, mutation 97508923906, and pdf 97508923983 all
  executed through supported built-in PowerShell 7, completed the production
  restricted-token containment and post-teardown subject checks, then failed in the
  first root assertion called by Set-Via000WindowsExportSecurity at helper line 547:
  `Windows export owner or protected DACL cardinality differs`. They failed before
  envelope creation, digest output, preflight, or cache save. Aggregate 97509106313
  observed exact results success,success,success,failure,failure,failure and empty
  Windows digests, then rejected at its first gate. Verifier 97509148309 was skipped;
  the run retained zero artifacts and no Windows cache.

  The combined exception is not a diagnosis. At this exact call, the helper has already
  accepted a non-null current runner SID, created a fresh DirectorySecurity, requested
  owner=current SID, protected the DACL while dropping inheritance, added one inheritable
  FullControl allow rule for that SID, successfully called Set-Acl, and applied the
  medium label. The throw combines three still-unobserved facts: owner SID equality,
  AreAccessRulesProtected=true, and rules.Count=1. It occurs on the root before any
  file exists; file inheritance, file label, ADS, and hash predicates cannot be the
  cause. The later native mandatory-label predicate also was not reached.

  The same exact helper succeeds locally: current SID equals owner SID, the DACL is
  protected, the raw DACL and managed rule views each contain one explicit FullControl
  allow ACE with ContainerInherit/ObjectInherit for that SID, raw control flags include
  DiscretionaryAclPresent, DiscretionaryAclAutoInherited,
  DiscretionaryAclProtected, and SelfRelative, and the mandatory label is one
  S-1-16-8192 NO_WRITE_UP ACE. The live low-integrity RR16 regression also passes.
  Thus Set-Acl adding an ACE, canonicalizing to a different cardinality, losing the
  protected bit, or assigning a different owner on the hosted image are plausible
  alternatives, not established facts. ACE canonical ordering alone cannot explain a
  cardinality change, and the logs do not support attributing the failure to elevated
  ownership or to a mandatory-label defect.

  The smallest evidence-producing next step is one proof-only Windows diagnostic cell,
  not a blind relaxation or another six-cell run. After complete containment teardown,
  it must record the exact non-secret normalized root and inherited-file descriptors,
  prove trusted access and production low-integrity write denial, recheck them in a
  separate trusted step, and fail closed without transport or lifecycle effects. The
  observed field then determines whether the production setter should use a native
  owner/protected-DACL application instead of Set-Acl. Exact owner, protected DACL,
  one reviewed runner-SID allow ACE, and medium NO_WRITE_UP remain required.
findings:
  - id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:478-563; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:397-407; .github/workflows/via000-r3-containment-proof.yml Windows producer steps"
    evidence: |-
      Three independent hosted Windows 2025 runners reached the same first root
      assertion and emitted only the combined line-547 exception. Sequential control
      flow proves the runner SID was available, containment/teardown and protected
      subject checks had completed, Set-Acl returned successfully, and the medium-label
      setter returned. It does not reveal owner SID, DACL protection/control bits, ACE
      count/content, or label facts. The file did not yet exist. No later digest/cache
      step ran. Local execution of the exact helper produces the intended owner,
      protected one-ACE DACL, and medium NO_WRITE_UP label, so there is no evidence for
      safely weakening any predicate or for choosing one hosted-specific correction.
    finding: "The hosted Windows export descriptor is rejected, but a combined assertion hides the field needed to distinguish a platform representation difference from a security-boundary defect."
    failure_scenario: |-
      A guessed fix relaxes owner, protected-DACL, or ACE-cardinality requirements to
      make the hosted job green. A broad or low-integrity-writable ACE, unprotected
      inheritance, or wrong owner could then be admitted without evidence that the
      low-IL child cannot mutate the staged envelope.
    consequence: "Windows remains 0/3, the retained six-cell proof cannot be assembled, and the RR4/RR5/RR6 execution-boundary claims remain unauthorized."
    required_action: |-
      Run one bounded, non-scientific, Windows-only diagnostic on an experiment branch
      or separately gated proof-only path. Use the exact built-in pwsh identity and
      sanitized PATH. After the exact production containment tree is torn down, create
      the fresh root and one known bounded file, apply the current setter, and emit only
      these non-secret descriptor facts for both root and file: current user SID; owner
      SID; full security-descriptor control flags and explicit DACL-protected boolean;
      raw and managed ACE counts; every ACE in observed order with SID, allow/deny type,
      numeric access mask/FileSystemRights, inheritance flags, propagation flags, and
      IsInherited; and mandatory-label ACE count, SID, numeric policy mask, and ACE
      flags. Do not log file contents or tokens.

      In the same job, prove the medium trusted control plane can create/read/hash the
      file, then use the unmodified production restricted low-IL primitive to attempt
      create, overwrite, rename, delete, replace, hardlink, and reparse operations on
      root/file. Require every operation denied, exact hashes unchanged, complete
      descendant teardown, and no marker residue. A following separate trusted pwsh
      step must re-query the same root/file fields and hash and require exact equality.
      The diagnostic is non-authoritative and must leave no cache/artifact.

      If the evidence shows Set-Acl changes owner, protection, or ACE shape, replace
      only the descriptor-application boundary with a reviewed native setter that
      applies OWNER_SECURITY_INFORMATION plus a protected DACL containing exactly the
      current runner-SID allow ACE; keep the existing native exact medium NO_WRITE_UP
      label query. Re-query the raw/native descriptor after setting and at both later
      workflow boundaries. Do not allow extra/broad writable ACEs, weaken exact owner
      or protection, move output to mutable/low integrity, or accept an SDDL/managed
      summary without raw ACE evidence. Then rerun the local hostile regression and
      the complete fresh hosted 2x3 proof with retained aggregate and verifier.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001"
    description: "On one hosted Windows proof-only cell, capture exact post-teardown root/file owner, raw control flags, protection, every ACE, and mandatory-label fields; prove trusted access, production low-IL mutation denial, complete teardown, hash stability, and a cross-step exact descriptor recheck before selecting the setter correction."
    rationale: "The current three-way exception cannot support a safe code change; this is the minimum test that isolates the failed invariant without weakening or running the campaign."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Immutable snapshot/authorization controls remain unchanged.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Validator-source closure remains unchanged.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Git-normalized cross-platform source controls remain unchanged.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Captured authorization-object boundary remains unchanged.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Replacement-disabled Git-object closure remains unchanged.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Hosted proof uses supported built-in Windows pwsh and fixed Ubuntu shell.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Ubuntu passes, but no retained exact 2x3 proof exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Windows containment executes, but its evidence cannot be exported or retained.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Windows reaches post-teardown export, but no retained subject exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Only three Ubuntu cells are transport-complete.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows rejects the root descriptor before envelope creation.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No Windows envelope or digest reaches transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Three Ubuntu caches exist; all Windows cache steps were skipped.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control plane and child invocation execute on hosted Windows.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "No custom-shell no-op recurred.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "All Windows low-IL children now execute through containment teardown.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: unresolved, evidence: "No-LUA low-IL production containment succeeds in three hosted cells, but retained acceptance is blocked after teardown.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "All scripts parse and every hosted producer progressed beyond parsing.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu hosted candidate/pdf/mutation passed and saved distinct exact caches.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: unresolved, evidence: "Explicit descriptor construction works locally but fails all hosted Windows roots at the combined assertion.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", notes: "Do not weaken the RR16 target."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Supported hosted shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained exact 2x3 proof.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Windows evidence is not retained.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No retained Windows post-quiescence subject.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted proof is 3/6.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No Windows envelope.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No Windows transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Ubuntu cache succeeds; Windows does not reach cache save.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in pwsh works; retained acceptance remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved diagnostic result.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "Three hosted Windows containment runs pass through teardown, but the required retained proof does not exist.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight files parse; focused regression and hosted parsing pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu passes 3/3 and saves exact caches.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: unresolved, evidence: "Local hostile boundary passes, but hosted root descriptor rejects before envelope creation.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC"
  predicted_outcome: "The diagnostic will identify one concrete hosted root mismatch in owner SID, protected-DACL control state, or ACE cardinality while retaining medium NO_WRITE_UP and low-IL write denial."
  predicted_failure_mode: "Most plausibly the hosted Set-Acl persistence/canonicalization or elevated ownership context differs from the local managed descriptor; current evidence cannot rank owner, protection, or ACE shape reliably."
  confidence_statement: "Moderate confidence that the boundary is recoverable; low confidence in any specific hosted descriptor field until it is logged. This is a non-scientific infrastructure prediction."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Run the one-cell post-teardown Windows descriptor diagnostic before changing the setter, then preserve exact owner/protected one-runner-ACE/medium-NO_WRITE_UP semantics and rerun the full retained 2x3 proof. Architecture remains viable; merge, signer/custodian work, refreeze, activation, and holdout are not authorized."
```

## Verification ledger

- Exact handoff `6ca4451bfcb79f16a838f0cbef2977d0f783eb83`, tree `c870ee28f97f75237403a79ca55edae669dcdacf`, sole parent/content `d1dde2bda60a929b039d9dd4ec2c2153d2fee89f`, tree `6abc384b21844aad08817cc8965aa28c6bba7f23`, and origin campaign head matched. The RR16 response Git-normalized SHA-256 is `665f703adef10bd233532ae961c852c208e299ca010c3e9d4f0fe5df38ce998b`; prior review Git-normalized SHA-256 is `f42ecf7475c8bdb144726e1f6a51b69cb51306aaa6352755a4f9b701eb46a2f3`.
- All eight jobs and all producer/aggregate logs for run `32751391135` were inspected. Exact Ubuntu digests were candidate `8582c425a58af155018d985bfd944803d8eb256bf2546d1249c9ed58ef5b756e`, pdf `ce4bd42c8f5456f9da2880c189d36edc8ad2a3412b8fc5e6383901d700567704`, and mutation `0b4804191819bf252620818a2a3ce07ec0a051b9056c1bca6dffdcdb8496100f`. The aggregate failed before checkout/restore; verifier skipped; artifact count was zero.
- Exact workflow/helper/proof-runner SHA-256 values are `3f4a0a45e8a4e63f3cd7b77780b221a56aed51d955710818dc6d82dc35ab4146`, `19f3da83be0498ccaea16571aaa7b8b95b4d636006cf7131aa2495fd841a0104`, and `ec827f0197e6eba1e40bc2bcf6f80f80833a2ef419c6829b70545dc5a338cd52`; helper and proof-runner receipt copies are byte-identical.
- Local exact helper descriptor probe passed and recorded current=owner SID, protected=true, managed/raw DACL count 1, exact runner FullControl allow ACE with CI/OI, and one medium NO_WRITE_UP label. Focused RR16 tests passed 2/2. RR15 parse regression passed 1/1; all eight R3 protocol/receipt PowerShell files parsed with zero errors. Campaign validator reported valid.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No R3 production tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
