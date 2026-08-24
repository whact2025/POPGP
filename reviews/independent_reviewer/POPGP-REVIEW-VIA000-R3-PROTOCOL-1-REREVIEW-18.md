# VIA-000 R3 recovery-protocol independent re-review 18

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-18"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b"
baseline_commit: "343794c00ca8808d56207576bef2d720a6f68199"
prior_review_ref: "343794c00ca8808d56207576bef2d720a6f68199:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-17.md"
builder_response_ref: "6ca4451bfcb79f16a838f0cbef2977d0f783eb83:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16-RESPONSE-1.md"
context_hash: "47eaec71c4c7ef1335f9fbe097bb4946fb602e37"
context_hash_method: 'git rev-parse "e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-windows-export-acl-diagnostic.yml"
  - "experiments/via000-r3-windows-export-acl-1/RR17-WINDOWS-EXPORT-DIAGNOSTIC.ps1"
  - "experiments/via000-r3-windows-export-acl-1/RR17-WINDOWS-EXPORT-ATTACK.ps1"
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-17.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub runs 32753682937 and 32751391135 job/log/cache/artifact metadata; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-18 at exact diagnostic commit
  e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b. Exact origin, parent/tree,
  experiment-only diff, complete diagnostic source/logs, the RR16 production failure,
  raw and managed descriptor facts, hostile operations, cross-step requery, prior
  review bytes, focused regressions, parser closure, and campaign state were inspected.
  No campaign implementation or external lifecycle state was changed. Operator and
  orchestrator are shared; session/worktree/branch differ. Builder model is shared and
  external scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity production descriptor-application blocker.
  RR17's diagnostic/observability finding is resolved. The experiment supports a
  direct bounded production correction; another experiment on the workspace-relative
  path is not required before that correction. Architecture remains viable, but the
  exact retained 2x3 proof is still absent and no lifecycle action is authorized.

  Experiment commit e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b has exact tree
  47eaec71c4c7ef1335f9fbe097bb4946fb602e37 and sole parent RR16 handoff
  6ca4451bfcb79f16a838f0cbef2977d0f783eb83. Its only changes are the non-
  authoritative diagnostic workflow and two experiment scripts. Run 32753682937 is
  exact push attempt 1 at that commit; sole job 97516221546 passed. The workflow had
  contents:read, no secrets, a sparse four-file closure, built-in PowerShell 7, fixed
  sanitized PATH, and no artifact, cache, repository write, or campaign path.

  The diagnostic first ran the unmodified production containment primitive and proved
  low-integrity SID S-1-16-4096, successful child exit, descendants quiescent, and zero
  active processes. Only after teardown it created a fresh root under RUNNER_TEMP and
  applied the exact RR16 production setter. Unlike production run 32751391135's three
  workspace-relative roots, this root was accepted. Managed and native queries agreed:
  current, managed-owner, and native-owner SID were the exact hosted runner SID ending
  in RID 500; DACL protected=true; managed/raw ACE count=1; the sole explicit allow ACE
  granted that SID FullControl mask 2032127 with CI/OI flags 3; root label was exactly
  S-1-16-8192, NO_WRITE_UP mask 1, inheritance flags 3, count 1.

  The newly created envelope exposed a separate deterministic defect that production
  would hit after its root is fixed. Its managed and native owner was BUILTIN
  Administrators S-1-5-32-544, not the current runner SID. Its access boundary was
  otherwise exact: one inherited runner-SID FullControl ACE, mask 2032127, inherited
  flag 16, unprotected file DACL, and one inherited medium S-1-16-8192 NO_WRITE_UP ACE
  with mask 1 and flag 16. The differing owner is consistent with an elevated token's
  default owner; the experiment establishes the fact, not its platform internals.

  A second unmodified production low-IL containment launched the hostile script. Its
  mutable write succeeded, proving execution, while create, overwrite, hardlink,
  reparse, rename, delete, and replace markers were all absent. The export retained
  exactly one single-link/default-stream envelope with unchanged SHA-256
  8de14cec87ba44fc3ed7ef56d17db1869cc8c184011a03a237a031ce4ea0e422;
  teardown again reported zero active processes. Root/file raw and managed descriptors
  remained byte-for-byte-equivalent as normalized JSON. A separate trusted workflow
  step re-queried the same path, identity, hash, descriptors, and labels and reported
  descriptor_unchanged=true, low_integrity_attacks_denied=true, and
  teardown_zero_active=true.

  The diagnostic used RUNNER_TEMP while production failed under GITHUB_WORKSPACE, so
  it does not identify which RR16 managed root subpredicate differed in production.
  That uncertainty does not justify relaxing anything, but it also does not require a
  third diagnostic. A native root owner/protected-DACL setter followed by native and
  managed exact queries removes dependency on parent/path descriptor behavior. An
  explicit file-owner application after creation closes the newly observed owner
  difference while preserving the inherited DACL. The mandatory fresh six-cell proof
  itself is the exact workspace-relative acceptance test and remains fail-closed.
findings:
  - id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:478-571; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:397-407 and 560-590; corresponding receipt/workflow rechecks"
    evidence: |-
      Run 32751391135 rejected all three workspace-relative roots at the combined owner,
      protected-DACL, or cardinality assertion. Run 32753682937 applied the same helper
      successfully to a RUNNER_TEMP root and logged exact managed/native descriptors.
      That proves the target descriptor and low-IL no-write-up boundary are viable but
      leaves the production path's failed root field unidentified. The same experiment
      proves a newly created inherited envelope has native/managed owner
      S-1-5-32-544 rather than the exact current runner SID, so the unchanged file
      assertion would reject it. Low-IL create/write/rename/delete/replace and link
      attacks were denied; trusted access, hashes, descriptors, labels, teardown, and
      a separate-step requery passed. Therefore neither a weaker predicate nor a path
      move is supported; deterministic native application and exact requery are.
    finding: "Production does not deterministically apply and re-query the exact root and file owner/DACL descriptor required on hosted Windows."
    failure_scenario: |-
      The managed root setter behaves differently under the workspace hierarchy, or
      the elevated creator assigns the envelope to BUILTIN Administrators. Production
      either fails 0/3 before transport or a guessed relaxation accepts a non-exact
      owner or inherited/broad writable DACL.
    consequence: "Windows evidence cannot reach digest/cache transport, the exact retained 2x3 proof cannot exist, and execution-boundary approval remains blocked."
    required_action: |-
      Replace only the Windows root owner/DACL application boundary with a reviewed
      native implementation. Capture the current runner SID once; construct one DACL
      containing exactly one ACCESS_ALLOWED ACE for that SID, access mask 2032127
      (FullControl), root ACE flags 3 (ContainerInherit|ObjectInherit), and no other
      ACE. Apply owner plus DACL with OWNER_SECURITY_INFORMATION,
      DACL_SECURITY_INFORMATION, and PROTECTED_DACL_SECURITY_INFORMATION to the fresh
      post-teardown root. Continue applying the exact medium S-1-16-8192 mandatory
      label with NO_WRITE_UP mask 1 and root flags 3. Do not inherit a parent DACL, add
      SYSTEM/Administrators/Everyone/Users write grants, or move output to a mutable or
      low-integrity root.

      Immediately after the canonical envelope's exclusive CreateNew/write/flush/close
      completes, explicitly set only its owner to the captured current runner SID with
      the native owner API. Do not replace or protect its inherited DACL. Then require
      managed and GetNamedSecurityInfo/raw agreement. Root invariants are: owner exact
      runner SID; DACL present, protected, non-defaulted, and non-null; raw/managed ACE
      count one; sole explicit allow ACE exact SID/mask/type/CI-OI/no-propagation and
      non-inherited; exact medium label SID/mask/flags/count. File invariants are:
      owner exact runner SID; DACL present and unprotected; raw/managed ACE count one;
      sole runner FullControl ACE inherited with flags 16, no extra explicit ACE; and
      exact inherited medium label SID/mask/flags 16/count. Record the complete control
      mask; if Windows retains DACL_AUTO_INHERITED, bind that observed normalized mask
      explicitly rather than wildcarding control flags. Reject DACL_DEFAULTED,
      DACL_UNTRUSTED, null DACLs, different owner, extra ACE, SID/type/mask/flag drift,
      or label drift.

      Run these queries in the proof runner after root setting, after file owner setting,
      after hash capture, in the separate digest step, and immediately before cache
      save using the manifest-hash-bound helper/receipt. Every failure must clean the
      entire cell cache and leave no digest, cache, aggregate, or commitment.

      Add Windows regressions for both RUNNER_TEMP and the exact nested workspace-
      relative hierarchy with varied parent owner/inheritance; elevated/default-owner
      file creation followed by exact owner correction; mutation of every owner,
      protection, null/defaulted DACL, ACE count/SID/type/mask/inheritance/propagation,
      label SID/mask/flags, link/ADS/hash field; trusted read/write; the production
      low-IL create/write/link/rename/delete/replace hostile matrix; descriptor/hash
      stability and separate-process requery. Update receipt copies and hashes. Then
      run the local live hostile regression and a fresh exact six-cell hosted proof,
      requiring six distinct digests/caches, exact aggregate restore/validation, one
      retained seven-file artifact, and a green redownload verifier. A separate
      workspace-path diagnostic before this patch is unnecessary; the full proof is
      the authoritative exact-path gate.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001"
    description: "Apply and independently re-query an exact native owner/protected one-runner-ACE root descriptor plus explicit current-runner file owner while preserving the one inherited file ACE and medium NO_WRITE_UP label; mutate every descriptor field, replay low-IL attacks, and pass the exact retained hosted 2x3 proof and verifier."
    rationale: "The diagnostic proves the boundary works, identifies a real elevated file-owner mismatch, and shows that deterministic application plus exact production-path acceptance is narrower and stronger than another diagnostic or any predicate relaxation."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Supported hosted shell and fixed command boundary execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained exact 2x3 proof.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Windows containment works, but production export remains blocked.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Diagnostic teardown passes; no retained production subject exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Production proof remains 3/6.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Diagnostic export works; production export does not.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No Windows production envelope reaches transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Only Ubuntu production caches exist.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control plane executes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "No custom-shell no-op recurred.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "Low-IL children execute and tear down.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: unresolved, evidence: "No-LUA low-IL boundary passes; retained production acceptance is absent.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Experiment and production scripts parse and execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu production proof passes 3/3.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: unresolved, evidence: "Exact target works in diagnostic root, but production path and file owner remain uncorrected.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Run 32753682937 logs every requested root/file managed/native descriptor and label field, low-IL attack result, teardown, and cross-step requery.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "The experiment used RUNNER_TEMP; direct deterministic production correction remains required."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Supported hosted command boundary executes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained exact 2x3 proof.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Production Windows export is blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Diagnostic passes; retained production subject absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Production proof remains 3/6.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Exact production export not yet rerun.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No Windows production transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No Windows production cache.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in pwsh works; retained production acceptance remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "Low-IL production helper passes; retained exact proof absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Production and experiment scripts parse and execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu production proof passes 3/3.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: unresolved, evidence: "Diagnostic boundary passes; exact production root/file acceptance remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Run 32753682937 records exact requested facts, denies all seven attacks, proves two teardowns, and passes separate-step requery.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR"
  predicted_outcome: "The native root setter plus explicit file-owner application will produce exact descriptors on the workspace-relative path and allow all six proof cells, aggregate, artifact, and verifier to complete."
  predicted_failure_mode: "If path-specific hosted behavior remains, an exact native post-set query will reject before envelope/digest/cache rather than admit a weaker descriptor."
  confidence_statement: "High confidence that the file-owner correction is required; moderate confidence that the native root setter closes the unresolved workspace-path difference. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Implement the exact native root owner/protected-DACL setter and explicit file-owner application, retain every exact query and cleanup gate, add hostile/mutation/path regressions, then rerun the full retained 2x3 proof. No additional path-only diagnostic is required. Architecture remains viable; merge, signer/custodian work, refreeze, activation, and holdout are not authorized."
```

## Verification ledger

- Exact experiment commit `e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b`, tree `47eaec71c4c7ef1335f9fbe097bb4946fb602e37`, sole parent `6ca4451bfcb79f16a838f0cbef2977d0f783eb83`, origin branch, and three-file experiment-only diff matched. RR17 review Git-normalized SHA-256 is `343975a9fbe6c213f625346d188aebefa1bd50b4823dffe8304bde7d9410727e`.
- Diagnostic workflow/script/attack/helper Git-normalized SHA-256 values are `c1dcc87e6c5ed8c297f471dd5c5a9b12678be5ae82b816354ee784e5bb280267`, `d7006f3bddef19392c7c5d63d25de1eaade84d07721e31d5ff8e1cdbf726d3fd`, `73d1a902bb2f3cc64af63147acf7acc324008ae44efa74e595b18369c614d82d`, and `19f3da83be0498ccaea16571aaa7b8b95b4d636006cf7131aa2495fd841a0104`.
- Run `32753682937`, job `97516221546`, every source/log step, full logged descriptor JSON, seven hostile operations, both teardown results, and separate-step recheck were inspected. Source/run/attempt and all logged hashes matched. The run retained zero artifacts and performed no campaign/lifecycle action.
- The two experiment scripts and production helper parsed with zero PowerShell errors. Focused RR16 boundary tests passed 2/2. Campaign validator reported valid. The experiment commit changes no protocol, campaign, packet, test, receipt, lifecycle, or scientific file.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No R3 production tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
