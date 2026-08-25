# VIA-000 R3 recovery-protocol independent re-review 29

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-29"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-29"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-25"
commit_reviewed: "42defc265ed04cec104ed18542ab1eb966ffdd17"
baseline_commit: "2c605ff2894eed22c61f2292699a27fd8d8bff41"
prior_review_ref: "e1c7d0a5a68cb8f653b2d752d3d2503cd94f0462:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28.md"
builder_response_ref: "42defc265ed04cec104ed18542ab1eb966ffdd17:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28-RESPONSE-1.md"
context_hash: "506b09ed43d590c3030f6170459e68c709c2b9bd"
context_hash_method: 'git rev-parse "42defc265ed04cec104ed18542ab1eb966ffdd17^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - ".github/workflows/via000-r3-protocol.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_review_guidance.py"
  - "scripts/assemble_via000_r3_containment_proof.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "GitHub Actions run 32807276129, its eight exact jobs, logs, API metadata, caches, and artifact 9548587384"
  - "GitHub Actions run 32807276273 and job 97679724938"
access_level: "public repository, exact GitHub proof-only and generic-CI evidence, and retained public proof artifact; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-29 at exact campaign handoff
  42defc265ed04cec104ed18542ab1eb966ffdd17. Exact origin/ref/tree/sole-parent
  identity, the sealed RR28 import and response, bounded remediation diff, source
  and receipt equality, complete proof-run API/log/cache/artifact evidence, exact
  seven retained files and inner hashes, frozen redownload validation, generic CI,
  independent local tests and gates, all prior finding/test IDs, and drafted
  campaign state were inspected. No campaign, implementation, main-branch,
  lifecycle, or scientific state was changed. Operator and orchestrator are shared;
  session, worktree, and branch differ. Builder model is shared and external
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
  APPROVE with zero blocking findings. The exact campaign handoff, source/receipt
  bindings, all prior remediations, fresh six-cell hosted containment proof,
  normalized retained-artifact identity, frozen read-only redownload verifier,
  generic CI, and independent local gates agree. The R3 recovery/proof architecture
  is viable, and campaign/via000-r3-protocol-1 is ready to merge as a drafted,
  inactive protocol branch.

  Handoff 42defc265ed04cec104ed18542ab1eb966ffdd17 has exact tree
  506b09ed43d590c3030f6170459e68c709c2b9bd and sole parent content
  2c605ff2894eed22c61f2292699a27fd8d8bff41, exact tree
  b030fb61f138439fdf706eee77c89af985508274; origin's campaign ref matches.
  The handoff adds only the RR28 response and campaign/packet review metadata over
  content. Sealed RR28 was imported artifact-only at
  e1c7d0a5a68cb8f653b2d752d3d2503cd94f0462 from its unchanged review branch;
  its normalized SHA-256 is
  1aef8cb8e61be4b3794f3f17a8a641020d63ace5ba0e1b5565d036b98758e3bc.
  The response normalized SHA-256 is the required
  a59b6b03c6028186d3ab6d54379b5c21de4772756d8f59ecc98839a4e351ffb9.

  Proof run 32807276129 is success at the exact handoff. Windows mutation
  97679724943, Ubuntu mutation 97679725073, Ubuntu PDF 97679725120, Windows
  candidate 97679725147, Windows PDF 97679725179, Ubuntu candidate 97679725217,
  aggregate 97679879494, and verifier 97679972328 all completed success. The six
  producer envelopes have exact SHA-256 values 2d790cf34207974b9c3795194acacae0cc5a8e325bc79dbf9607c1e65b1890e6,
  d606194bac4e98002eff98b3194eb8a76f58885dae6b9361ee86b5ab3f5f9dba,
  2a2ad57a4e756c43ee9870f1b0730e02b74e2a832f7a78aee02a21282e0a2a58,
  56a66d3383b1b16ac8adcae88e856126657885c96805bf7d15ea048e77f12044,
  a7d27fbb153d179747570c6e5a75adf75ede4aa469f3df85898b9d889a5ad84b,
  and 7c2606bf4774558647bd002bb547e364787c0e06feb9f63cd2fbefad15993e3d
  for Ubuntu candidate/mutation/PDF and Windows candidate/mutation/PDF,
  respectively. All six exact cache keys restored with exact key equality.

  Exactly one artifact exists: ID 9548587384, name
  via000-r3-containment-proof-retained, size 48014, digest
  sha256:d1692063a4550d4897a62499a4c9cbb86af9586e32b2301e796a380660675f5f.
  Its archive contains exactly aggregate.json and the six expected envelope.json
  files. The downloaded bytes pass the frozen aggregator's retained-artifact mode;
  all declared inner proof/result/stdout/stderr hashes, exact 2x3 cells, repository,
  ref, source, workflow, run, attempt, platform/stage, containment, teardown,
  post-sign recheck, and non-scientific declarations match. The aggregate reports
  all_cells_passed true and the exact corrected normalizer emits the canonical
  artifact ID/digest/URL; the dependent verifier independently redownloads and
  accepts those bytes.

  Generic CI run 32807276273/job 97679724938 is success at the exact handoff:
  450 passed and two expected non-Windows platform skips, with lint, manuscript,
  examples, and validation contracts green. Independent Windows local validation
  is recorded below. The proof workflow/receipt normalized SHA-256 is
  e6747e1bd931ecf8bdaba17fe00fecbf0ccbafe721cf6b7e9e2b0d6dede311c2;
  protocol, packet, and protocol-manifest normalized SHA-256 values are
  6385d5c53fddaf6bf142ad30fc89390b5ea9685788bee55290aa1769ef5d4d6a,
  6ec1e2fe4981f86d5846a6ff0b9e39fc5ed7cef8476a412c5b98d2db84f35f02,
  and 051eb547cf5ecc873a8a53010ab2aab8faf51600d5c709966d5c2a94299d24b3.

  This approval is deliberately narrow. R3 remains drafted and pending,
  holdout_started false, and unrevealed. No tag exists at the reviewed handoff and
  no signer/key, custody, refreeze, activation, holdout, scientific execution,
  commitment, adjudication, or reveal was created, accessed, performed, or
  authorized. Merge readiness applies only to the campaign protocol branch; every
  lifecycle transition remains a distinct governed and separately reviewed gate.
findings: []
requested_tests: []
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved by source, identity suite, and exact hosted bindings.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Frozen verifier closure and retained inner subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Windows 3/3 producer and transport paths pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Captured-object and final-ref controls remain tested and unchanged.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Replacement-disabled sanitized Git closure remains tested and unchanged.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Six exact hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "All six tool identities and corrected aggregate identities pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six isolated contexts and retained evidence pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects and attestations pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-resolved, evidence: "Exact fresh 2x3 proof, retained aggregate, and verifier are green.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 exact envelopes survive protected export and transport.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "All six canonical envelopes survive exact transport and aggregation.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six exact cache saves/restores and distinct keys pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Supported Windows control plane and cross-job transport pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported built-in/literal shells execute all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity correction is the preserved resolution.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment and token evidence pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts and workflow blocks parse; all hosted paths execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 canonical results aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows medium export and low-IL denial evidence pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Exact native descriptor observations are retained in all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native root/file owner, DACL, and label predicates pass Windows 3/3.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows zstd pre/post identity and cache transport pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All Windows inner JSON subjects pass canonical-byte aggregation.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: verified-resolved, evidence: "Bare action output is strictly normalized once; canonical digest reaches green verifier.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: verified-resolved, evidence: "Exact literal launcher, module, PSHOME, version, and argv pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Observation contract led to exact hosted predicates now passing.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", outcome: verified-resolved, evidence: "Exact entry and reset PATH predicates pass the aggregate normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Exact argv and runner-script identity pass in production context.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", outcome: verified-resolved, evidence: "Extensionless UUID and complete metadata pass in production context.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Exact setup-python context is bound and passes production normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001", outcome: verified-resolved, evidence: "Exact five-component ordinal/raw binding and sanitized rechecks pass run 32807276129.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Exact hosted source/ref/run bindings and identity tests pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Frozen retained-artifact verifier accepts exact closure.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Exact Ubuntu/Windows canonical subjects pass aggregation.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved mutation corpus passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved replacement corpus passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Six hosted workflow boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Producer and aggregate tool identities pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six contexts and retained closure pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "All six post-quiescence attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-satisfied, evidence: "Exact fresh six-cell proof, retained artifact, and verifier are green.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 exact envelopes retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes and inner subjects verify.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact cache keys save/restore without fallback.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Supported control-plane execution and cross-job transport pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Built-in/literal Windows PowerShell execution passes 3/3.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "No-LUA low-integrity factor was isolated and production result passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows low-IL children and token evidence pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight R3 scripts and workflow blocks parse and execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu canonical empty-array results retain 3/3.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows protected medium export and low-IL denials pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Exact descriptor evidence is retained for all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native root/file descriptor acceptance passes 3/3.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd identity and cache transport pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "All retained inner JSON bytes pass canonical validation.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: verified-satisfied, evidence: "Strict bare-to-prefixed digest normalization and verifier pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: verified-satisfied, evidence: "Literal pwsh launcher, module, PSHOME, version, and argv pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "Its incomplete diagnostic was superseded by RR27's exact-context test.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", outcome: verified-satisfied, evidence: "Exact initial and reset PATH predicates pass the live normalizer.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Exact argv5 and runner-script metadata pass in setup-python context.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", outcome: verified-satisfied, evidence: "Extensionless UUID and exact metadata pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Exact context was observed and bound in production.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001", outcome: verified-satisfied, evidence: "Positive/negative corpus passes and exact hosted normalizer/verifier are green.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No new scientific or protocol experiment is proposed by this final artifact-only approval. Lifecycle actions remain separately gated."
recommendation:
  approve: true
  blocking_findings: 0
  rationale: "APPROVE. Architecture is viable and the exact campaign branch is merge-ready. This authorizes only merge of the reviewed drafted protocol branch and preparation for separately reviewed signer/custodian/lifecycle gates; it does not authorize a key, tag, refreeze, activation, holdout, scientific execution, commitment, adjudication, or reveal."
```

## Verification ledger

- Exact handoff/tree/sole-parent/content-tree/origin identities matched. The handoff scope is response plus review metadata; the imported RR28 commit is artifact-only and its bytes and response bytes match the required normalized hashes.
- The bounded remediation binds the exact setup-python 3.11.15 outputs and roots, exact ordinal/raw five-component initial PATH, immediate `/usr/bin:/bin` reset, and repeated reassertions. The source and receipt are byte-identical at normalized SHA-256 `e6747e1bd931ecf8bdaba17fe00fecbf0ccbafe721cf6b7e9e2b0d6dede311c2`; the negative mutation corpus remains complete.
- Proof API metadata and full logs for all eight exact jobs were inspected. All six producer/containment/staging/digest/cache paths, aggregate restores and seven-file consolidation, corrected normalizer, and dependent frozen verifier are green at the exact handoff.
- Artifact 9548587384 was downloaded independently. It is the only run artifact, has the exact name/size/digest, contains exactly seven regular files, and passes the exact reviewed frozen verifier. Each envelope is canonical and its declared inner members/hashes and 2x3 identity/containment facts match.
- Generic CI is green at the exact handoff with 450 passed/two expected platform skips and all lint/manuscript/examples/contracts steps green. Independent Windows gates: 452 passed in 2662.02 seconds; focused RR28 + PowerShell parse closure + review guidance: 11 passed in 22.17 seconds; R3 identity: 73 passed in 772.58 seconds; Ruff: all checks passed; actionlint 1.7.12: clean; campaign validator: valid; two-pass TeX: success.
- Review YAML/schema, exact 32/32 prior finding and 32/32 prior-test ID reconciliation, duplicate-ID checks, source/receipt equality, hashes, response/history/diff/ref/tree checks, normal and ignored cleanliness, and no-tag/drafted/pending/unrevealed/holdout-false state were checked.
- No custody directory, sealed/hidden/reveal material, external invalid package, or forbidden handoff memo was accessed. No main-branch, signer/key/tag/refreeze/custodian/lifecycle/holdout/scientific/commitment/adjudication/reveal action was performed.
