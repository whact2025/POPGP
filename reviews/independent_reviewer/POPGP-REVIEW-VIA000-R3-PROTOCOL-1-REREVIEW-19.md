# VIA-000 R3 recovery-protocol independent re-review 19

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-19"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "c39626d098f6874216fbd0e3a8d78224569d4e82"
baseline_commit: "962370ce46d0acf48382ca994038aa6d000317a1"
prior_review_ref: "962370ce46d0acf48382ca994038aa6d000317a1:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18.md"
builder_response_ref: "c39626d098f6874216fbd0e3a8d78224569d4e82:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18-RESPONSE-1.md"
context_hash: "9a1263598a5f26ad04434ab3bc930ee8467ffccd"
context_hash_method: 'git rev-parse "c39626d098f6874216fbd0e3a8d78224569d4e82^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - ".github/workflows/via000-r3-protocol.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "actions/runner-images@0900c002193dc2d3fade0cd9133ae70e7088eb05:images/windows/scripts/build/Install-Zstd.ps1"
  - "actions/runner-images@0900c002193dc2d3fade0cd9133ae70e7088eb05:images/windows/Windows2025-VS2026-Readme.md"
  - "actions/runner-images@0900c002193dc2d3fade0cd9133ae70e7088eb05:images/windows/scripts/tests/Tools.Tests.ps1"
  - "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9:README.md"
access_level: "public repository, authoritative public runner-image/action sources, and GitHub run 32763190366 job/log/cache/artifact metadata; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-19 at exact handoff
  c39626d098f6874216fbd0e3a8d78224569d4e82. Exact origin, remote head,
  sole-parent content ancestry, trees, review/response history, full proof workflow
  and receipt, all eight hosted job timelines/logs, cache and artifact metadata,
  authoritative image-build sources, focused hostile/cache regressions, parser
  closure, guidance, and campaign state were inspected. No implementation, campaign,
  cache, lifecycle, or scientific state was changed. Operator and orchestrator are
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
  CHANGES REQUESTED with one high-severity proof-transport tool-identity blocker.
  The RR18 native root/file descriptor finding is verified resolved: every Ubuntu and
  Windows candidate/pdf/mutation job passed the production containment, teardown,
  protected-root, canonical-envelope, staged-export, cross-step descriptor/hash, and
  digest-output steps. The failure is later, fail-closed, and bounded to the proof-only
  Windows cache precondition. Architecture remains viable, but the required retained
  exact 2x3 artifact and verifier do not exist, so no merge or lifecycle action is
  authorized.

  Handoff c39626d098f6874216fbd0e3a8d78224569d4e82 has exact tree
  9a1263598a5f26ad04434ab3bc930ee8467ffccd and sole parent/content
  3daeab1305a90430824257b78e6053d383c2687e with tree
  c74ae5f6abc9c3e6561a2159dd8e1493204e0804. Origin's campaign branch matched.
  The handoff adds only the RR18 response and campaign/packet review metadata over the
  reviewed content. The imported RR18 artifact retained normalized SHA-256
  6059bcc082ce03771115eeaaba58eaaf76b66d05630bac48718adaab3970a13b;
  the RR18 response retained normalized SHA-256
  d08b03770c9e3828800f570ba79c003a4cb38d311fd2b657e3235a4e68fae9a3.

  Hosted push run 32763190366 attempt 1 was exact at the handoff. Ubuntu jobs
  97546576015/97546575802/97546575949 (candidate/pdf/mutation) passed and saved
  caches 6946922883/6946924719/6946926355 under six-field identity and distinct
  envelope digests. Windows jobs 97546575887/97546575857/97546575869 likewise
  reached distinct digest outputs d3f036814a2eaa7d97391880f9e744321427ab28e5bb8589d92c0dbe3337b2ac,
  2f2766c1ff4d53b42137a81930bc5f8e70a4d25b6ed4668274966658cc6c46d2,
  and e3694236343031f8398e46ab8a65d3a3a8d7ceb0c5733c2381f755ed48ae2409.
  Each then failed only when Get-Item rejected the identical nonexistent path
  C:\Program Files\Git\mingw64\bin\zstd.exe. No Windows preflight/save ran.
  Aggregate job 97546762267 observed the three green and three failed producer results
  plus all six distinct digests, rejected before checkout/restore/output, and verifier
  97546807321 was skipped. The run retained zero artifacts and performed no campaign,
  signing, lifecycle, custody, holdout, commitment, reveal, or scientific action.

  All Windows jobs identify hosted image windows-2025-vs2026 version
  20260818.207.1 and link the exact runner-images tag win25-vs2026/20260818.207,
  commit 0900c002193dc2d3fade0cd9133ae70e7088eb05. At that immutable source,
  Install-Zstd.ps1 SHA-256
  17653b59e05c3a52f8281f6a232f2e459c177f4d0c154bc42f8772553d7acf92
  installs the upstream win64 release beneath C:\tools\zstd and adds that exact
  directory to machine PATH. The image inventory SHA-256
  4cd6067bec8e0eb5ac04e1134fb71f23b99ca296caeab1cc5d30ca4838e218ba
  declares zstd 1.5.7; its Pester source runs zstd from PATH. Therefore the image-owned
  executable location is C:\tools\zstd\zstd.exe, not a Git installation subtree.
  The failed job did not stat or hash the correct executable, so its actual runtime
  file hash remains intentionally unclaimed and must be captured and rechecked by the
  corrected trusted step. A separate diagnostic is unnecessary: the next exact six-
  cell proof can fail closed while establishing the runtime path/hash/version and the
  cache save itself.
findings:
  - id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:201-249,410-571; reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml:201-249,410-571"
    evidence: |-
      All three Windows jobs on image 20260818.207.1 passed containment, staged one
      exact envelope, and emitted a distinct valid digest. The shared archive-tool
      assertion then failed at Get-Item because
      C:\Program Files\Git\mingw64\bin\zstd.exe does not exist. The job-level
      sanitized PATH also omits the image's zstd installation directory, so changing
      only the assertion would not establish which executable the pinned cache action
      resolves. The exact runner-image tag's install script places the standalone
      package under C:\tools\zstd and adds that directory to machine PATH, while its
      exact inventory declares zstd 1.5.7. Git's mingw64 tree is not the installation
      source. Ubuntu's three exact caches prove the transport design works on the
      other platform; aggregate and verifier correctly failed closed with zero
      artifacts.
    finding: "The Windows proof workflow binds zstd to a nonexistent Git-relative path and omits the actual image-owned zstd root from its exact child/action PATH."
    failure_scenario: |-
      Every otherwise-valid Windows containment cell deterministically stops before
      cache preflight/save. Alternatively, a path-only assertion edit without an exact
      action PATH and pre-use hash recheck lets actions/cache discover a different
      zstd executable than the one the trusted step inspected.
    consequence: "No Windows envelope cache, six-cell aggregate, retained seven-file proof, or independent redownload verification can exist; R3 approval remains blocked."
    required_action: |-
      Correct only the proof workflow and its exact receipt copy to bind the current
      hosted-image standalone installation at the literal canonical path
      C:\tools\zstd\zstd.exe. Do not derive zstd from Git, use Get-Command/where/PATH
      discovery as authority, or accept the first member of a multi-location
      allowlist. The authoritative image recipe supplies one installation root; an
      allowlist would silently turn future image drift into acceptance instead of a
      reviewable fail-closed event.

      Extend the exact Windows PATH assertion to include C:\tools\zstd in one fixed
      position while retaining only the already trusted PowerShell, Git usr/mingw/cmd,
      and Windows system roots. This is necessary because pinned
      actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 invokes GNU tar/zstd
      for enableCrossOsArchive through its process environment. At the trusted archive
      step, require the canonical directory and executable to exist as ordinary,
      non-reparse objects; reject reparse ancestors, alternate streams, links or path
      aliasing as applicable; require the exact 1.5.7 banner; capture SHA-256 from that
      exact file; and prove PATH command resolution selects that same canonical file.
      The trusted pre-save step must receive the captured hash as control-plane data,
      restat the exact path, recheck path/type/reparse/version/hash and staged-envelope
      descriptor/hash, and leave the immediately following pinned cache/save action no
      intervening untrusted step. Do the analogous recheck before any cache operation
      that actually invokes compression/extraction. Do not claim a binary hash that
      this failed run never observed.

      Add source regressions asserting the exact standalone path, exact PATH, exact
      version, hash capture/recheck ordering, action pin, workflow/receipt equality,
      and rejection of missing, wrong-version, changed-hash, PATH-shadow, reparse,
      alias, Git-relative, and multi-allowlist substitutions. Then rerun the exact
      proof at the amended immutable source. Acceptance requires all six producers
      green with six distinct digest-bound caches, aggregate exact-key restore and
      in-memory validation, exactly one retained seven-file artifact, a green
      redownload verifier, zero extra/missing files, and unchanged fail-closed/no-
      scientific behavior. No separate diagnostic run is required.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001"
    description: "Bind and mutation-test the exact C:\\tools\\zstd\\zstd.exe path, 1.5.7 banner, captured/rechecked SHA-256, canonical non-reparse identity, and sole PATH resolution immediately around pinned cross-OS cache use; then pass and retain the exact six-cell aggregate plus redownload verifier."
    rationale: "The hosted run resolves containment/export but proves that a wrong transport executable path alone prevents the mandatory retained evidence; the same test must also exclude assertion/action executable divergence."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Supported hosted shells and fixed command boundaries executed in all producer jobs.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "The proof-only Windows archive tool identity is wrong and no retained exact proof exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Six containment/envelope stages pass, but transport prevents retained independent inspection.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Six teardown/staging steps pass, but no retained six-envelope subject exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted proof reaches six digests but not the aggregate/artifact/verifier acceptance boundary.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 stage canonical envelopes and pass cross-step export descriptor/hash validation through digest output.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Windows canonical envelopes exist but are not cached or retained.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Only the three Ubuntu cache entries exist; Windows transport stops before preflight/save.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in Windows control-plane steps execute and enforce their same-step postconditions.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "All Windows producer steps executed through supported built-in pwsh; no custom dot-source shell remains in this path.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute successfully in all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 pass the production no-LUA explicit-low-IL containment and zero-descendant boundary.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "All proof scripts parsed and executed through the affected boundaries.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu production proof passes 3/3 and saves three caches.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 pass exact root/file descriptor checks at staging and digest boundaries.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved diagnostics and exact production descriptor assertions execute green in all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "The exact production native root setter, explicit file owner, raw/managed requery, low-IL denial, envelope hash, and cross-step checks pass Windows 3/3.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "The requested retained-proof test remains incomplete only because of the distinct later transport defect."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Supported hosted command boundary executes in all six cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Windows zstd identity is incorrect and the exact retained proof is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Six execution contexts pass, but final retained closure is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Six teardown gates pass; retained subjects are absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Six digests exist, but aggregate and verifier did not run.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows export succeeds 3/3, but the requested retained proof is not complete.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Windows envelopes never cross the cache channel.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Three Ubuntu caches and zero Windows caches exist.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Built-in control plane and digest outputs execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in pwsh passes all producer boundaries, but the requested retained final proof is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "No-LUA low-IL production passes 3/3; retained aggregate acceptance remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Six production/receipt proof scripts parse and hosted execution crosses the former parse boundary.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu production proof passes 3/3 and caches all envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: unresolved, evidence: "Exact Windows medium export passes 3/3, but retained cross-cell acceptance is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved exact diagnostic and production assertions.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: unresolved, evidence: "Native root/file descriptors and low-IL attacks pass all Windows cells; its final retained 2x3 acceptance clause is blocked only by zstd transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY"
  predicted_outcome: "The exact standalone zstd path/PATH/hash/version correction will allow all three Windows exact-key cache saves and the six-cell aggregate plus verifier to complete."
  predicted_failure_mode: "If the hosted image drifts or actions/cache resolves a different executable, the exact path/version/hash/resolution recheck rejects before cache save and no aggregate/artifact is emitted."
  confidence_statement: "High confidence that the observed failure is a bounded proof-transport path defect and that no separate diagnostic is required; the fresh complete proof remains authoritative. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Correct the proof workflow and receipt to the one authoritative standalone zstd path with exact PATH, version, hash, resolution, and pre-use rechecks; add substitution regressions; then rerun the complete retained 2x3 proof. RR18 containment/export is verified resolved and architecture remains viable. Merge, signer/custodian work, refreeze, activation, and holdout are not authorized."
```

## Verification ledger

- Exact handoff/content/tree/sole-parent/origin identities matched. The handoff changed only the RR18 response and campaign/packet review metadata over the RR18 content; proof workflow and receipt are byte-identical with Git-normalized SHA-256 `183ac47877fc422e646a6dee829120cea0ff3f0063f29e7d10efa6852ad4000b`.
- Run `32763190366` and every job/source/log timeline were inspected. Ubuntu producer jobs `97546576015`, `97546575802`, and `97546575949` saved cache IDs `6946922883`, `6946924719`, and `6946926355`; Windows producer jobs `97546575887`, `97546575857`, and `97546575869` failed only at the nonexistent zstd assertion after valid digest output; aggregate `97546762267` failed closed and verifier `97546807321` skipped. GitHub reports zero run artifacts.
- Exact runner-image release/tag, commit, inventory, install recipe, and tool test establish the standalone `C:\tools\zstd` root and zstd `1.5.7`. The reviewed jobs did not observe the correct file hash; this review does not invent one.
- Focused canonical transport, cache binding, parser closure, Ubuntu canonical-value, Windows low-IL export, and RR18 native descriptor tests passed 6/6 in 50.07 seconds. Review-guidance tests passed 9/9. Six production/receipt PowerShell proof scripts parsed with zero errors. Campaign validator reported valid. Full suite, Ruff, and TeX were not rerun because this narrow review changes no implementation and the decisive hosted defect precedes cache transport.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No R3 production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
