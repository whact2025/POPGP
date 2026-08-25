# VIA-000 R3 recovery-protocol independent re-review 27

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-27"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-27"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "fe83901ca3a78c99de1a199d21e6a157ba5c6308"
baseline_commit: "afe5c67c1577e2131944384893df57d3e3d757e9"
prior_review_ref: "afe5c67c1577e2131944384893df57d3e3d757e9:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26.md"
builder_response_ref: "fe83901ca3a78c99de1a199d21e6a157ba5c6308:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26-RESPONSE-1.md"
context_hash: "fade7016544ae26b817ab7c42c2a619251876a26"
context_hash_method: 'git rev-parse "fe83901ca3a78c99de1a199d21e6a157ba5c6308^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26-RESPONSE-1.md"
  - ".github/workflows/via000-r3-ubuntu-pwsh-compound-diagnostic.yml at a0eb47d2405c17d8f593823f5c9bb7566b59936a"
  - "actions/setup-python src/find-python.ts at ece7cb06caefa5fff74198d8649806c4678c61a1 via GitHub API"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, exact GitHub proof run 32796722783/jobs/artifact, generic CI run 32796722808, and prior diagnostic run 32791645412/job 97634217441; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-27 at exact handoff
  fe83901ca3a78c99de1a199d21e6a157ba5c6308. Exact origin, branch, handoff/content
  ancestry, trees, response/review hashes, workflow source, full run/job logs and API,
  downloaded retained bytes, frozen local verifier, pinned setup-python source, prior
  diagnostic, schema/parser/guidance checks, generic CI, and campaign state were
  inspected. No implementation, campaign, lifecycle, or scientific state was changed.
  Operator and orchestrator are shared; session, worktree, and branch differ. Builder
  model is shared and external scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, bounded observability blocker. RR26's
  extensionless lowercase-UUID correction is present, but proof run 32796722783
  still rejects at the same pre-metadata compound predicate. The run does not expose
  which term differs. The strongest evidence-weighted cause is a context difference
  omitted from the RR25 diagnostic: the aggregate job runs pinned
  actions/setup-python before the normalizer with update-environment true and exact
  pythonLocation `/opt/hostedtoolcache/Python/3.11.15/x64`. The pinned action source
  calls `core.addPath(installDir)` and `core.addPath(_binDir)`, whereas production
  still requires the pre-action in-process PATH
  `/opt/microsoft/powershell/7:/usr/bin:/bin`.

  The log preamble's `PATH: /usr/bin:/bin` is not evidence of the PowerShell
  process's effective entry PATH. The prior diagnostic itself demonstrates this:
  the same preamble preceded a directly observed in-process PATH with PSHOME
  prepended. Setup-python also exports pythonLocation, Python_ROOT_DIR variants,
  PKG_CONFIG_PATH, and LD_LIBRARY_PATH in every subsequent aggregate step. It is
  therefore highly likely that Python's bin/root entries are also present between
  PSHOME and `/usr/bin:/bin`; however neither their exact order nor every current
  compound term is logged. A direct correction to a guessed PATH is not reviewable.
  One exact-context, no-artifact Ubuntu diagnostic is required; no full 2x3 replay
  should precede it.

  Identity is exact. Handoff fe83901ca3a78c99de1a199d21e6a157ba5c6308 has
  tree fade7016544ae26b817ab7c42c2a619251876a26 and sole parent/content
  df6528a75fcc5f1f4b0792b155b350e118577139, tree
  559cc795d44171d65ea3d8303f6854df0872cd7c; origin matches. Imported RR26 review
  normalized SHA-256 is 32c93262b49ca662a9abd4288f3d035a259dbd244f588ed2cdbedb850bd1ff66;
  RR26 response normalized SHA-256 is
  b185c4e49227602a4f7e6ba207f5a8b9e5bfcce2c23b37d2349064f00f62bef9.

  All six producers passed exact containment, staging, digest, and cache gates.
  Aggregate job 97649557396 required six distinct outputs, restored six exact
  cache keys, validated all envelopes, created seven retained files, and uploaded
  artifact 9545097332 before the normalizer rejected. API identity is exact: name
  `via000-r3-containment-proof-retained`, size 48014, digest
  `sha256:eedbc008ae16f6c6f9eeb5cda7b2f4fc941ec13b2cbb70b8494fb061ae4a8752`.
  Independent download contained exactly six envelopes plus aggregate.json; all
  file hashes matched the six job digests and the frozen aggregator accepted the
  run/source/workflow/2x3 bindings. This is useful non-scientific evidence but not
  an accepted proof: normalizer outputs are absent and verifier job 97649643116 is
  skipped. Generic CI run 32796722808 is independently green. The artifact declares
  no campaign execution, candidate checkout, custody, lifecycle, commitment/reveal,
  or scientific action; it cannot authorize later gates.
findings:
  - id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:723-727,807-838 at fe83901ca3a78c99de1a199d21e6a157ba5c6308 and mirrored receipt"
    evidence: |-
      Aggregate step 5 runs setup-python commit
      ece7cb06caefa5fff74198d8649806c4678c61a1 with Python 3.11.15,
      check-latest false, and update-environment true. The log records exact
      pythonLocation `/opt/hostedtoolcache/Python/3.11.15/x64` in every later step.
      The pinned action's find-python.ts calls addPath for both installDir and its
      bin directory. Production nevertheless compares live entry PATH to the same
      pre-action value used by diagnostic 32791645412, which had no setup-python.
      Run 32796722783 then fails only at the opaque pre-metadata compound. The
      explicit log environment cannot settle the live value: the prior diagnostic
      logged explicit `/usr/bin:/bin` but observed PSHOME-prepended PATH in-process.
    finding: "The normalizer binds an unobserved pre-action PATH although setup-python mutates the later-step process environment; the opaque compound cannot prove the exact live PATH or sole mismatching term."
    failure_scenario: |-
      A fresh valid 2x3 proof reaches upload, then the literal pwsh process inherits
      setup-python GITHUB_PATH additions. The normalizer compares that context to a
      path observed only in a job without setup-python, rejects before outputs, and
      skips the retained verifier. Guessing one prefix/order risks another red proof
      or broadening the trusted tool boundary.
    consequence: "The hosted proof gate remains red and neither the retained artifact nor any downstream merge/signer/custodian/lifecycle gate is authorized."
    required_action: |-
      Run one minimal experiment-branch Ubuntu 24.04 diagnostic, not the six-cell
      proof. Mirror the aggregate job's explicit PATH, exact pinned checkout/tool
      assertion if retained, and exact pinned setup-python action/inputs
      (`3.11.15`, check-latest false, update-environment true), then invoke the exact
      literal `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive
      -File {0}` shell. Before any compound rejection, log only nonsecret normalized
      observations and an individual boolean for MainModule, PSHOME, full version,
      effective entry PATH, argc, argv0-5, RUNNER_TEMP parent, extensionless UUID,
      ordinary/non-reparse/single-link/runner-owned/0644 metadata, setup-python's
      python-path output, and pythonLocation. Then set PATH to `/usr/bin:/bin` and
      prove the ordered sanitization. Do not checkout candidate code, restore/save a
      proof cache, upload an artifact, write job outputs, or enter campaign/lifecycle
      paths.

      If and only if that exact-context vector isolates PATH, amend production and
      receipt to require the exact observed entry string/order (expected to include
      literal PSHOME plus exact pinned Python bin/root entries), immediately replace
      it with `/usr/bin:/bin`, and preserve all RR26 script-object, raw artifact,
      output-control, and cross-step checks. Add fixtures for the exact positive and
      reject missing, extra, reordered, alternate-root, symlink-substituted, or
      unpinned Python path entries. If another term fails, correct only that observed
      term. Remove the diagnostic, update hashes/receipts, then rerun the exact fresh
      2x3 proof and require green normalizer plus green redownload verifier.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001"
    description: "After exact production setup-python inputs, independently log every pre-metadata normalizer term and exact effective entry PATH, isolate one mismatch, then bind only the observed value and require a fresh green 2x3 retained proof/verifier."
    rationale: "The current compound and step preamble cannot distinguish setup-python's persisted GITHUB_PATH effect from any other term; one no-artifact exact-context observation is cheaper and safer than another guessed full replay."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six exact cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported literal shells execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts parse; hosted workflow reaches normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All live inner JSON subjects retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Normalizer/verifier remain red after exact upload.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher reaches opaque compound.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Original PATH mismatch isolated.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", outcome: unresolved, evidence: "Pre-setup PATH passed only in diagnostic context.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", outcome: verified-resolved, evidence: "No-setup diagnostic exposed all prior terms and extensionless argv5.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", outcome: unresolved, evidence: "Source has extensionless UUID and metadata checks, but the current opaque compound rejects before individual proof.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Hosted boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six contexts pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes verify.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Supported literal shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows low-IL subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight scripts parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows export boundaries retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live writers and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Normalizer/verifier remain blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher reaches opaque compound.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "Original diagnostic isolated PATH but redacted argv5.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", outcome: unresolved, evidence: "Pre-setup PATH passed only in diagnostic context.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", outcome: superseded, evidence: "No-setup diagnostic isolated its context; exact setup-python context is now required.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", outcome: unresolved, evidence: "Source fixture is corrected; current compound prevents an individual live disposition.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC"
  predicted_outcome: "Every current term except entry PATH passes; live entry PATH contains exact PSHOME and both setup-python toolcache additions before `/usr/bin:/bin`."
  predicted_failure_mode: "If exact order or another predicate differs, the diagnostic names it without artifact, campaign, or lifecycle side effects; no guessed production amendment is authorized."
  confidence_statement: "High confidence setup-python context explains the new failure; insufficient direct evidence to bind the exact string/order without one exact-context diagnostic. Architecture remains viable. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. The only authorized next action is one exact-context, no-artifact Ubuntu setup-python/Pwsh predicate diagnostic. After it, only the observed production/receipt correction, fixtures, hash updates, diagnostic removal, and a fresh exact 2x3 proof through a green normalizer and redownload verifier may proceed. Merge and all signer/custodian/refreeze/activation/holdout/scientific gates remain unauthorized."
```

## Verification ledger

- Exact handoff/content trees, sole-parent chain through imported RR26, remote campaign/review/experiment refs, artifact-only handoff scope, and normalized RR26 review/response hashes matched.
- Full run/job/API/log evidence was inspected. Six producers, six exact cache restores, aggregate validation, seven-file creation, and upload passed. Failure is exact at aggregate step 15's pre-metadata compound; verifier skipped. Generic CI at the same SHA passed.
- Artifact 9545097332 was downloaded read-only. It contains exactly seven ordinary files. Their SHA-256 values match all six producer digests plus aggregate.json, and the frozen reviewed aggregator accepted exact repository/source/ref/workflow/run/attempt and 2x3 identities. This does not cure the missing normalizer/verifier gates.
- RR25 diagnostic source/log proves the pre-setup context and extensionless runner script. Exact pinned setup-python source proves it requests two PATH additions; production logs prove setup-python ran first and exported its exact tool root. The effective later-step PATH/order remains unlogged, so the proposed diagnostic is intentionally narrower than another full proof.
- Review YAML/schema/parser/guidance, PowerShell parse closure, campaign validator, exact diff, and normal/ignored cleanliness were checked. Broad unit/full suites, TeX, and Ruff were not repeated because generic CI is exact-SHA green and the hosted compound failure is decisive.
- Campaign remains drafted, `holdout_started: false`, unrevealed, and pending. No tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
