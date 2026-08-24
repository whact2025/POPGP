# VIA-000 R3 recovery-protocol independent re-review 20

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-20"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-20"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "93347027f764a2d112e79d63cb1d95c6292f310a"
baseline_commit: "a910a27c6e2c2cd02724ae4445da21d581b0335a"
prior_review_ref: "a910a27c6e2c2cd02724ae4445da21d581b0335a:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19.md"
builder_response_ref: "93347027f764a2d112e79d63cb1d95c6292f310a:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19-RESPONSE-1.md"
context_hash: "d7c69be13061d272688e6796089c1b61ecee9c07"
context_hash_method: 'git rev-parse "93347027f764a2d112e79d63cb1d95c6292f310a^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32767703776 job/log/cache/artifact metadata; cache payload download is not exposed by the public post-run API; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-20 at exact handoff
  93347027f764a2d112e79d63cb1d95c6292f310a. Exact origin, remote head,
  sole-parent content ancestry, trees, review/response history, proof/receipt writer
  and validator sources, all eight hosted job timelines/logs, six cache records,
  focused transport regressions, direct Windows byte reproduction, parser closure,
  guidance, and campaign state were inspected. No implementation, campaign, cache,
  lifecycle, or scientific state was changed. Operator and orchestrator are shared;
  session/worktree/branch differ. Builder model is shared and external scientific
  validation is not claimed.
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
  CHANGES REQUESTED with one high-severity canonical-inner-JSON writer blocker.
  RR19's zstd identity/cache boundary is verified resolved. All six producers and
  all six exact digest-bound cache saves/restores passed. The aggregate then failed
  closed before creating output because the decoded Windows candidate proof.json
  contains CR bytes. Architecture remains viable, but the retained exact 2x3 artifact
  and verifier are still absent, so no merge or lifecycle action is authorized.

  Handoff 93347027f764a2d112e79d63cb1d95c6292f310a has exact tree
  d7c69be13061d272688e6796089c1b61ecee9c07 and sole parent/content
  c11b678f20da5a0f4312c30fd23f7eb69f4db089 with tree
  a312bbb2ee68b10a6c12a1b1cd2bd04bb8575810. Origin's campaign head matched.
  The handoff adds only the RR19 response and campaign/packet review metadata over
  the remediation content. The imported RR19 artifact retained Git-normalized
  SHA-256 624df3e5dcfa1639068a842eadff86e19fd507209542230e381fc320da6219c2;
  its response retained 577bfb474587f8173069e98d93d2dea372068ea7d6811364a0e7f7ab7f3c207c.

  Hosted push run 32767703776 attempt 1 was exact at the handoff. Producer jobs
  97560823330/97560823399/97560823437 and
  97560823375/97560823442/97560823069 were all green. They emitted six distinct
  digests and saved exact caches 6948336868/6948335773/6948337858 and
  6948341459/6948345917/6948341323. Windows 3/3 proved the literal
  C:\tools\zstd\zstd.exe path, version 1.5.7, unique PATH resolution, and identical
  pre/save/post SHA-256
  8076aae03feac7c66b319579e82172eed168deed2a3f25e5e2d3c60f55e84111.
  Aggregate job 97561040230 required six green/distinct producers, restored all six
  exact keys, and passed hit=primary=matched equality. It then reported exactly:
  "inner proof via000-r3-containment-proof-windows-x86_64-candidate/envelope.json is
  not UTF-8/LF/no-BOM JSON." Upload was skipped, verifier 97561190202 was skipped,
  the API reports zero artifacts, and no lifecycle or scientific path ran.

  The error is not the outer envelope. The producer creates that envelope with
  JsonSerializer.SerializeToUtf8Bytes plus one explicit 0x0A. The aggregate verifies
  its cache digest, strict parse, and canonical byte reserialization before decoding
  members. It failed later at _validate_cell while parsing decoded proof.json. That
  member is written by `ConvertTo-Json | Set-Content -Encoding utf8NoBOM`. On Windows,
  ConvertTo-Json itself emits CRLF between indented lines and Set-Content appends a
  final CRLF. A direct local byte reproduction under PowerShell 7.6.4 produced no BOM
  but five CR and five LF bytes, including final CRLF; the ConvertTo-Json string
  already contained four CR/LF pairs. The exact hosted error and source therefore
  establish the cached defect without inventing inaccessible payload bytes.

  The same latent defect occurs in containment-result.json: the production helper
  uses the identical pipeline at its initial write and its Ubuntu post-account-removal
  rewrite. The aggregate parses proof.json first, so it has not yet reached the result
  member. Fixing only proof.json would deterministically expose the result failure in
  all Windows cells. Both inner JSON writers and their receipt copies must be corrected
  together; stdout/stderr and the already-canonical outer envelope need no change.
findings:
  - id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:496-497; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:918-966; corresponding receipt copies; VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py:165-167,320,401"
    evidence: |-
      All six outer envelope digests survived exact cache save/restore and the outer
      canonical decoder. The first Windows cell then failed only when `_parse_json`
      rejected CR in decoded proof.json. Its writer is the platform-sensitive
      ConvertTo-Json/Set-Content pipeline. Direct Windows reproduction proves that
      utf8NoBOM removes the BOM but neither converts ConvertTo-Json's internal CRLF nor
      changes Set-Content's final platform newline. The production containment helper
      writes decoded containment-result.json with the same pipeline twice. Existing
      tests synthesize inner members with Python `json.dumps(...)+b"\n"`; they do not
      execute or byte-check the production PowerShell JSON writers, explaining why
      focused tests remained green.
    finding: "The Windows producer hashes and transports platform-CRLF inner JSON members even though the frozen aggregate accepts only strict UTF-8, no BOM, and LF-only JSON."
    failure_scenario: |-
      Each Windows envelope remains hash-exact through cache transport but aggregate
      validation rejects proof.json at the first CR. If only that writer is changed,
      containment-result.json reaches the same rejection next. No retained 2x3 proof
      can be produced.
    consequence: "The aggregate/upload/verifier gate cannot complete, so independent hosted containment acceptance and R3 approval remain blocked."
    required_action: |-
      Introduce one reviewed canonical JSON byte writer in the frozen containment
      helper and use it for both containment-result.json writes and the proof runner's
      proof.json write (the runner already loads the hash-bound helper). Serialize a
      compact object with no platform line separators, encode using strict UTF-8
      without BOM, append exactly one final byte 0x0A, and reject any BOM, CR, embedded
      LF, missing final LF, or extra trailing bytes. Write through an explicit file
      mode appropriate to the fresh proof and rewritable containment result, flush,
      close, read all bytes back, and require exact byte/hash equality before those
      bytes enter subject hashing. Do not use Set-Content, Out-File, default encoding,
      Environment.NewLine, or a text writer with platform newline defaults.

      Preserve every existing member size/hash/base64, envelope canonicalization,
      schema, identity, cache key/hit, link, descriptor, cleanup, aggregate, and output
      check. Mirror the helper and runner receipt copies byte-for-byte and update all
      protocol/packet/manifest receipt hashes. Do not relax `_parse_json` to accept CRLF
      or normalize after hashing/transport; producers must emit the frozen bytes.

      Add byte-level tests that execute the actual production writers on Windows and
      Linux, read proof.json and every final containment-result.json write as bytes,
      require no BOM/CR, strict UTF-8, compact JSON plus exactly one trailing LF, and
      verify stable read-back hashes. Decode both inner members from actual Windows
      outer envelopes and feed them through the real aggregator. Add BOM, CRLF, bare
      CR, missing LF, double LF, trailing space, invalid UTF-8, rewritten-after-hash,
      and source/receipt divergence mutations, each requiring zero aggregate output.
      Exercise all candidate/pdf/mutation cells and then rerun the exact hosted 2x3
      proof. Acceptance requires six green producers, exact caches/hits, one retained
      seven-file artifact, and a green independent redownload verifier. No separate
      transport diagnostic is required.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001"
    description: "Execute the real cross-platform proof/result JSON writers, require strict UTF-8 without BOM or CR and exactly one final LF for both decoded members, adversarially mutate every encoding/newline case, aggregate all six actual envelopes, and pass the retained artifact plus redownload verifier."
    rationale: "Python-synthesized canonical members masked a deterministic PowerShell Windows byte contract violation; only production-writer bytes and the real aggregate close the gap."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "All hosted command boundaries execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Producer tools pass, but retained exact evidence remains absent.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Six producer contexts pass; decoded retained evidence is invalid.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Six teardown/staging gates pass, but no retained validated subjects exist.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted proof reaches exact cache restore but not artifact/verifier acceptance.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export, hash, and cache their exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Outer transport is exact, but decoded Windows JSON is noncanonical and cannot be retained.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "All six distinct digests, cache saves, exact restores, hits, primary keys, and matched keys pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control-plane steps execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported built-in pwsh executes all Windows steps.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 pass production low-IL containment and teardown.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Scripts parse and execute through hosted proof.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 pass and restore exact caches.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 pass exact descriptor checks through post-cache recheck.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved exact diagnostic and production assertions.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass Windows 3/3 through post-cache recheck.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 prove exact path/version/resolution/hash before and after successful cache save.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "All hosted command boundaries execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained exact proof.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Decoded retained evidence is invalid.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No retained validated subjects.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Aggregate output and verifier remain absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows export/cache succeeds; retained final proof remains incomplete.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Decoded Windows members violate the frozen byte contract.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Cache channel passes, but its required retained aggregate/verifier clause is incomplete.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in pwsh passes; retained final proof is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "Low-IL production passes; retained aggregate acceptance is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Six source/receipt proof scripts parse and hosted execution passes the former boundary.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 aggregate inputs pass until the first Windows cell.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: unresolved, evidence: "Windows boundary passes 3/3; retained cross-cell acceptance is absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: unresolved, evidence: "Native descriptors pass; final retained 2x3 clause is blocked by inner JSON bytes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: unresolved, evidence: "Exact zstd/cache identity passes 3/3; its retained artifact/verifier clause is incomplete.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES"
  predicted_outcome: "One compact UTF-8/no-BOM/one-LF writer used for both proof.json and containment-result.json will allow all six decoded members, aggregate, upload, and verifier to pass without changing transport or security predicates."
  predicted_failure_mode: "If any writer/receipt remains platform-text based or bytes change after hashing, byte-level tests or aggregate strict parsing rejects and no retained output is emitted."
  confidence_statement: "High confidence that this is a bounded deterministic writer defect and that architecture remains viable. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Replace all proof/result Set-Content JSON pipelines with one explicit canonical byte writer, mirror receipts/hashes, add actual-writer cross-platform and inner-member mutation regressions, then rerun the exact retained 2x3 proof. RR19 zstd/cache identity is verified resolved. Merge, signer/custodian work, refreeze, activation, and holdout are not authorized."
```

## Verification ledger

- Exact handoff/content/tree/sole-parent/origin identities matched. The handoff changed only the RR19 response and campaign/packet review metadata. Proof workflow and receipt are byte-identical with Git-normalized SHA-256 `a63f616626b5aae80876771c018433180e4c505afcdeccc19fd2f48530d085af`; containment helper/receipt and proof runner/receipt are also byte-identical.
- Every source/log timeline for run `32767703776` was inspected. Six producer jobs and cache IDs are listed above; aggregate `97561040230` restored all exact keys then failed on decoded Windows proof bytes before output-root creation; upload and verifier `97561190202` skipped; artifacts are zero.
- The public post-run cache API exposes exact cache metadata but no supported payload download. The exact aggregate error, writer source, outer-envelope validation order, and direct Windows ConvertTo-Json/Set-Content byte reproduction independently establish CRLF as the defect. No payload bytes are claimed beyond observed digests and metadata.
- Focused hosted-proof, Windows export, canonical envelope, cache transport, and RR19 zstd tests passed 5/5 in 57.89 seconds; their Python-generated LF members expose the missing production-writer byte test. Review-guidance tests passed 9/9. Six source/receipt PowerShell scripts parsed with zero errors. Campaign validator reported valid. Full suite, Ruff, and TeX were not rerun because this narrow review changes no implementation and the decisive hosted byte failure is already reproduced.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No R3 production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
