# VIA-000 R2 pre-holdout protocol-amendment independent re-review 3

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-3"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-protocol-amendment-independent-rereview-session-3"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-21"
commit_reviewed: "60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee"
baseline_commit: "71fc1fd6381a79a799ed1d4cc61dda640a22f382"
prior_review_ref: "4e262c2457be80f6b87544671832b3eed6a23db5:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2.md"
builder_response_ref: "60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee:reviews/codex/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-3.md"
context_hash: "ecd3dbd554c9bdb86c78f3f69f2438c1ae57411f"
context_hash_method: "git rev-parse \"60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee^{tree}\""
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
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-3.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-4.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attacks/VIA-000-R2-ATTACK-PLAN-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/mutation-runner.py"
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
  - "tests/unit/test_validation_artifact_contract.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
  - "uv.lock"
access_level: "public repository, public GitHub Actions metadata/logs/artifacts, and explicitly supplied public calibration assembly"
independence_statement: |-
  This was a fresh independent-reviewer task in isolated worktree
  C:\src\POPGP-via000-r2-protocol-rereview-3 on branch
  review/via000-r2-protocol-amendment-rereview-3 at exact response-containing
  handoff 60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee. HEAD, tree
  ecd3dbd554c9bdb86c78f3f69f2438c1ae57411f, origin, baseline ancestry,
  immutable prior-review reference, builder-response reference, and clean normal
  and ignored state were verified before conclusions. Prior review commit
  4e262c2457be80f6b87544671832b3eed6a23db5 is preserved as an immutable
  reference; its candidate copy has the identical Git blob even though the review
  commit itself is intentionally not an ancestor because remediation preserved the
  artifact by cherry-pick.

  Builder claims, positive fixtures, hosted status, control-plane metadata, supplied
  downloads, and the supplied assembly were treated as hypotheses. All eight prior
  finding IDs and all eight prior requested-test IDs were reconciled exactly once.
  Producer-attestation, coherent replacement, pseudo-PDF, opposed-raster, semantic
  recomputation, symlink-manifest, numeric-portability, repository-status separator,
  Windows CRLF/source-byte, receipt-ledger/artifact-ID, clone-contract, blockage,
  atomic no-commitment, custody/reveal, manifest, packet-rule, and hosted-artifact
  boundaries were independently replayed. A fresh assembly from the immutable
  platform fragments reproduced the supplied public calibration bytes exactly.

  The same human operator and Codex Desktop orchestrator are shared with the builder.
  Reviewer task, session, worktree, and branch are distinct. Builder model identity
  remains unknown, so model separation is not established. This is internal
  adversarial separation, not external scientific validation.

  No custody file, hidden holdout, secret seed, private evaluator, final label,
  restricted result, credential, or untracked handoff memo was accessed. The only
  raw result and output commitment inspected were the explicitly supplied public
  protocol-calibration assembly, not holdout or scientific-outcome material. R2
  remained preregistered, holdout_started=false, and unrevealed throughout.
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
  Approved with zero blocking findings for the narrowly scoped pre-holdout protocol
  amendment. The two blockers from re-review 2 are independently resolved. Each
  accepted summary and evidence manifest is now externally authenticated by a
  retained GitHub OIDC/Sigstore bundle bound to repository whact2025/POPGP, the
  frozen signer workflow, source commit 60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee,
  SLSA provenance, and a GitHub-hosted runner. All four exact subjects verify; wrong
  source, repository, workflow, cross-platform bundle, and post-signing evidence
  substitutions fail before output.

  Exact CI run 32444650755 and exact protocol run 32444650784 are terminal success.
  GitHub artifact IDs 9433870344 and 9434026263 report the registered Ubuntu and
  Windows archive digests. The downloaded fragments contain matching candidate and
  protocol identities, 16 successful typed command records per platform, 366 tests,
  six examples, 18 generated artifacts, twelve visuals, eighteen execution-derived
  mutation receipts, typed source/environment manifests, an eleven-page genuine PDF,
  allowlisted generated status, and empty final status. The 145 observed parametrized
  mutation nodes on each platform partition uniquely across the eighteen receipts.

  The unmodified fragments pass the authoritative public semantic validator. A fresh
  assembler invocation on this Windows reviewer seat reproduces raw-results SHA-256
  53259d0e944de26ab71ea1109c19b0e6eea49d2c02fee0abadb9fcadf2421019
  and output-commitment SHA-256
  b987b73ab463e1f28214655aabe32d86e57afe6fd8daa899f4caf53e4fe3a431
  byte-for-byte. Genuine Ubuntu typed symlinks, Windows literal Git worktree bytes,
  calibrated floating-point recomputation, repository-status framing, direct
  cross-platform rasters, and provenance copies all validate. Live cross-bundle and
  post-signing byte mutations return nonzero and create no output or commitment.

  This approval is not a campaign activation, holdout authorization, reveal, result
  adjudication, or scientific claim. The currently activated campaign intentionally
  remains on its older protocol snapshot and therefore fails the current checkout's
  manifest/contract-file validation before refreeze. Amendment 4 explicitly requires
  this zero-blocker review first. A maintainer must next refreeze the exact approved
  bytes, validate the newly frozen campaign, and obtain a new independent SAFE
  falsifier recommendation before any holdout transition.
findings: []
requested_tests: []
prior_finding_results:
  - finding_id: "VIA000-R2-PA1-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Both exact hosted platform-summary.json and evidence-manifest.json subjects
      verify with gh 2.97.0 against their retained Sigstore bundle under exact
      repository, signer-workflow, source-digest, SLSA-predicate, and hosted-runner
      constraints. Wrong source, wrong repository, wrong workflow, and the other
      platform's otherwise-valid bundle all return nonzero. Replacing the Ubuntu
      bundle in a complete coherent package makes the real assembler fail before
      creating its output directory or commitment. Post-signing artifact-byte
      mutation likewise fails against the signed manifest before output.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The frozen threat model still excludes compromise or collusion of GitHub's
      OIDC/Sigstore control plane and an authorized repository administrator. Within
      the declared untrusted transport/input boundary, producer provenance is now
      externally authenticated rather than self-asserted.
  - finding_id: "VIA000-R2-PA1-IDENTITY-001"
    outcome: verified-resolved
    evidence: |-
      Exact hosted summaries, signed manifests, assembled platform records, source
      manifests, mutation suite records, and all receipts agree on scientific
      candidate 5be3c38a0822d49953d0933f14ccab32ca12c896, candidate tree
      6ad387f9f4e0bab7f97df1bb54a03177887f0707, and protocol source 60a9a61.
      Focused negative controls reject commit/tree, platform, contract, schema,
      command-map, executable/argument, count, capability, and outcome contradictions.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No identity or asserted-Boolean substitution was accepted."
  - finding_id: "VIA000-R2-PA1-BLOCKED-001"
    outcome: verified-resolved
    evidence: |-
      blocked=true remains schema-invalid for a raw scientific result. Missing,
      partial, unavailable, failed, nonzero, or unattested fragments return nonzero
      and create no commitment; complete exact fragments assemble with blocked=false
      and derive failed=false from the retained evidence.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Unavailable or incomplete attempts cannot be converted into scientific pass/fail outcomes."
  - finding_id: "VIA000-R2-PA1-ASSEMBLY-001"
    outcome: verified-resolved
    evidence: |-
      The authoritative public validator returns [] on the supplied production
      calibration package. A fresh real assembler execution from the two immutable
      fragments reproduces both supplied output hashes byte-for-byte. Cross-bundle
      provenance and changed artifact bytes fail before any output directory exists.
      The 13-test raw/assembler suite covers missing, duplicate, changed, nonzero,
      partial, unsafe, malformed, identity, mutation, PDF, artifact, and semantic
      inputs plus atomic no-commitment and synthetic custody/reveal compatibility.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No assembler path committed unverified or semantically invalid evidence."
  - finding_id: "VIA000-R2-PA1-FREEZE-001"
    outcome: verified-resolved
    evidence: |-
      At exact handoff 60a9a61, the five unique protocol artifact IDs are present in
      the packet ledger, each campaign copy is byte-identical to its protocol source,
      each declared SHA-256 matches, and each raw Git blob at the handoff has the same
      SHA-256. Primary protocol and validator Git-blob digests are respectively
      9bca5f0b852627e588591e133820232be86e84e827b126b299b7e927bbfb1131
      and 9b529c287454b929e4c63aa33f25bf9171f9836ff346ff72cfc2bb5056b78db9.
      Canonical VIA-000 packet-rule SHA-256 is
      98bbe33c2c5f2a2a321675d12b47e074e83f92ee3463a131f398c81c21d3367b.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      CAMPAIGN.yaml still binds pre-amendment protocol commit da5dd8a and its old
      manifest digest, as Amendment 4 requires until approval. Exact-byte refreeze
      and campaign revalidation are mandatory next actions, not completed actions.
  - finding_id: "VIA000-R2-PA2-VISUAL-XPLAT-001"
    outcome: verified-resolved
    evidence: |-
      The direct opposed-raster negative remains in the passing authoritative suite
      and rejects the minus-four/plus-four pair. The production validator decodes and
      semantically checks all twelve visuals on both platforms and directly compares
      corresponding platform rasters under the frozen four-channel bound. All twelve
      Ubuntu scientific visuals were independently inspected; the retained plots and
      animation contain substantive, legible content. Both PDFs render to identical
      eleven-page platform raster sets.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Independent candidate-relative acceptance cannot mask cross-platform opposition."
  - finding_id: "VIA000-R2-PA2-CI-HISTORY-001"
    outcome: verified-resolved
    evidence: |-
      Exact-head ordinary CI run 32444650755 is terminal success at 60a9a61. Its
      full-history checkout resolves historical scientific candidate 5be3c38; Ruff,
      TeX-source validation, and all 379 tests pass. The frozen protocol workflow also
      clones full history with core.autocrlf=false before detached candidate checkout.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The former shallow-history and Windows checkout-byte regressions are closed."
  - finding_id: "VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    outcome: verified-resolved
    evidence: |-
      Exact protocol run 32444650784 is terminal success on Ubuntu and Windows.
      Ubuntu artifact 9433870344 reports archive digest
      sha256:19e67331e39b57572403034c795e1b5f37ed1a8851eab3aec2dd24f181162ba8;
      Windows artifact 9434026263 reports
      sha256:eb9b2826db4beef1874e5c7bf63f41e5914037b6e6f322ea3faf972e7a5db627.
      All four attested subjects verify. Ubuntu's 22,215 files plus four canonical
      symlinks and Windows' 21,319 files validate under disjoint typed shapes. Windows
      renderer.py is 8,341 bytes and hashes exactly like its candidate Git blob.
      Numeric recomputation passes at 2e-18 while the 4.46e-18 attack rejects. The
      exact fragments assemble and validate on this Windows reviewer seat, producing
      the registered canonical commitment.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The genuine cross-platform path is feasible and the registered boundary controls remain fail closed."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Dummy/hash-closed partial evidence rejects. A structurally coherent replacement
      without the correct external producer binding cannot verify: wrong source,
      repository, workflow, subject bytes, and cross-platform bundle all fail. The
      real assembler creates neither output nor commitment for the coherent
      cross-bundle control.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Authenticated provenance closes the no-execution substitution within the declared trust boundary."
  - requested_test_id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    outcome: verified-satisfied
    evidence: |-
      Commit/tree, source commit, platform-set, contract/schema, command map,
      executable/arguments/result streams, count, gate, capability, and outcome
      contradictions reject independently in the focused suite and production
      semantic replay.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "All accepted identities are derived from retained typed evidence."
  - requested_test_id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    outcome: verified-satisfied
    evidence: |-
      blocked=true and every incomplete, unavailable, partial, failed, nonzero, or
      unattested input fail validation/assembly and cannot create a scientific output
      commitment. Only complete evidence can encode blocked=false.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Blocked attempts remain operational states rather than scientific outcomes."
  - requested_test_id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    outcome: verified-satisfied
    evidence: |-
      Fresh exact-fragment assembly reproduces the canonical raw result and commitment.
      The focused suite confirms public semantic validation, atomic no-commitment for
      invalid inputs, and exact synthetic custody/reveal compatibility. Live
      attestation and artifact-byte attacks also fail before output.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Only the explicitly supplied public calibration commitment was inspected."
  - requested_test_id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    outcome: verified-satisfied
    evidence: |-
      Five unique protocol receipt IDs, their ledger records, protocol sources,
      campaign copies, declared hashes, and exact handoff Git blobs reconcile. The
      protocol, validator, and canonical packet-rule digests match Response-3.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Post-approval refreeze must preserve these exact bytes and recompute the campaign envelope."
  - requested_test_id: "TST-VIA000-R2-PA2-VISUAL-XPLAT-001"
    outcome: verified-satisfied
    evidence: |-
      The pairwise delta-eight opposed-raster control rejects. Exact hosted
      corresponding images decode, pass semantic reconstruction, and satisfy direct
      cross-platform comparison; visual inspection found no blank or substituted
      plot. Corresponding PDF-page rasters are byte-identical.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The direct relation is enforced in addition to candidate-relative checks."
  - requested_test_id: "TST-VIA000-R2-PA2-CI-HISTORY-001"
    outcome: verified-satisfied
    evidence: |-
      Exact ordinary CI run 32444650755 completes successfully at 60a9a61 with a
      full-history checkout and 379 passing tests. Historical candidate 5be3c38 is
      resolvable to its exact tree and source blobs.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Hosted CI no longer relies on an unavailable shallow-history object."
  - requested_test_id: "TST-VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    outcome: verified-satisfied
    evidence: |-
      Exact immutable Ubuntu and Windows fragments verify, validate, and assemble
      unmodified on Windows. Typed symlinks/files, 150,000 expanded-node bound,
      2e-18 recomputation tolerance, 4.46e-18 rejection control, literal Windows
      source bytes, direct raster comparison, canonical commitment, and synthetic
      custody/reveal all pass their registered positive or negative expectation.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No genuine supported-platform byte required normalization or weakening."
predictions:
  experiment_id: "VIA-000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-3-PREDICTION"
  predicted_outcome: |-
    Refreezing the exact reviewed bytes should yield a campaign-valid pre-holdout
    snapshot whose exact authenticated Ubuntu and Windows calibration fragments
    reproduce the canonical public raw result and pre-reveal commitment.
  predicted_failure_mode: |-
    The principal remaining procedural risk is refreeze drift: changing any reviewed
    protocol, workflow, validator, packet, schema, or receipt byte without repeating
    this evidence would invalidate the approval. GitHub control-plane or authorized
    repository-administrator compromise remains outside the frozen threat model.
  confidence_statement: |-
    High confidence within the declared boundary because exact hosted identities and
    artifacts, four external attestations, byte-level manifests, semantic validators,
    a fresh real assembly, live no-output attacks, focused tests, PDFs, visuals, and
    protocol/ledger bindings were independently reproduced.
recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    The complete historical finding and requested-test matrix is independently
    resolved/satisfied, exact hosted authority is green, immutable artifacts are
    authenticated, and the real cross-platform assembly deterministically produces
    the canonical validated calibration commitment. The amendment may proceed to
    exact-byte refreeze. Do not start holdout, access custody, reveal, commit a
    scientific result, advance lifecycle, or adjudicate an outcome until the refrozen
    campaign validates and a new independent falsifier recommends SAFE.
```

## Executed evidence

| Command or probe | Observed result |
|---|---|
| Exact identity, ancestry, immutable refs, and initial status | Commit `60a9a61f28c5b047a9bd5b2cff5a92cc1d731dee`, tree `ecd3dbd554c9bdb86c78f3f69f2438c1ae57411f`, isolated review branch/worktree, baseline ancestor `71fc1fd`, prior-review blob preserved, normal and ignored state clean before dependency setup. |
| Exact ordinary CI `32444650755` | Terminal success at exact head; Ruff green and `379 passed in 111.46s`. |
| Exact frozen protocol run `32444650784` | Terminal success for Ubuntu job `96661867221` and Windows job `96661867337`; clean runner, mutation matrix, two-subject attestation, retained bundle, and upload all green. |
| Artifact API identity | Ubuntu ID `9433870344`, 3,189,659 bytes, digest `sha256:19e67331e39b57572403034c795e1b5f37ed1a8851eab3aec2dd24f181162ba8`; Windows ID `9434026263`, 3,052,579 bytes, digest `sha256:eb9b2826db4beef1874e5c7bf63f41e5914037b6e6f322ea3faf972e7a5db627`; both bind run/head/repository and are unexpired. |
| Four exact Sigstore subject verifications | All pass with `gh attestation verify` under exact repo/workflow/source/SLSA/GitHub-hosted constraints. Wrong source/repo/workflow and cross-platform bundle fail. |
| Supplied production package validation | Public `validate_via000_raw_results` returns `[]`; raw SHA-256 `53259d0e944de26ab71ea1109c19b0e6eea49d2c02fee0abadb9fcadf2421019`; commitment SHA-256 `b987b73ab463e1f28214655aabe32d86e57afe6fd8daa899f4caf53e4fe3a431`. |
| Fresh exact-fragment production assembly | Real assembler completes in 138.5 seconds and independently reproduces both supplied hashes byte-for-byte. |
| Live coherent provenance substitution | Replacing the Ubuntu bundle with the valid Windows bundle returns nonzero at producer verification; no output directory or commitment is created. |
| Live post-signing artifact mutation | Appending bytes to one signed-manifest artifact returns nonzero on manifest-byte disagreement; no output directory or commitment is created. |
| Raw-evidence and assembler authority | `13 passed in 588.88s`; covers attestation, identity, blocked/failed/partial inputs, pseudo-PDF, direct opposed raster, command/count/manifest/mutation/status semantics, atomic no-commitment, and custody/reveal roundtrip. |
| Targeted numeric, visual, and custody controls | `4 passed in 37.68s`; honest 1.25e-18 moment drift passes, 4.46e-18 attack rejects, structured visual/annotation mutations reject, and custody leaks/role reuse/manifest mutations reject. |
| Mutation execution audit | On each platform, 41 frozen selectors expand to 145 observed PASSED nodes; eighteen receipts contain 145 unique nodes whose union exactly equals stdout, and every receipt is rejected=true with matching suite command/timestamps/hashes. |
| Source/environment/status audit | Both 560-entry source manifests validate against candidate Git objects. Ubuntu has 22,215 file plus four typed symlink entries; Windows has 21,319 file entries. Windows `popgp/renderer.py` is 8,341 bytes and SHA-256 `a09270caa9eb9890862d26648bccdcada8277a1b01fa80c07080bdf744aa44aa`, exactly matching its Git blob. Final statuses are zero bytes. |
| PDF and visual audit | `pdfinfo` reports unencrypted PDF 1.7, pdfTeX 1.40.29, eleven letter-size pages on each platform. Poppler rendered all 22 pages; corresponding platform page PNGs are byte-identical and all pages were visually inspected. All twelve retained Ubuntu visuals were decoded and inspected; cross-platform semantic/raster comparison passes. |
| Protocol/ledger/packet audit | Five protocol sources, five campaign copies, five ledger IDs, declared SHA-256 values, and exact handoff Git blobs reconcile. Packet rule is `98bbe33c2c5f2a2a321675d12b47e074e83f92ee3463a131f398c81c21d3367b`. |
| Pre-refreeze campaign validation | Expected nonzero: current campaign still binds protocol commit `da5dd8a` and old protocol-manifest digest. Amendment 4 expressly requires zero-blocker re-review before refreeze; no current protocol byte was altered to force premature activation. |

Local execution used Windows x86_64. TeX Live 2026 supplied `pdfinfo` and Poppler
rendering locally; PDF compilation itself was taken from both exact pinned hosted jobs
and not rerun as a reviewer mutation. CUDA/nvcc was not exercised because VIA-000
registers no accelerator requirement; the available Blackwell processor is outside this
protocol check. The GitHub artifact archive digests were verified from authenticated API
metadata; retained subject and manifest bytes were verified cryptographically and
semantically after extraction. No custody or hidden-holdout material was available.
