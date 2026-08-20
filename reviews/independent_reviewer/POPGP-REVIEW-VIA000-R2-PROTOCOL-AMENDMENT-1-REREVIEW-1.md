# VIA-000 R2 pre-holdout protocol-amendment independent re-review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-protocol-amendment-independent-rereview-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-20"
commit_reviewed: "e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a"
baseline_commit: "66eac5c7aede6595bd7402fc9196ea7d29203e6a"
prior_review_ref: "755a6a4d20bac2f4ca35432a98027918a4972d94:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1.md"
builder_response_ref: "e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a:reviews/codex/POPGP-RESPONSE-VIA000-R2-PROTOCOL-AMENDMENT-1.md"
context_hash: "145d59bf5f215ae8a4b9ac949bdb86094a563ab6"
context_hash_method: "git rev-parse \"e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a^{tree}\""
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
  - "pyproject.toml"
  - "reviews/codex/POPGP-RESPONSE-VIA000-R2-PROTOCOL-AMENDMENT-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/amendments/VIA-000-R2-PREHOLDOUT-AMENDMENT-2.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attacks/VIA-000-R2-ATTACK-PLAN-1.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/raw-results.schema.json"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
  - "uv.lock"
access_level: "public-repository-only plus public GitHub Actions metadata, logs, and artifact metadata"
independence_statement: |-
  This was a fresh independent-reviewer task in an isolated worktree and branch at
  exact response-containing handoff e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a.
  Before conclusions, HEAD, tree 145d59bf5f215ae8a4b9ac949bdb86094a563ab6,
  branch, source remote, original review and response ancestry, immutable review
  reference, response reference, and clean normal/ignored state were verified. The
  complete baseline-to-handoff diff and every changed protocol, validator, runner,
  assembler, schema, workflow, test, campaign, packet, manifest, receipt copy,
  amendment, and review/response artifact were inspected. The governing review
  documents and the complete scientific plan, matrices, and reproducibility contract
  were also read. Candidate, protocol, validator, workflow, test, campaign, custody,
  threshold, and lifecycle files were not modified; only this review artifact is added
  on the reviewer branch.

  Builder statements and positive fixtures were treated as hypotheses. The five prior
  findings and five prior requested tests were reconciled by stable ID. Accepted prior
  counterexamples were replayed and broadened through complete campaign validation,
  independently inconsistent identity/contract fields, malformed PDF bytes, opposed
  cross-platform visual drift, assembler commitment behavior, output-custody fields,
  runner failure paths, exact Git-blob binding, and exact-SHA hosted execution. The
  same human operator and Codex Desktop orchestrator are shared with the builder; the
  Windows identity NVIDIA.COM\rfuoco and builder alias fuocor refer to that same
  operator. The reviewer task/session, branch, and worktree are distinct. The builder
  model is unknown, so model separation cannot be established and is false. This is
  internal adversarial process separation, not external scientific validation.

  No custody file, hidden holdout, secret seed, private evaluator, output commitment,
  final label, restricted evidence, credential, or untracked handoff memo was accessed.
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
  Changes requested with four blocking scopes: two prior findings remain unresolved
  and two distinct findings are new. The response materially improves syntax,
  identity binding, fail-closed incomplete-attempt handling, deterministic assembly,
  and manifest byte binding. Exact mutations of top-level and per-platform commit/tree,
  parameter/contract/schema platform sets, command map/count, and blocked=true now
  reject. All ten protocol-manifest contract-file Git-blob digests, the requirements
  digest, and the VIA-000 packet-rule digest match the exact reviewed handoff. Those
  results resolve the prior identity, blockage, and freeze findings and satisfy their
  requested tests.

  The decisive evidence counterexample remains. A coherent package that executed no
  frozen command, but self-authored sixteen successful command records and streams per
  platform, copied the candidate's eighteen validation/visual artifacts and the R1
  PDF, invented a one-entry environment manifest, and supplied thirty-six arbitrary
  one-line mutation receipts was accepted by the raw-evidence helper and by the full
  R2 campaign validator with zero errors. Mutation receipts needed only the expected
  ID, rejected=true, a nonempty attack string, and a generic nonempty oracle_errors
  list. A separate 119,015-byte pseudo-PDF consisting of `%PDF-1.4`, repeated
  `not-a-pdf-object` text, and `%%EOF` was accepted after its hashes were updated. The
  validator does not parse PDF object structure, derive the page count, or enforce the
  declared platform PDF engine; it trusts a self-authored 11-page log. The frozen
  assembler directly returned zero and emitted raw-results.json plus
  output-commitment.json for that same invalid pseudo-PDF input.

  Assembly is therefore not fail-closed at the commitment boundary. The assembler
  performs JSON Schema validation but does not invoke the authoritative semantic
  evidence validator before atomic output. Its commitment contains raw_results_path
  and raw_results_sha256, while the packet custody contract and campaign validator
  require output_receipt_id and output_sha256. The public positive assembler test is
  built from the same synthetic fixture and checks its private hash field rather than
  a campaign custody-to-reveal round trip. Both exact-SHA hosted platform jobs ran 366
  tests, all six generators, artifact checks, and two eleven-page PDF passes, then
  failed at runner line 287 because the final Git status treats expected regenerated
  artifacts as residue before copying them. Consequently neither job produced a clean
  successful platform fragment. The retained workflow path includes the complete
  temporary environment: artifact metadata reports 712,758,054 bytes on Windows and
  17,876,913,630 bytes on Ubuntu.

  A distinct cross-platform visual counterexample also passes. At channel index
  [24,424,0] of chain_1d/results/clock_potential.png, the candidate value 31 was changed
  to 27 for Ubuntu and 35 for Windows. Each platform remains exactly at the allowed
  candidate-relative delta of four, but the direct platform-to-platform delta is eight.
  Updated hashes and summaries validate with zero errors because every raster is
  compared only to the candidate, while the cross-platform capability is the
  conjunction of the two individual results.

  Exact ordinary CI run 32420571769 is additionally red at this handoff: 366 tests
  passed and six raw-evidence/assembler tests failed because the shallow checkout did
  not contain historical candidate commit 5be3c38a0822d49953d0933f14ccab32ca12c896.
  Exact calibration run 32420571846 is cancelled overall with both execute steps
  failed. Focused public tests passed 6/6 locally, but their synthetic positive cannot
  outweigh the executed counterexamples or failed exact-SHA hosted runs. No local
  Ubuntu runner, nvcc, or Blackwell/CUDA path was exercised; accelerators are outside
  this CPU protocol. A local pdflatex execution was not repeated because the hosted
  pinned TeX engines completed both passes before the independent runner-cleanliness
  failure. This review does not authorize refreeze, holdout start, reveal, lifecycle
  advancement, a scientific outcome, merge, or custody commitment.

findings:
  - id: "VIA000-R2-PA2-VISUAL-XPLAT-001"
    severity: high
    category: code
    location: "scripts/check_viability_campaign.py:1534-1687,1857-1872"
    evidence: |-
      Starting from the accepted typed fixture, the reviewer changed one raster channel
      in examples/physics_qg/chain_1d/results/clock_potential.png. At [24,424,0], the
      candidate value is 31; Ubuntu was changed to 27 and Windows to 35, with manifest
      and artifact-result hashes updated. Each platform-to-candidate maximum delta is
      four, satisfying compare_visual_artifact, while the actual corresponding
      Ubuntu-to-Windows delta is eight. `_validate_raw_evidence_contract` returned `[]`.
      The validator calls compare_visual_artifact(reference, evidence_file) separately
      for each platform and defines cross-platform-reproduction as all platform-clean
      Booleans; it never compares the two retained platform rasters to one another.
    finding: |-
      The claimed cross-platform raster bound is not enforced pairwise. Two platforms
      may drift in opposite allowed directions and exceed the declared inter-platform
      tolerance while cross-platform-reproduction remains true.
    failure_scenario: |-
      Ubuntu and Windows render a meaningful feature on opposite sides of the
      candidate-relative tolerance. Both individual comparisons pass, summaries and
      capabilities are internally consistent, and the campaign accepts reproduction
      even though the platform-to-platform difference exceeds four channels.
    consequence: |-
      The frozen evidence cannot support its direct Windows/Ubuntu visual-reproduction
      claim, so a platform-dependent rendering disagreement can be adjudicated as a
      successful cross-platform result.
    required_action: |-
      After canonical decoding, compare every corresponding required raster directly
      between every required platform pair and enforce maximum per-channel delta <=4
      in addition to the candidate-relative checks. Retain the derived pairwise result
      and add an opposed-drift control where each side is candidate-relative in-bound
      but the pairwise difference is eight.
    verification: confirmed-by-execution
    blocking: true

  - id: "VIA000-R2-PA2-CI-HISTORY-001"
    severity: high
    category: governance
    location: ".github/workflows/ci.yml:10-12; tests/unit/test_viability_raw_evidence_contract.py:148-154"
    evidence: |-
      Public exact-SHA run 32420571769 has head SHA
      e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a and conclusion failure. The test job
      reports `6 failed, 366 passed in 122.85s`; each failing raw-evidence/assembler
      fixture reaches `git ls-tree -r -z 5be3c38a0822d49953d0933f14ccab32ca12c896`
      and exits 128 because actions/checkout@v4 used the default shallow history.
      Lint and TeX passed, but generation and artifact validation were skipped after
      the failed test step.
    finding: |-
      The authoritative ordinary CI workflow cannot execute the new retained-candidate
      evidence tests from a clean hosted checkout, so the response claim of a complete
      exact-tree green suite is not reproduced at the response-containing handoff.
    failure_scenario: |-
      A branch or future merge is evaluated in ordinary CI. The historical scientific
      candidate object is absent, the six evidence/assembly tests error before their
      assertions, and downstream regeneration/contract checks do not execute.
    consequence: |-
      Required protocol regressions are not continuously executable at the immutable
      handoff and a red exact-SHA quality gate prevents this commit from being a safe
      refreeze or merge candidate.
    required_action: |-
      Fetch sufficient retained Git history in ordinary CI, rerun the exact new handoff,
      require the complete suite plus regeneration and artifact validation to succeed,
      and retain the exact run identity. Add a guard that the historical candidate
      commit/tree exists before evidence fixtures execute.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VIA000-R2-PA2-VISUAL-XPLAT-001"
    description: |-
      For every required raster, directly compare canonical Ubuntu and Windows bytes.
      Require acceptance at pairwise channel delta four and rejection when opposed
      candidate-relative drifts of minus four and plus four create pairwise delta eight.
      Recompute the cross-platform capability from the pairwise results.
    rationale: |-
      Independent candidate-relative comparisons do not imply the registered direct
      platform-to-platform visual bound.
    blocking: true

  - id: "TST-VIA000-R2-PA2-CI-HISTORY-001"
    description: |-
      Run ordinary CI from a fresh exact-SHA checkout that retains candidate commit
      5be3c38a0822d49953d0933f14ccab32ca12c896. Require all tests, all six generators,
      and semantic/visual artifact validation to complete successfully.
    rationale: |-
      The exact reviewed handoff currently errors six tests because its ordinary CI
      checkout cannot resolve the frozen historical candidate.
    blocking: true

prior_finding_results:
  - finding_id: "VIA000-R2-PA1-EVIDENCE-001"
    outcome: unresolved
    evidence: |-
      The typed fixture is more detailed, but the retained result records, streams,
      PDF logs, environment manifest, and mutation receipts remain authored by the
      same package producer and are not independently derived. The reviewer built a
      coherent package that ran zero commands, self-authored sixteen success records
      and streams per platform, used an invented one-entry environment manifest,
      copied the candidate's exact eighteen artifact bytes and an R1 PDF, and used
      thirty-six arbitrary one-line mutation receipts with generic oracle_errors.
      `_validate_raw_evidence_contract` returned `[]`; after binding the receipt into a
      temporary exact R2 campaign at reproduced lifecycle, `validate_campaign`
      returned zero errors. Separately, both the helper and frozen assembler accepted
      a 119,015-byte framed but structurally invalid pseudo-PDF and the assembler
      emitted a commitment. The validator checks `%PDF`, a tail `%%EOF`, byte size,
      and a self-authored page-count log, but does not parse PDF structure/pages or
      enforce the retained pdf_engine value. Mutation receipt content is reduced to
      expected ID, rejected=true, nonempty attack, and nonempty oracle_errors.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Exact candidate artifacts in a fabricated positive do not establish that the
      frozen commands generated them or that the eighteen claimed mutations ran. If
      hostile or faulty evidence producers are outside the intended trust boundary,
      the claim and threat model must say so explicitly rather than describing these
      self-authored records as executable proof.

  - finding_id: "VIA000-R2-PA1-IDENTITY-001"
    outcome: verified-resolved
    evidence: |-
      Independent mutations of the top-level commit, per-platform commit, and per-
      platform tree reject. Reducing platform_families, reducing the frozen contract
      platform list, independently reducing the schema required list, changing the
      006-pytest command mapping, and changing required_command_count also reject.
      Observed errors include candidate commit/tree differs, malformed or contradictory
      raw_results_contract, raw-results schema platform set differs, and executable
      contract violations. These errors occur independently of recomputed failed and
      capability values.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Exact candidate identity and parameter/contract/schema/command-set agreement are
      now fail-closed validity invariants.

  - finding_id: "VIA000-R2-PA1-BLOCKED-001"
    outcome: verified-resolved
    evidence: |-
      The raw-results schema fixes blocked to false and the semantic helper independently
      rejects any true value. The assembler emits blocked=false only after two complete,
      zero-exit platform fragments and all thirty-six rejected mutation records; public
      negative execution confirmed a nonzero command and a missing platform mutation
      input both return nonzero and create no output directory or commitment. The design
      therefore treats unavailable, partial, and early-failure attempts as invalid and
      uncommitted rather than permitting an author-selected blocked scientific outcome.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolution is by removing terminal blocked=true from this public protocol, not by
      adding a typed blockage outcome. A future reintroduction of blockage would require
      the original causal-evidence controls.

  - finding_id: "VIA000-R2-PA1-ASSEMBLY-001"
    outcome: unresolved
    evidence: |-
      The new assembler deterministically rebases paths and handles the tested missing
      and nonzero cases atomically, but it validates only the raw JSON Schema before
      writing raw-results.json and output-commitment.json. In an independent execution,
      each platform input contained a 119,015-byte invalid pseudo-PDF with all input
      hashes and summary hashes updated. The assembler returned 0, created the output
      directory, and created the commitment. Thus malformed semantic evidence can be
      committed before the authoritative helper runs. The emitted commitment fields
      are raw_results_path/raw_results_sha256 at assembler lines 313-314, whereas packet
      custody and campaign validation require output_receipt_id/output_sha256. The
      public round-trip test asserts only raw_results_sha256 and never exercises a full
      commitment-to-custody/reveal path.

      Exact calibration run 32420571846 also did not produce successful fragments.
      Windows job 96591479733 and Ubuntu job 96591479930 each passed 366 tests, all six
      generators, artifact checks, and both PDF passes, then failed at runner line 287:
      `candidate repository has tracked, untracked, or ignored residue after execution`.
      The runner checks final status before copying and restoring the declared generated
      paths. Artifact uploads include the entire temporary environment and report
      712,758,054 Windows bytes and 17,876,913,630 Ubuntu bytes. No clean two-platform
      runner-to-assembler-to-custody round trip exists at this handoff.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The frozen assembler is a real improvement, but schema-valid assembly is not the
      same as semantically valid, custody-compatible, successfully produced evidence.

  - finding_id: "VIA000-R2-PA1-FREEZE-001"
    outcome: verified-resolved
    evidence: |-
      Every one of the ten PROTOCOL_MANIFEST contract-file SHA-256 values equals the
      raw Git-blob SHA-256 at exact handoff e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a,
      including validator digest
      eedc4b7add380c9ea1d64174dab7ba4e289cc2dd8bf92065e768e50acd61ff70.
      The requirements digest also matches. The declared and independently recomputed
      VIA-000 packet-rule digest both equal
      cf1a27bcffd1702fc7478377489a0dcfb2772028f282655590ae69adedcefa10.
      Primary protocol, runner, raw schema, and assembler receipt copies are byte-identical
      to their frozen sources.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This verifies exact byte binding at the reviewed handoff only; later remediation
      must recompute and re-review its new manifest bindings.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    outcome: unresolved
    evidence: |-
      The minimal summary-only document and missing artifact map reject, but a coherent
      hash-closed zero-execution package with all expected typed roles and self-authored
      records passes the raw helper and full campaign validator. The invalid framed
      pseudo-PDF also passes and can be committed by the assembler. The public positive
      fixture constructs evidence rather than executing the frozen runner, so no real
      successful runner-to-raw package was demonstrated at this handoff.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Required remaining controls are coherent no-execution records, independently
      parsed PDF pages/engine, substantive mutation-oracle evidence, and a real positive
      producer-to-validator round trip.

  - requested_test_id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    outcome: verified-satisfied
    evidence: |-
      Top-level and platform commit/tree, parameter list, contract platform list,
      schema required platform list, command mapping, and command count were mutated
      independently. Every mutation produced a validity error, including when summary
      outcomes could otherwise be made self-consistent.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The exact platform and command identity cross-bindings now fail closed."

  - requested_test_id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    outcome: verified-satisfied
    evidence: |-
      blocked=true rejects at schema and semantic layers. Nonzero and missing-input
      assembly attempts return nonzero and create neither an output directory nor a
      commitment. Because the frozen R2 protocol admits no blocked=true terminal result,
      partial or unavailable attempts cannot be converted into a scientific outcome.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Typed genuine blockage is intentionally absent. If later added, absent/dummy/
      malformed receipt and cause-code tests become required again.

  - requested_test_id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    outcome: unresolved
    evidence: |-
      The synthetic positive and two public negative cases pass locally, but the
      assembler committed an invalid pseudo-PDF package; its commitment is incompatible
      with packet custody field names; and both exact hosted platform producers failed
      their final status boundary. There is no successful real two-platform fragment,
      thirty-six-mutation, assembler, authoritative semantic validator, output-
      commitment, custody, and reveal round trip. Missing/changed/duplicate/unsafe path,
      nonzero/partial, identity, mutation, PDF, artifact, atomicity, and custody cases
      therefore are not closed end to end.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      The next round must validate semantics inside the frozen assembler before atomic
      output and use the exact emitted commitment through the public campaign chain.

  - requested_test_id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    outcome: verified-satisfied
    evidence: |-
      Exact raw Git-blob SHA-256 recomputation matched all ten contract-file entries,
      the requirements entry, and the VIA-000 packet-rule digest at e2ea7ec. All four
      protocol receipt copies matched their source bytes.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Any later protocol change requires a new exact-SHA all-entry audit."

predictions:
  experiment_id: "TST-VIA000-R2-PA2-END-TO-END-FALSIFIER-001"
  predicted_outcome: |-
    At exact handoff e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a, repeated coherent
    self-authored evidence, invalid pseudo-PDF, and opposed cross-platform raster
    controls will continue to pass or produce a commitment, while both hosted platform
    runners will continue to fail after generation. Identity, blocked=true, and exact
    manifest-binding controls will continue to reject or match as described.
  predicted_failure_mode: |-
    The protocol attributes execution and cross-platform meaning to producer-authored
    records, candidate-relative checks, and schema-valid assembly; generated candidate
    drift then prevents the real runner from reaching its evidence-copy boundary.
  confidence_statement: |-
    High confidence for this public pre-holdout prediction because the coherent package
    passed the full campaign validator, the pseudo-PDF and opposed-drift attacks were
    executed independently, the assembler emitted an invalid commitment, and both exact-
    SHA hosted operating-system jobs reproduced the runner failure. No prediction is
    made about hidden evidence, a private holdout, or POPGP's physical mechanism.

recommendation:
  approve: false
  blocking_findings: 4
  rationale: |-
    Changes requested. VIA000-R2-PA1-EVIDENCE-001 and
    VIA000-R2-PA1-ASSEMBLY-001 remain unresolved; VIA000-R2-PA2-VISUAL-XPLAT-001
    and VIA000-R2-PA2-CI-HISTORY-001 are new blockers. Identity, blockage, and freeze
    are independently verified resolved, and their three requested tests are satisfied.
    The reviewed handoff still accepts coherent zero-execution evidence, commits an
    invalid pseudo-PDF under a custody-incompatible document, accepts pairwise visual
    disagreement beyond the registered bound, fails both frozen hosted producers after
    generation, and has red ordinary CI. It is not safe to refreeze or merge and grants
    no permission to access custody material, begin holdout execution, reveal, advance
    lifecycle, or assign a scientific outcome.
```

## Re-review method and executed evidence

The reviewer created `C:\src\POPGP-via000-r2-protocol-rereview-1` directly at the
immutable handoff and used branch
`review/via000-r2-protocol-amendment-rereview-1`. The original review was read from
its immutable commit rather than a mutable working-tree copy. Temporary evidence was
created only under operating-system temporary directories. Test-created ignored
environments and caches were removed by exact path after execution.

| Command or probe | Observed result |
|---|---|
| `git rev-parse HEAD`; `git rev-parse HEAD^{tree}` | Exact commit `e2ea7ec2fc087c100e35c3f9fea6b39a80cc693a`; tree `145d59bf5f215ae8a4b9ac949bdb86094a563ab6`. |
| Ancestry and immutable `commit:path` checks | Original review commit `755a6a4d20bac2f4ca35432a98027918a4972d94` and implementation fix `3876cc0738b5474f49e10468b1e1e675fa6fcba1` are ancestors; prior review and response parse without duplicate keys and conform to their v2 schemas. |
| `uv run --frozen --no-editable pytest -q tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_via000_r2_assembler.py` | `6 passed in 206.53s`. The positive is synthetic; its limitations are reproduced above. |
| Coherent zero-execution fixture through `_validate_raw_evidence_contract` and temporary full `validate_campaign` | Both returned zero errors. No custody or hidden material was read. |
| Invalid pseudo-PDF through raw validator | Zero errors for 119,015 structurally invalid bytes with updated hashes. |
| Invalid pseudo-PDF through frozen assembler | Return code 0; output directory and `output-commitment.json` both created. |
| Opposed visual drift at `[24,424,0]` | Candidate-relative deltas 4 and 4; direct platform delta 8; raw validator returned zero errors. |
| Independent identity/contract mutations | Top/platform commit/tree, parameter/contract/schema platform sets, command map/count all rejected. |
| blocked=true, nonzero fragment, missing mutation input | blocked=true rejected; assembler negatives returned nonzero with no output/commitment. |
| All-entry manifest audit via `--git-blob-sha256` and packet-rule recomputation | Ten contract files, requirements, and VIA-000 rule all matched exact handoff bytes. Protocol and receipt copies matched. |
| JSON duplicate-key parse, v2 schemas, PowerShell AST parse, Ruff on changed Python, `git diff --check` | Passed at the reviewed handoff. |
| Public CI run `32420571769` | Exact head SHA; failure: six errors and 366 passes because historical candidate is absent from shallow checkout. |
| Public calibration run `32420571846` | Exact head SHA; Windows and Ubuntu execute steps failed at final repository-residue check after 366 tests, generators, artifact checks, and two PDF passes. |

The failed calibration artifacts were not downloaded in full. GitHub artifact metadata
reports Windows artifact `9426082298`, 712,758,054 bytes, digest
`sha256:0ead8e83d4d0fb702d611b166850bc27fdba113085181b9d8aafcbd5ee4c61ef`, and Ubuntu
artifact `9426398597`, 17,876,913,630 bytes, digest
`sha256:148d3e0e4723e6abb8147322e4677dba69db143da08bc80962272fc02bfc9a84`. The workflow
uploads the complete platform temporary root, including its environment/cache, rather
than a bounded immutable fragment. Because both producers failed before their clean
fragment boundary and the artifacts are respectively about 0.7 GB and 17.9 GB, they
cannot serve as the requested clean two-platform positive. Their exact logs and
metadata were sufficient to reproduce and locate the failure; their content was not
represented as successful scientific evidence.

No local Ubuntu host, `nvcc`, or CUDA/Blackwell protocol path was available or required
by the frozen CPU-only platform contract. Hosted TeX Live 2026 produced an eleven-page
PDF twice on both platforms before the runner failure. The independent pseudo-PDF
attack establishes that the validator does not derive those properties from retained
PDF bytes; it does not dispute that the hosted TeX command itself completed.
