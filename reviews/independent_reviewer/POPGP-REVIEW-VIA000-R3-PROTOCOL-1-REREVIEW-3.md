# VIA-000 R3 recovery-protocol independent re-review 3

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-3"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "2cf2ac2b3b2fe2a22673eee49cd2275048aa38c7"
baseline_commit: "766b147438e5e5ba8da5aa87cb20a721b79aeb62"
prior_review_ref: "766b147438e5e5ba8da5aa87cb20a721b79aeb62:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2.md"
builder_response_ref: "2cf2ac2b3b2fe2a22673eee49cd2275048aa38c7:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2-RESPONSE-1.md"
context_hash: "f0e99a541f838293837384c91362fda29922cd29"
context_hash_method: 'git rev-parse "2cf2ac2b3b2fe2a22673eee49cd2275048aa38c7^{tree}"'
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
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2.md"
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
  C:\src\POPGP-via000-r3-protocol-rereview-3 on dedicated branch
  review/via000-r3-protocol-rereview-3. Exact handoff/content commits and trees,
  parentage, review/response history, and byte immutability were verified before
  review. Builder tests and response claims were treated as hypotheses. The reviewer
  changed no implementation, protocol, packet, campaign, signer, ref, lifecycle,
  evidence, commitment, custody, threshold, result, or reveal material.

  The human operator and Codex Desktop orchestrator are shared with the builder.
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
  CHANGES REQUESTED with one new critical blocker. The replacement-object remediation
  correctly forces the frozen guard, unmodified assembler, validator, and their
  security-critical Git reads through full object IDs, `--no-replace-objects`, a
  scrubbed Git environment, and fixed Git/ssh-keygen binaries. The 33-case focused
  suite independently replays the default and custom replacement namespaces,
  authorization/source/campaign/packet/manifest/validator substitutions, hostile Git
  environment/config/program controls, direct signed happy path, prior ref races,
  cleanup, cross-run, and Windows autocrlf controls. All five prior findings remain
  resolved and all five prior requested tests remain satisfied in their scopes.

  The hosted workflow nevertheless inserts the caller-controlled
  `inputs.authorization_ref` expression directly into double-quoted PowerShell source
  before the guard. A reviewer-shaped input closed the quote, executed an injected
  statement, replaced the captured OID with forty `b` characters, reset
  `LASTEXITCODE`, and passed the workflow's OID check. The same input is interpolated
  again into the guard command. This is arbitrary code execution at the production
  authorization boundary and can modify the checkout, guard, signer, authorization
  output, or subsequent environment before any authenticated decision. In addition,
  the workflow obtains Git through PATH using `(Get-Command git ...).Source`; on this
  Windows seat that expression returned two paths and the exact invocation failed,
  and a hostile earlier PATH application would be selected before Git variables are
  scrubbed. These paths contradict the claimed fixed-binary hosted boundary.

  R2 remains immutable. R3 remains drafted, holdout_started=false, unrevealed, and
  pending. The comment-only signer blocks authorization; no production key, R3 tag,
  execution, raw output, commitment, custody transition, or reveal was created or
  inspected. Public commitment hashes still require fresh private fourteen-of-
  fourteen custodian verification. This review authorizes neither merge nor a signer-
  key amendment, tag, refreeze, activation, preregistration, holdout, assembly,
  commitment, reveal, or scientific claim.
findings:
  - id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    severity: critical
    category: code
    location: ".github/workflows/via000-r3-protocol.yml:4-9,45,56-69,93,161; tests/unit/test_via000_r3_identity.py:1187-1207"
    evidence: |-
      `authorization_ref` is an unrestricted required workflow_dispatch string.
      Workflow lines 56-57 render `${{ inputs.authorization_ref }}` directly inside a
      double-quoted PowerShell native-command argument before the dispatch guard, and
      lines 61-69 render it directly a second time into the guard command. The focused
      static workflow test positively requires this unsafe interpolation token rather
      than proving that the expression crosses into PowerShell only as data.

      The reviewer rendered the exact capture/check shape with input
      `"; Write-Output VIA000_AUTH_INPUT_CODE_EXECUTED;
      $capturedAuthorizationTagOid=("b"*40); $global:LASTEXITCODE=0; #`. PowerShell
      printed `VIA000_AUTH_INPUT_CODE_EXECUTED`, then
      `captured=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` and `exit=0`; the injected
      value passed the workflow's forty-lowercase-hex condition after Git itself had
      failed with `fatal: Needed a single revision`. This proves execution before the
      Python guard and before any authorization output exists.

      Separately, the exact line-45 discovery returned a `System.Object[]` with two
      applications on this supported Windows seat:
      `C:\Program Files\Git\cmd\git.exe` and the Codex bundled Git. Invoking
      `& $trustedGit --version` then failed because PowerShell combined both paths as
      one command name. `Get-Command` searches caller PATH before the workflow removes
      `GIT_*`, so a single attacker-controlled earlier application would instead be
      executed. The same PATH lookup is repeated before each platform runner.

      By contrast, source inspection and the 33-case suite confirm that the Python
      guard/assembler/validator remediation itself ignores default/custom replace
      refs and inherited
      object/alternate/repository/config/namespace/exec/SSH controls, and malicious
      PATH for Git and ssh-keygen. Invalid originals reject without authorization,
      temporary assembly, output, or commitment, while the exact signed path passes.
      SHA-256 repositories are outside this protocol's deliberately frozen forty-hex
      SHA-1 identity grammar; abbreviated and nested tag identities reject.
    finding: |-
      The production hosted workflow treats an untrusted manual-dispatch input as
      PowerShell program text and resolves Git from caller PATH. Authentication can
      therefore be bypassed or the job can fail before the hardened guard executes.
    failure_scenario: |-
      A dispatcher supplies a quote/statement/comment sequence as authorization_ref.
      Git failure is followed by injected PowerShell that changes the checkout or
      replaces the guard, writes a forged authorization JSON, or changes variables and
      exit status. The workflow then invokes attacker-chosen code or passes attacker-
      chosen values onward. Independently, a PATH-prepended Git application runs at
      the initial rev-parse before any trusted-program decision.
    consequence: |-
      Manual dispatch is a code-execution capability rather than a data-only request.
      Required equality among the authorization ref/OID, signed record, checkout,
      runner source, assembler identity, and evidence can be defeated outside the
      otherwise hardened Python boundaries, and a normal supported runner may fail
      merely because more than one Git application is discoverable.
    required_action: |-
      Pass every GitHub expression into the PowerShell step only through a step-level
      environment variable; never embed workflow expressions in `run:` program text.
      Validate the authorization-ref environment value against the exact canonical
      full-ref grammar before any use and pass the quoted environment value as one
      native-command/Python argument. Select exactly one reviewed absolute Git path by
      explicit platform branch plus `Test-Path`/file validation, not PATH or
      `Get-Command`; apply that choice consistently to all three workflow Git gates.
      Freeze the corrected workflow hash/receipt and add executable command-boundary
      controls before another independent re-review.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    description: |-
      Exercise the production workflow command boundary, or an exactly extracted
      script generated from it, with authorization inputs containing PowerShell quote,
      statement, subexpression, variable, newline, backtick, comment, whitespace, and
      option-like payloads. Prove every value remains one inert data argument and is
      rejected by the canonical full-ref grammar before Git/guard execution; no marker
      program, checkout mutation, authorization JSON, temporary output, runner,
      assembly, raw result, or commitment may occur. Prepend a fake Git executable to
      PATH and provide zero, one, and multiple discoverable Git applications on both
      supported platform branches. The workflow must use exactly its reviewed absolute
      system binary or fail closed before object access. Retain all replacement,
      ref-race, fixed ssh-keygen, packed/loose, cleanup, and signed happy-path controls.
    rationale: |-
      Python guard tests cannot protect code that GitHub renders and PowerShell parses
      before Python starts. This test must cover the actual hosted shell boundary and
      tool selection, not merely search the YAML for identity tokens.
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: |-
      The separately SSH-signed canonical record binds the exact snapshot,
      authorization commit, campaign, packet, and manifest. Later lifecycle, wrong
      branch/event/SHA/HEAD, tag kind/suffix/ref, signer, record, campaign, packet, and
      manifest substitutions reject inside the guard/assembler boundaries.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new workflow command-injection path runs before this correctly hardened Python authority check."
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: |-
      The assembler uses manifest-hash-verified Git-object extraction for the guard
      and validator closure, checks worktree bytes, supplies the real authorized
      packet, and validates before commitment. Dependency, repo-root, sparse content,
      identity, attestation, cross-run, and cleanup controls remain green.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding is outside and earlier than the unmodified assembler boundary."
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: |-
      Git-normalized hashes, receipt copies, validator closure hashes, autocrlf
      true/false fixtures, and the exact signed Python happy path remain reconciled on
      Windows. The workflow PATH-array failure is a distinct hosted bootstrap issue.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No line-ending regression was found."
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    outcome: verified-resolved
    evidence: |-
      Parse/peel/signature/final-check swaps, move/delete races, invalid-object ref
      swap, assembler post-capture movement, loose-over-packed shadowing, abbreviated
      IDs, and nested tags reject; the direct captured-object path passes.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The hosted interpolation bypass precedes object capture and does not reopen the internal ref/OID logic."
  - finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    outcome: verified-resolved
    evidence: |-
      The focused suite replayed the invalid-original/valid-replacement construction
      through the production guard and unmodified assembler with default refs/replace,
      custom GIT_REPLACE_REF_BASE, and replacements of source, campaign, packet,
      manifest, and validator blobs. Every invalid original rejected without residue;
      hostile inherited Git/object/config/program state was ignored and the direct
      signed original passed. Source inspection confirms fixed binaries, no-replace
      options, scrubbed Git environments, and command-line SSH verifier config across
      Python security boundaries.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding concerns the workflow's pre-guard PowerShell text and Git selection, not Git replacement in the remediated Python code."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: "The retained signed authority, lifecycle, ref-kind, signer, record, target, packet, and zero-output cases pass in the 33-case focused suite."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Workflow shell parsing requires a distinct requested test."
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: "Frozen verifier closure, real packet validation, dependency/worktree substitutions, and cleanup controls remain green."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No validator-closure regression was found."
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: "Git blob hashes, receipt hashes, LF normalization, and Windows autocrlf true/false signed paths remain reconciled."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The new tool-discovery defect is separate from blob normalization."
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    outcome: verified-satisfied
    evidence: "All captured-object race stages and the unmodified assembler post-capture race reject, while the exact immutable-object path passes."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No ref/OID TOCTOU regression was found."
  - requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    outcome: verified-satisfied
    evidence: |-
      Five real-Git replacement controls cover default/custom namespaces, guard and
      extracted assembler, source/campaign/packet/manifest/validator objects, hostile
      repository/worktree/object/alternate/config/exec/SSH/PATH state, cleanup, and
      the exact signed happy path. All pass in the 33-case suite.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The workflow command boundary was not exercised by those Python tests."
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment was run or inspected; this artifact assesses only pre-holdout protocol and governance machinery."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    CHANGES REQUESTED. Replacement-object isolation is fixed, but caller input is
    executable PowerShell and workflow Git remains PATH-selected before the hardened
    guard. Correct and independently re-review the hosted command boundary. This
    review authorizes no merge, signer amendment, tag, refreeze, activation,
    preregistration, custody transition, holdout, assembly, reveal, or scientific
    claim.
```

## Verification ledger

- Exact identity: handoff/tree `2cf2ac2b3b2fe2a22673eee49cd2275048aa38c7` / `f0e99a541f838293837384c91362fda29922cd29`; content/tree `47fb5477ac4d8a44b8e5edc22230f383583b5409` / `279d562964e7eb9048f4c60db8678487e52fc778`; prior rereview `766b147438e5e5ba8da5aa87cb20a721b79aeb62`.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py` — exit 0; 33 passed in 415.81 seconds.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r2_assembler.py tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_review_guidance.py` — exit 0; 22 passed in 713.85 seconds.
- Full collection — exit 0; 412 tests collected in 7.04 seconds. The full suite was not executed after the decisive critical reproducer.
- Changed-Python Ruff — exit 0; all checks passed.
- `python scripts/check_tex.py` — exit 0; balanced and valid TeX source.
- `python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml` — exit 0; campaign contract valid.
- This rereview and all three builder responses parsed against their v2 schemas with zero errors; all five prior findings/tests and the new finding/test reconcile exactly once.
- Workflow expression probe — direct authorization input interpolation executed `VIA000_AUTH_INPUT_CODE_EXECUTED`, replaced the captured OID with forty `b` characters, reset exit status to zero, and passed the workflow-shaped check after Git failure.
- Workflow Git-selection probe — `.Source` was an array of two application paths; the exact invocation failed. Selection occurs from PATH before Git-environment scrubbing.
- R2 scoped diff from `6e0e0c8ebaecef6d129c68666f113fbd47af4ce7` was empty. Original/REREVIEW-1/REREVIEW-2 reviews and the two earlier responses were byte-identical to their first committed versions; the new response was schema-valid at the handoff.
- Local and origin R3 tag counts were zero. The comment-only signer failed with `exactly one frozen authorization signer is required`; all seven packets were drafted/holdout-false and the campaign was pending. Hash/receipt checks in the focused suite and campaign validator passed without custody or reveal access.
