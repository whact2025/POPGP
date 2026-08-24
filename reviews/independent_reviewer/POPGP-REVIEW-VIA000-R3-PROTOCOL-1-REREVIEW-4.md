# VIA-000 R3 recovery-protocol independent re-review 4

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-4"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "9fb95a12047d9d038cb1f9b62caa3a2abcc97e2d"
baseline_commit: "8c2d4c4c4d037bb57b0a462c86db34546e972925"
prior_review_ref: "8c2d4c4c4d037bb57b0a462c86db34546e972925:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3.md"
builder_response_ref: "9fb95a12047d9d038cb1f9b62caa3a2abcc97e2d:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3-RESPONSE-1.md"
context_hash: "e1f3f8a06d2f63d9ef7ccc2049a6c522b7a1c12d"
context_hash_method: 'git rev-parse "9fb95a12047d9d038cb1f9b62caa3a2abcc97e2d^{tree}"'
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
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3.md"
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
access_level: "public-repository-only plus local Windows execution and public official runner/action documentation; no custody, reveal, external invalid-package, or handoff-memo access"
independence_statement: |-
  This was a fresh independent re-review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-4 on dedicated branch
  review/via000-r3-protocol-rereview-4. Exact handoff/content commits and trees,
  parentage, complete review/response history, and byte immutability were verified
  before review. Builder tests and response claims were treated as hypotheses. The
  reviewer changed no implementation, protocol, packet, campaign, signer, ref,
  lifecycle, evidence, commitment, custody, threshold, result, or reveal material.

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
  CHANGES REQUESTED with one new critical blocker. The REREVIEW-3 remediation closes
  its stated first-shell boundary: every GitHub expression consumed by a PowerShell
  `run:` step crosses through step environment data, the exact authorization grammar
  is checked before the authorization step touches Git or files, array arguments keep
  values inert, and the initial Git/SSH paths are literal. The exact reviewer payload
  plus an independent CR/CRLF, `${}`, subexpression, backtick, statement, pipe,
  ampersand, Unicode control, YAML-scalar, environment-file, and option-like corpus
  all reject with no authorization-step temporary residue. No `run:` source contains
  `${{`, `Get-Command`, or `Invoke-Expression`. Attestation/output values also cross
  shell boundaries as data. The 35-case focused suite is green. All six prior
  findings remain resolved and all six prior requested tests remain satisfied in
  their original scopes.

  The claimed literal-tool boundary stops before the actual experiment, however.
  The unmodified production PowerShell runner still invokes `uv`, `pdflatex`, `git`,
  and `python` by bare PATH name throughout cloning, source extraction, environment
  creation, validation, test execution, and evidence generation. The mutation runner
  likewise launches bare `uv` and `python`. An exact production-runner probe placed
  command shims first in PATH: forged `uv 0.11.11` output passed the version gate,
  both the uv and pdflatex shims executed, and the runner created a partial evidence
  workspace before rejection. The builder's PATH test extracts only the initial
  authorization step and cannot see this downstream boundary.

  The initial step also accepts any absolute leaf as the setup-action Python output.
  In a real Git repository with a grammar-valid authorization ref, substituting an
  absolute `.cmd` for `VIA000_BASE_PYTHON` executed the marker and returned overall
  step success without any authorization JSON. A malicious program can instead write
  forged authorization state and alter the checkout. Literal Git/SSH locations are
  not byte/hash/reparse bound, while the `*-latest` hosted images update in place.
  Thus tool location and a version string do not bind the executable bytes that make
  or attest the evidence, and the always-upload step can retain attacker-created
  partial evidence after a later failure.

  R2 remains immutable. R3 remains drafted, holdout_started=false, unrevealed, and
  pending. The comment-only signer blocks authorization; no production key, R3 tag,
  execution, raw result, commitment, custody transition, or reveal was created or
  inspected. Public commitment hashes still require fresh private fourteen-of-
  fourteen custodian verification. This review authorizes neither merge nor a signer-
  key amendment, tag, refreeze, activation, preregistration, holdout, assembly,
  commitment, reveal, or scientific claim.
findings:
  - id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    severity: critical
    category: code
    location: ".github/workflows/via000-r3-protocol.yml:34-54,77-94,117-129,139-146,148-205,239-298,331-361; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1:45-49,123-220,247-365; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py:107-143; tests/unit/test_via000_r3_identity.py:1187-1322"
    evidence: |-
      The workflow's literal `$trustedGit` is used only for three authorization-ref
      equality gates. It is not passed to `VIA-000-RUNNER.ps1`. That runner invokes
      `uv --version`, `pdflatex --version`, retained commands with `FilePath` equal to
      bare `git`, `uv`, or `python`, and numerous direct `& git`/`& python` calls.
      `ProcessStartInfo.UseShellExecute=false` does not turn a bare filename into a
      content identity; executable lookup still uses process PATH. The mutation runner
      constructs `['uv', 'run', ..., 'python', ...]` in a copied inherited environment.
      No pre-execution hash, canonical path, link/reparse ancestry, ACL, image version,
      or immutable tool receipt is compared with frozen protocol authority.

      The reviewer invoked the exact unmodified production runner with a fresh valid-
      grammar identity and PATH-prepended Windows command shims. A fake `uv.cmd` wrote
      `UV_FORGED_VERSION_ACCEPTED`, printed the exact expected `uv 0.11.11`, and exited
      zero. The runner accepted it and proceeded to a fake `pdflatex.cmd`, which wrote
      `PDFLATEX_PATH_SHIM_EXECUTED`. The run rejected only at that later shim and
      `workspace_created` was true. A second fake uv returning 99 wrote
      `RUNNER_UV_PATH_SHIM_EXECUTED` before the version error. Arbitrary code therefore
      runs before source clone, frozen-boundary extraction, environment manifest, or
      evidence identity exists; a correct-looking version string is not a defense.

      The builder test named `...ignores_path_tool_shims` runs only the extracted
      `Reject non-snapshot lifecycle refs` step. It never invokes the frozen platform
      runner or mutation runner, so its three PATH layouts cannot support the response
      claim that PATH is not consulted for Git or Python during protocol execution.

      A separate exact authorization-step probe created a real repository and
      grammar-valid authorization ref, then set `VIA000_BASE_PYTHON` to an absolute
      existing `.cmd`. `IsPathFullyQualified` and `Test-Path -PathType Leaf` both
      passed. The substituted program executed, the step exited zero, and no
      authorization JSON existed. The workflow checks neither the setup action's
      `python-version` output nor path root, extension/type, reparse ancestry, file
      digest, or a frozen tool receipt before execution.

      Current Windows literal Git and ssh-keygen paths exist and were regular
      non-reparse entries under administrator/system-controlled directories, but
      their observed bytes and versions are absent from protocol authority. Official
      GitHub runner-image documentation states that `ubuntu-latest`/`windows-latest`
      labels migrate and images receive typically weekly software updates. The current
      Ubuntu manifest confirms Git/OpenSSH packages but not immutable executable
      bytes, and its cached CPython list does not contain requested 3.11.15. These are
      mutable platform dependencies, not content-addressed protocol inputs.
    finding: |-
      The workflow hardens only the pre-guard ref query. Actual experiment and mutation
      execution still trusts mutable PATH names and an unbound setup-action path, so
      executable location/version text can be substituted without changing any
      frozen protocol hash.
    failure_scenario: |-
      A contaminated hosted image, action-modified environment, PATH/PATHEXT shim, or
      substituted setup-python output supplies a program that prints the expected
      version and returns success. It then replaces Git/Python/uv behavior, writes a
      forged authorization JSON or evidence tree, modifies the checkout, or invokes
      another program. The pinned workflow subsequently attests and always uploads
      the attacker's subjects under the expected source/run identity.
    consequence: |-
      Authentication, exact-source execution, mutation evidence, environment/source
      manifests, attestation subjects, and cleanup are not functions solely of frozen
      bytes. Arbitrary unbound executable code can run before the retained evidence
      boundary and can leave or upload partial/forged evidence after rejection.
    required_action: |-
      Define the hosted runner/shell bootstrap trust assumption explicitly, then bind
      every additional executable used by the authorization, clean runner, mutation
      runner, validation, TeX, and attestation preparation paths. Pass canonical
      absolute Git, ssh-keygen, base Python, uv, pdflatex, environment Python, and
      PowerShell paths as explicit single arguments; never launch bare tool names or
      inherit a search PATH in the production/mutation subprocesses. Before creating
      the workspace or reading authorization objects, reject unexpected platform/
      image version, setup-python version/path, non-regular or reparse/junction/symlink
      ancestry, and tool digests against separately reviewed platform-specific frozen
      receipts. Move temporary/output creation after those checks and delete any
      partial tree on failure; ensure `if: always()` cannot upload a failed or
      unauthenticated tree. Add real end-to-end runner and mutation controls for PATH,
      PATHEXT, multiple installations, absolute-output substitution, same-path byte
      replacement, reparse paths, forged version output, and cleanup on both supported
      hosted images.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    description: |-
      Execute the exact workflow authorization step, unmodified platform runner, and
      unmodified mutation runner on Windows and Ubuntu with zero/one/multiple PATH and
      PATHEXT shims for git, ssh-keygen, Python, uv, pdflatex, and PowerShell; shims
      must print the expected version and attempt markers, forged authorization JSON,
      checkout/evidence mutation, and self-cleanup. Substitute the setup-python path
      with absolute command/script files, wrong tool-cache roots, symlinks/junctions/
      reparse ancestry, same-path changed bytes, and multiple installations. Each
      substitution must reject before any untrusted executable runs, workspace or
      authorization/output file is created, attestation is requested, or artifact is
      uploaded. The exact reviewed platform receipts and signed happy paths must pass
      using only explicit absolute content-bound tools. Retain the full hostile input,
      ref race, replacement, validator closure, cross-run, LF, and cleanup controls.
    rationale: |-
      Testing only the first shell step proves input inertness but says nothing about
      the executables that clone, extract, run, validate, mutate, or generate attested
      evidence. Path and version strings are not executable-byte identities.
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: "The separately signed canonical record still binds snapshot, authorization commit, campaign, packet, and manifest; lifecycle/ref/record/signer substitutions reject inside the authenticated Python boundary."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new tool-execution boundary can bypass that correct decision by changing the program that performs or consumes it."
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: "Manifest-hash-verified Git-object extraction, real packet validation, dependency/worktree substitution, sparse/repo-root, cross-run, and zero-commitment controls remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding precedes the frozen validator closure and concerns its launching tools."
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: "Git-normalized workflow/protocol/runner hashes, receipt copies, validator closure hashes, autocrlf true/false fixtures, and the exact signed Python path remain reconciled on Windows."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Executable identity is distinct from LF-normalized source identity."
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    outcome: verified-resolved
    evidence: "All capture/parse/peel/signature/final-check ref swaps, move/delete races, assembler post-capture movement, loose/packed shadowing, abbreviated IDs, and nested tags reject; direct captured-object passes."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No mutable-ref regression was found."
  - finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    outcome: verified-resolved
    evidence: "Default/custom replacement refs, authorization/source/campaign/packet/manifest/validator replacements, hostile Git environment/config/program controls, and guard/assembler cleanup remain green in the focused suite."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding is bare executable lookup in the runner, not Git replacement inside hardened guard/assembler/validator calls."
  - finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-resolved
    evidence: |-
      No PowerShell run source contains GitHub expression interpolation. The original
      quote-close payload and an expanded hostile data corpus remain inert and reject
      before the authorization step's Git/file operations with no step residue. The
      exact signed initial step ignores PATH Git/ssh/Python name shims by selecting
      reviewed literal paths and setup-action output through argument arrays.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The new finding is a downstream platform/mutation tool boundary not exercised by the RR3 first-step test."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: "Signed authority, lifecycle, ref-kind, signer, record, target, packet, and no-commitment cases pass in the 35-case focused suite."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No authority-logic regression was found."
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: "Frozen verifier closure, real packet validation, dependency/worktree substitutions, and cleanup controls remain green."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No validator-closure regression was found."
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: "Git blob hashes, receipt hashes, LF normalization, and Windows autocrlf true/false paths remain reconciled."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Executable-byte identity requires a distinct test."
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    outcome: verified-satisfied
    evidence: "All captured-object race stages and the unmodified assembler post-capture race reject, while the exact immutable-object path passes."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No ref/OID TOCTOU regression was found."
  - requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    outcome: verified-satisfied
    evidence: "Real default/custom replacement namespaces, tag/source/blob substitutions, hostile Git controls, fixed verifier, cleanup, and direct signed cases all pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Bare downstream executable lookup was not part of the Git-object replacement test."
  - requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-satisfied
    evidence: |-
      The exact first production step rejects quote, statement, subexpression,
      variable, CR/LF, backtick, comment, whitespace, option-like, Unicode/control,
      YAML-scalar, environment-file, pipe, ampersand, and quote values as inert data.
      Its exact signed path ignores zero/one/multiple PATH name shims.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The new end-to-end test must continue through the platform and mutation runners."
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment was run or inspected; this artifact assesses only pre-holdout protocol and governance machinery."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    CHANGES REQUESTED. The initial manual-input shell boundary is fixed, but the tools
    that execute and mutate the candidate remain PATH/output/location trusted rather
    than content bound. Correct and independently re-review the complete execution-
    tool chain. This review authorizes no merge, signer amendment, tag, refreeze,
    activation, preregistration, custody transition, holdout, assembly, reveal, or
    scientific claim.
```

## Verification ledger

- Exact identity: handoff/tree `9fb95a12047d9d038cb1f9b62caa3a2abcc97e2d` / `e1f3f8a06d2f63d9ef7ccc2049a6c522b7a1c12d`; content/tree `aab992457139f22fd0c1279f54cb5139e0bad915` / `39271e06293153dc6ba2f04be2b80eebde7e0e3f`; prior rereview `8c2d4c4c4d037bb57b0a462c86db34546e972925`.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py` — exit 0; 35 passed in 417.26 seconds.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r2_assembler.py tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_review_guidance.py` — exit 0; 22 passed in 679.26 seconds.
- Full collection — exit 0; 414 tests collected in 44.74 seconds. The full suite was not executed after the decisive critical reproducer.
- Changed-Python Ruff — exit 0; all checks passed.
- `python scripts/check_tex.py` — exit 0; balanced and valid TeX source.
- `python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml` — exit 0; campaign contract valid.
- Expanded exact-step input probe — fourteen additional CR/CRLF, `${}`, subexpression, backtick, statement/operator, Unicode, YAML-scalar, environment-file, and option cases all rejected with zero authorization-step residue.
- Exact runner PATH probe — forged uv version accepted; uv/pdflatex markers executed and partial workspace existed. A failing uv marker also executed before rejection.
- Exact setup-output probe — an absolute existing command file substituted for base Python executed and returned step exit 0 without an authorization JSON.
- Current Windows literal Git and ssh-keygen files existed at the workflow paths and were not reparse points; their SHA-256 values were not present in frozen protocol authority.
- Official platform evidence consulted: `https://github.com/actions/runner-images`, `https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2404-Readme.md`, and exact pinned setup action source `https://github.com/actions/setup-python/blob/ece7cb06caefa5fff74198d8649806c4678c61a1/src/find-python.ts`.
- All four builder responses parsed against `review-response-v2.schema.json` with zero errors. Original/REREVIEW-1/2/3 reviews and the first three responses were byte-identical to their first committed versions; the new response was schema-valid at handoff.
- R2 scoped diff from `6e0e0c8ebaecef6d129c68666f113fbd47af4ce7` was empty. Local and origin R3 tag counts were zero. The comment-only signer failed with `exactly one frozen authorization signer is required`; all seven packets were drafted/holdout-false and the campaign was pending.
