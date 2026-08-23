# VIA-000 R3 recovery-protocol independent re-review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "38ae35619ca8027c570d87868c48564dc109e7f6"
baseline_commit: "beba0f19ea7979c1676b45e2912b1ad6a2c3ba4f"
prior_review_ref: "beba0f19ea7979c1676b45e2912b1ad6a2c3ba4f:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
builder_response_ref: "38ae35619ca8027c570d87868c48564dc109e7f6:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-RESPONSE-1.md"
context_hash: "4f7cd8f2f05ed6e73fa4eb282f94ad03db6d7b8a"
context_hash_method: 'git rev-parse "38ae35619ca8027c570d87868c48564dc109e7f6^{tree}"'
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
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
  C:\src\POPGP-via000-r3-protocol-rereview-1 on dedicated branch
  review/via000-r3-protocol-rereview-1. HEAD, tree, content parent, prior-review
  commit, response artifact, ancestry, and artifact immutability were verified before
  review. Builder claims and tests were treated as hypotheses. The reviewer changed
  no implementation, protocol, packet, campaign, signer, ref, lifecycle, evidence,
  commitment, custody, threshold, result, or reveal material.

  The same human operator and Codex Desktop orchestrator are shared with the builder.
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
  CHANGES REQUESTED with one new critical blocker. The remediation genuinely closes
  the three prior defects in their original static scopes. A separately SSH-signed
  annotated authorization object now binds an immutable campaign, preregistered
  packet, manifest, and exact lightweight-tagged protocol snapshot. The assembler
  extracts its guard and complete declared local validator bundle from Git objects,
  verifies both primary-protocol and authorized-manifest hashes, supplies the real
  authorized packet, and creates the commitment only after isolated validation.
  Targeted LF attributes make the declared Git blobs and receipt copies exact under
  Windows core.autocrlf=true and false.

  The signed authorization check is nevertheless not single-object atomic. The guard
  captures and parses one tag object ID, but later peels and verifies the mutable ref
  name. A deterministic independent reproducer replaced the ref between those two
  phases. The guard accepted and returned invalidly signed parsed object
  1681f705972079d5b07326057ee021d78d4b7e36, while Git actually verified different
  validly signed object b8a446e73b62cf924478ea1dff99bb9f8d7d19eb with an unrelated
  message. Direct verification of the returned object exited 1. A ref swap can be
  repeated at hosted dispatch and assembly, so the returned authorization record and
  tag OID are not the bytes authenticated by the frozen signer.

  Identity/history, response schema, R2 immutability, seven execution-artifact hashes
  and receipt copies, validator-manifest hashes, campaign validation, Ruff, TeX, and
  the 23-case R3 suite are green. The placeholder signer file contains no key, raises
  before authorization, and no local or remote R3 tag exists. All seven R3 packets
  remain drafted with holdout_started=false; the campaign remains pending. The public
  R2 commitment carry-forward remains conditioned on fresh 14/14 custodian
  verification. The complete suite was not rerun after the decisive critical
  reproducer; no broad positive result can change this fail-closed authorization
  blocker.

  This review authorizes neither merge nor a signer-key amendment. It authorizes no
  tag, snapshot/refreeze, activation, preregistration, custody transition, falsifier
  run, holdout, evidence assembly, commitment, reveal, or scientific outcome. After
  this blocker is fixed and independently re-reviewed, any approval must remain
  limited to the protocol bytes and a separately reviewed one-key signer amendment.
findings:
  - id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    severity: critical
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py:143-179,206,235-262,296-305; .github/workflows/via000-r3-protocol.yml:41-53; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py:160-195"
    evidence: |-
      `_tag_record` resolves `authorization_ref` once, reads and returns its
      `tag_oid`, canonical record, record-hash suffix, tag name, and target header.
      `verify_campaign_authorization` then discards that immutable identity for the
      two security-critical operations: line 235 peels
      `authorization_ref^{commit}`, and line 255 passes `authorization_ref` to
      `git verify-tag`. The function nevertheless returns the earlier `tag_oid`.

      The independent probe created a real temporary sparse Git repository, an
      Ed25519 allowed signer, a correct lightweight source tag, and two annotated tag
      objects having the same authorization tag header name and target. Object
      1681f705972079d5b07326057ee021d78d4b7e36 contained the canonical record but had
      a deliberately corrupted signature. Object
      b8a446e73b62cf924478ea1dff99bb9f8d7d19eb was validly signed by the frozen key but
      carried `signed unrelated message`. The ref initially selected the invalid
      object and was changed to the valid unrelated object immediately after
      `_tag_record` returned. The real remaining guard accepted and printed:
      `ACCEPTED 1681f705972079d5b07326057ee021d78d4b7e36`,
      `REF_NOW b8a446e73b62cf924478ea1dff99bb9f8d7d19eb`, and
      `PARSED_TAG_VERIFY_EXIT 1`.

      Static controls were sound: a second signing key, noncanonical JSON, wrong
      campaign/packet/manifest hashes, lightweight/annotated kind substitutions,
      moved/deleted/wrong-suffix refs, later self-consistent source tags, wrong event,
      branch ref, SHA, HEAD, run/attempt, platform and cross-run fragments all reject.
      The defect is specifically the time-of-check/time-of-use split across two Git
      objects hidden behind the same mutable ref name.
    finding: |-
      The guard does not prove that the authorization tag object it parsed and returns
      is the object whose SSH signature and target Git verified. Its authorization
      decision can combine the unsigned record from one object with the signature and
      target of another object selected later by the same mutable ref.
    failure_scenario: |-
      A repository principal able to update the authorization tag moves it after the
      workflow guard parses the tag but before target peeling/signature verification.
      The same controlled swap is repeated during assembly. Both calls return and
      propagate the invalid parsed tag OID and record even though the frozen signer
      authenticated different bytes. Attestations faithfully retain the wrong
      composite identity, and otherwise valid fragments can reach assembly.
    consequence: |-
      The required equality among authorization record, tag object, signed bytes,
      target commit, workflow identity, fragments, assembler identity, and retained
      evidence is falsified. Authorization is not fail closed under the explicitly
      in-scope moved/substituted-tag threat, so production activation or merge would
      rely on a signature that may not cover the returned record.
    required_action: |-
      After resolving the ref once, use only the captured full `tag_oid` for every
      subsequent operation. Parse its target header (or peel
      `<tag_oid>^{commit}`), compare that exact target with the record, and invoke
      `git verify-tag --raw <tag_oid>`. Never peel or verify the mutable ref again.
      Return the same verified object ID. Add a deterministic real-Git regression
      that swaps the ref after parsing and proves rejection, zero authorization JSON,
      zero platform execution, zero temporary assembly, and zero output commitment.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    description: |-
      In a real temporary Git repository with the frozen Ed25519 trust root, create
      one authorization tag object containing a canonical record and invalid
      signature and another same-name/same-target tag object with a valid signature
      over different message bytes. Move the ref after the first object is parsed.
      Exercise the hosted guard and the extracted assembler guard without replacing
      their decision logic. Require that target peeling and `git verify-tag` use the
      captured object ID, that the invalid parsed object rejects, and that no
      authorization JSON, runner/mutation step, temporary assembly, raw result, or
      commitment survives. Retain controls proving the unmoved exact signed object
      passes and static moved/deleted/wrong-kind/wrong-suffix cases reject.
    rationale: |-
      Existing tests mutate a ref before entry. They cannot detect a ref changing
      between parse, peel, and signature verification, which is the reproduced
      object-substitution gap.
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: |-
      Real disposable repositories and keys confirmed that the separately signed
      canonical record independently binds the exact source snapshot, authorization
      commit, campaign, preregistered packet, and manifest. The exact path passes.
      A later correctly content-addressed lifecycle commit, branch/event/SHA/HEAD
      substitutions, static moved/deleted/wrong-suffix/wrong-kind tags, noncanonical
      records, wrong campaign/packet/manifest hashes, and a second signing key reject.
      The current placeholder signer rejects before production authorization and no
      production key or tag exists.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The newly reported race is distinct: the static authorization construction is
      sound, but its implementation must remain on one immutable tag object after
      initial ref resolution.
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: |-
      The assembler extracts the guard and declared validator closure from the exact
      source Git object. It checks each dependency against both the frozen primary
      protocol and the authorized manifest, checks worktree bytes, writes only those
      Git blobs into a temporary bundle, and invokes it in isolated Python with the
      real authorized packet before commitment. Mutating each of the four nonempty
      local dependencies, the package initializer, or removing a sparse dependency
      rejected; wrong repo-root and protocol/schema substitutions rejected; no output
      survived. The invalid-real-packet probe reached the frozen validator and failed
      on raw-results semantics.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Installed third-party packages and the pinned base interpreter remain the
      explicitly declared environment boundary. File-symlink creation was unavailable
      on this Windows seat (WinError 1314), but path resolution precedes containment
      and byte checks and the existing safe-source controls remain green.
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: |-
      Fresh clone fixtures under core.autocrlf=true and false both materialized the
      workflow byte-for-byte equal to its LF Git blob. All seven declared execution
      artifacts matched their source-commit SHA-256 values and receipt copies; every
      validator dependency matched both its primary-contract and protocol-manifest
      SHA-256. The real SSH-signed exact-snapshot happy path passed on this Windows
      reviewer seat.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "A separate Linux reviewer execution remains desirable but is not needed to reproduce or fix the new platform-independent blocker."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: |-
      The 23-case identity suite and independent real-Git probes exercised the exact
      signed happy path, self-consistent later commit, wrong event/ref/SHA/HEAD,
      annotated source, lightweight authorization, deleted/moved/wrong-suffix
      authorization, canonical-record/hash and signer-key substitutions, cross-run
      identity, and zero-output lifecycle rejection. Those static cases behave as
      requested. The newly requested race control covers a distinct inter-call ref
      mutation absent from the original matrix.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The original static authority matrix is satisfied; a new blocking race test is required."
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: |-
      Worktree substitution of every declared nonempty local dependency rejected 4/4;
      an independent package-initializer mutation and missing sparse dependency also
      rejected. Primary and manifest hashes were recomputed from Git objects. The
      isolated bundle ran against the real authorized packet and rejected invalid raw
      evidence. Repo-root, protocol/schema, identity, attestation, platform, run,
      attempt, and cross-run mismatch controls fail before output; temporary cleanup
      checks found zero assembled or hidden temporary outputs.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Windows file-symlink creation required unavailable privilege; containment is enforced on resolved paths before reading or copying."
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: |-
      Both autocrlf clone fixtures passed, the workflow blob contains no CRLF, its
      attribute resolves to eol=lf, source/receipt Git bytes are exact for all seven
      execution artifacts, and validator dependency hashes agree with both frozen
      authorities. The exact signed snapshot and all focused identity negatives pass
      on Windows. Ruff, TeX, review guidance, campaign validation, and compatibility
      tests were separately exercised.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The complete 402-case suite was not rerun after the decisive critical authorization reproducer; focused and compatibility results are reported explicitly."
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment was run or inspected; this artifact assesses only pre-holdout protocol and governance machinery."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    CHANGES REQUESTED. The prior remediation is materially successful, but the
    reproduced authorization-object substitution violates the central fail-closed
    identity invariant. Fix and independently re-review exact-object peel/signature
    verification before merge. Even a later approval would authorize only the
    reviewed protocol bytes plus a separately reviewed signer-key amendment; it would
    not authorize a tag, refreeze, activation, preregistration, custody transition,
    holdout, assembly, reveal, or scientific claim.
```

## Verification ledger

- Exact identity: handoff/tree `38ae35619ca8027c570d87868c48564dc109e7f6` / `4f7cd8f2f05ed6e73fa4eb282f94ad03db6d7b8a`; content/tree `f00115330166a77ccc9630b4773dc21700b76377` / `d678d22afc38615a42a366bc57f9769059d03361`; prior review/tree `beba0f19ea7979c1676b45e2912b1ad6a2c3ba4f` / `2f342a9e69916d610c28783876de3d9ca8dc0c83`.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py` — exit 0; 23 passed in 264.89 seconds.
- `python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r2_assembler.py tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_review_guidance.py` — exit 0; 22 passed in 685.30 seconds.
- Changed-Python `ruff check` — exit 0; all checks passed.
- `python scripts/check_tex.py` — exit 0; balanced/valid source.
- `python scripts/check_viability_campaign.py reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml` — exit 0; campaign contract valid.
- Builder response parsed against `review-response-v2.schema.json` — zero errors; three finding responses and three requested-test responses reconcile one-for-one.
- Independent authorization substitutions — exact happy path accepted; wrong campaign, packet, and manifest digests, noncanonical JSON, and second signing key all rejected; sparse exact path accepted.
- Independent validator substitutions — package initializer and missing sparse dependency rejected with zero output; repo-root substitution rejected; file-symlink probe unavailable because Windows denied symlink creation.
- Local and origin R3 tag queries — zero tags. Comment-only signer probe — rejected with `exactly one frozen authorization signer is required`.
- The full 402-case suite — not rerun after the decisive critical blocker; no contrary positive evidence was inferred.
