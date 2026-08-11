# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5"
response_round: 5
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md"
review_commit: "1647e4f118a773256edb838e1d4f61bd853d5548"
candidate_commit_reviewed: "dccccfb685c2181192896c70c225b4076e8d8eca"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  The remaining VPLAN-INDEPENDENCE-001 blocker and both active requested tests were
  accepted and implemented in 35a8b9979e18b5147187ec7f16ec6385d89ba64e.
  Tier-E comparison now uses exactly the candidate raw result protected by blind
  custody and its output commitment. Candidate and external receipt IDs, resolved
  paths, and SHA-256 bytes must be distinct. A typed post-reveal adjudicator receipt
  binds both output hashes, the preregistered JSON metric pointer and absolute
  tolerance, both measured values, and the validator-recomputed agreement result.

  External implementation commit/tree identities must resolve from a retained,
  SHA-256-addressed Git bundle. Canonical repository comparison rejects common URL,
  transport, credential, and `.git` aliases of the candidate repository; candidate
  commit/tree reuse and mismatched or nonexistent bundle objects are rejected. The
  external operator now records an orchestrator that must differ from every internal
  seat. Packet freeze v4 binds the expanded contract. A real external disagreement is
  retained as valid failed evidence with `external-replication-disagreed`; it cannot be
  silently discarded or promoted. These checks establish contract consistency only,
  not the real-world truth of unaffiliated authorship or a POPGP viability result.

finding_responses:
  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The comparison candidate ID/hash must equal
      blind_custody.output_commitment.output_receipt_id/output_sha256. Candidate and
      external receipts must have distinct IDs, resolved paths, and hashes. The
      adjudicator comparison receipt is parsed as typed JSON and checked against the
      actual two receipt documents; agreement is recomputed from the frozen metric JSON
      pointer and nonnegative finite absolute tolerance. False agreement requires a
      failed packet with the external-replication-disagreed cause.

      External provenance now includes a content-addressed Git bundle receipt. The
      validator clones it without checkout in a disposable directory, verifies that
      the declared commit exists and resolves to the declared tree, and rejects
      candidate commit/tree reuse. Repository identities are canonicalized across
      common URL/path aliases, and external orchestrator identity is reconciled against
      all campaign seats. Malformed URLs/numeric extremes fail closed. Packet freeze v4
      prevents silent migration of older preregistrations.
    changed_files:
      - "schemas/viability/packet-v2.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
      - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["35a8b9979e18b5147187ec7f16ec6385d89ba64e"]
    verification:
      - command: "uv run pytest -q"
        result: "exit 0 on the final implementation; 176 passed in 302.41 s, including all 17 campaign-contract tests"
      - command: "uv run pytest tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison tests/unit/test_review_guidance.py -q"
        result: "exit 0 on the final implementation; 10 passed in 118.13 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py && uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; lint, 652-line manuscript source, structured artifacts, and required visuals passed"
      - command: "six documented python -m examples.physics_qg.* commands"
        result: "all six exited 0; regeneration left no tracked or untracked artifact diff"
    residual_risk: "A valid Git bundle and internally consistent identities cannot prove unaffiliated real-world authorship, exposure history, or institutional custody. Tier E still requires an external maintainer to verify those off-system facts."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_requires_typed_unaffiliated_clean_room"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 17 passed, with the original same-identity/copied-core/generic-receipt matrix retained and expanded"
    rationale: "The originally requested clean-room positive remains valid, while same internal identities, copied-core declarations, generic or malformed receipts, missing/mismatched Git provenance, substituted outputs, and dishonest comparison results are rejected."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
    verification:
      - command: "uv run pytest tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison -q"
        result: "exit 0; honestly frozen unrelated-candidate and identical-byte/path substitutions, repository alias/nonexistent commit, wrong tree, reused orchestrator, false receipt agreement, and valid disagreement cases behaved as required"
    rationale: "The narrowed reviewer counterexamples are persisted directly: output commitment identity, distinct bytes/paths, canonical repository identity, resolvable bundle commit/tree, orchestrator separation, and validator-derived comparison agreement are all executable gates."
    disagreement_ref: ""

new_or_changed_risks:
  - "Packet freeze popgp-packet-freeze-v4 intentionally invalidates earlier draft packet-rule hashes; affected campaigns must be preregistered again before holdout execution."
  - "Git-bundle cloning is bounded and does not check out files, but campaign operators must still apply storage/resource limits before accepting large untrusted evidence packages."
  - "The comparison contract currently implements one typed scalar JSON-pointer/absolute-tolerance computation; campaigns requiring vector, distributional, or multi-observable comparison need a new versioned method rather than an ad hoc field."
  - "No POPGP Tier R, G, or E campaign or external empirical validation was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, have an unaffiliated custodian verify organization, exposure, authorship, and retained Git-bundle provenance off-system."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-INDEPENDENCE-001, TST-VPLAN-INDEPENDENCE-001, TST-VPLAN-INDEPENDENCE-002, every prior finding/test for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
