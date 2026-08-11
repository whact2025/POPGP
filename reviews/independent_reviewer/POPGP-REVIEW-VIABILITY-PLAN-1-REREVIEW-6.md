# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_6"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "50df04b05e6991598a9ca0d0ffac45f8b0e0643e"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "e264b03babd09a428d1a1b236e6dbc6567cf55fa:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5.md"
builder_response_ref: "50df04b05e6991598a9ca0d0ffac45f8b0e0643e:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6.md"
context_hash: "8a0e7ce46e8ecf0853666f18bd69688bf6e633c2"
context_hash_method: "git rev-parse \"50df04b05e6991598a9ca0d0ffac45f8b0e0643e^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/DISAGREEMENT_LOG_TEMPLATE.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/DECISIONS.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6.md"
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..50df04b05e6991598a9ca0d0ffac45f8b0e0643e (complete 39-file diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in a dedicated worktree and session, but it
  remained under the same human operator and root Codex orchestrator as the builder.
  Builder and reviewer are both identified as OpenAI Codex GPT-5; no exact model
  snapshot is exposed, so model separation is false and the version is `unknown`.
  The preserved review/response chain was necessarily available. Builder claims were
  treated only as hypotheses: every status below was rechecked against the exact
  frozen tree, the complete baseline diff, independent execution, and fresh disposable
  real-Git Tier-E campaigns. No final labels, secret seed, private evaluator,
  unaffiliated implementation, credentials, private hardware result, or empirical
  campaign result was available. This is process-separated contract review, not
  external scientific validation or proof of off-system affiliation or authorship.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex GPT-5"
  builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
  builder_orchestrator_id: "codex-multi-agent-root"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with two unresolved blocking findings. The exact REREVIEW-5
  examples are repaired: `file:///%00bad` and related decoded control, host, port, and
  bracket cases return deterministic errors without raising; numeric 1 and 1.0 cannot
  replace Boolean agreement; Boolean measured values cannot replace numbers; and the
  four named default-port, DNS-dot, dot-segment, and terminal `/.git` aliases resolve
  to candidate reuse. Recursive strict equality also rejected independent nested
  Boolean/number, integer/float, array, and extra-mapping substitutions in every typed
  external receipt family exercised.

  The public boundary is still not complete. A schema-valid, honestly frozen Tier-E
  campaign using external repository `https://example.com/%ZZ/repo` returned `[]`
  even though `%ZZ` is not a valid URI percent escape. Five further complete campaigns
  returned `[]` when candidate repository identity was reused through spellings the
  new canonicalizer does not reconcile: an unreserved percent-encoded host
  (`github.com` versus `%67ithub.com`), equivalent compressed/expanded IPv6 literals,
  a Windows path versus its equivalent `file:///C:/...` URI, a leading-zero IPv4
  spelling, and `ssh` versus `git+ssh` with port 22. The unreserved-host, IPv6, and
  Windows-file cases are demonstrably the same identities; the last two are retained
  as additional accepted portability risks rather than needed to establish the
  blocker. These are local validator gaps within the existing VPLAN-SCHEMA-001 and
  VPLAN-INDEPENDENCE-001 scopes. No new stable finding or test ID is needed.

  The authoritative suite is otherwise green: the separate campaign-contract run
  passed 18 tests, the full repository run passed 177 tests, all six documented
  examples regenerated, validation artifacts passed, and regeneration left the tree
  clean. These results do not demonstrate Tier R, G, or E or validate POPGP physics.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The RESPONSE-6 regression and a fresh full campaign batch rejected
      `file:///%00bad`, percent-decoded newline and DEL controls, double-encoded NUL,
      text and out-of-range ports, a malformed IPv6 bracket, and a missing host with
      deterministic error lists; none raised. Malformed JSON/YAML receipts and the
      earlier nested requirements/governance cases remain covered by the 18-test suite.
      However, a complete real-Git Tier-E campaign honestly frozen with external
      repository `https://example.com/%ZZ/repo` returned `[]`. Python `unquote` leaves
      a malformed escape unchanged, and lines 1088-1169 accept the resulting repository
      identity instead of returning the runbook's required invalid-identity error.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The escaping NUL exception is fixed, but the requested fail-closed malformed-URL
      boundary still accepts an invalid percent escape. This is a local enforceability
      defect, not an off-system identity-truth question.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The complete campaign suite retains blind-seat exposure, prohibited role/session
      reuse, evaluator-only reveal, reproduced-phase order, runner output commitment,
      manifest substitution, resolved-path/byte distinction, retention, and structured
      commitment/reveal receipt reconciliation. Fresh nested structured-receipt
      mutations also failed closed without escaping.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally enforceable custody consistency; off-system custody remains externally verified."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The dedicated suite retained the complete Tier-G positive and raw-false
      countermodel. Missing 3D, acceleration/geodesic, same-source lensing/Shapiro,
      two-potential, and laboratory capability gates; false raw 3D/lensing values;
      alternate rules or pointers; missing bindings; and Boolean/number substitutions
      are rejected at the public validator boundary.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This verifies the gate contract only; no Tier-G scientific result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The 18-test suite retains scientific, capability, resource, access, and invalid
      cause families; all pass/fail/block truth vectors; missing evidence; valid/pending;
      failed-over-blocked precedence; and valid external disagreement. The dedicated
      execution returned deterministic results and no outcome exception.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for deterministic local adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Canonical requirements validation and the dedicated suite retain the lower-wave,
      acyclic, tier-transitively-closed dependency registry and reject unknown edges,
      cycles, same-wave prerequisites, malformed dependency types, and dependent
      holdout start before every prerequisite passes.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and holdout ordering."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and the VIA-900 E5 floor remain immutable. Declared or
      achieved downgrades, unknown evidence levels, and missing cumulative receipt
      kinds are rejected, while a stricter packet declaration remains valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence-class semantic truth remains a substantive independent-review obligation."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retain nonexistent candidate/baseline/protocol rejection,
      candidate-tree identity, packet-rule/requirements/executing-contract freeze,
      manifest/path/hash containment, v1 migration rejection, and Git-bundle
      commit/tree/content binding. Fix commit
      `5f31d810c1bbbfacc319f85f4ee67429373a2159` exists and is an ancestor of the
      reviewed candidate.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the frozen candidate/protocol and external bundle objects."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      The protocol-content regression retains same-path changes to every canonical
      preregistration field family, resources, thresholds, receipt/self hashes, and
      campaign paths. Protocol-commit blobs and packet-freeze-v4 remain decisive after
      mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema still requires the exact canonical nine-field
      envelope. Extra threshold, exclusion, measurement, command, and resource
      authorities are rejected, and all post-freeze protocol substitutions remain
      rejected in the dedicated suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Protocol uniqueness does not establish scientific adequacy."

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      RESPONSE-6 closes every exact REREVIEW-5 case: strict comparison scalar types,
      default HTTPS port, DNS trailing dot, dot segment, terminal `/.git`, custody
      output binding, distinct paths/bytes, Git-bundle commit/tree, orchestrator,
      chronology, and disagreement handling. Six additional nested external receipt
      mutations were rejected, including false-to-zero, 0.0-to-0, extra nested keys,
      Boolean provenance as numeric, list-valued commitment hashes, and list-valued
      comparison metrics.

      Five independently constructed, honestly frozen real-Git Tier-E campaigns still
      returned `[]` for repository reuse that the canonicalizer treated as distinct.
      Decisive examples were `https://github.com/...` versus
      `https://%67ithub.com/...` (unreserved host percent encoding),
      `https://[2001:db8::1]/repo` versus
      `https://[2001:0db8:0:0:0:0:0:1]/repo` (equal `ip_address` values), and
      `C:/src/same-repository` versus `file:///C:/src/same-repository` (equal resolved
      Windows paths). Leading-zero IPv4 and `git+ssh://...:22` variants were also
      accepted but are not needed for the finding. Lines 1116-1169 do not percent-decode
      reg-name hosts, canonicalize IP literals, or unify Windows drive and file-URI
      forms.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      These are locally reconcilable repository aliases, not a demand that the
      validator prove real affiliation or independent authorship. Those off-system
      facts remain correctly assigned to an external maintainer.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      Exact NUL/control, malformed host/port/bracket, duplicate-key, nested type,
      structured-receipt, immutable-ref, and governance provenance cases return error
      lists without raising. The broadened honestly frozen `%ZZ` repository case is
      accepted, so the requested fail-closed malformed-input boundary is incomplete.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain the existing totality matrix and add malformed percent-escape host/path cases."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Blind exposure, seat/session reuse, custodian authority, reveal order,
      output/manifest hashes, resolved path and byte distinction, structured receipts,
      retention, and missing custody fields remain executable negative controls.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for local custody consistency."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false countermodel, missing capability, alternate rule,
      pointer, binding, and Boolean/number cases remain persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      Every requested cause family, truth vector, missing-evidence, ambiguity,
      valid/pending, campaign precedence, and external disagreement row remains
      deterministic in the dedicated suite.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical DAG/wave/tier closure, cycles, missing dependencies, malformed values,
      and premature holdout cases remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown levels, cumulative receipt kinds,
      and stricter declarations remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, rules, requirements, executing-contract, manifest/path/hash,
      escape, schema-version, bundle content, commit/tree, and candidate reuse mutations
      remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the frozen Git/provenance scope."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration field, path, and byte substitutions remain frozen to
      the protocol commit and rejected after mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      Exact-envelope positives and extra threshold, exclusion, measurement, command,
      and resource negatives are persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The expanded clean-room suite now covers the exact REREVIEW-5 aliases and strict
      scalar/nested JSON substitutions while retaining custody, byte/path, bundle,
      orchestrator, chronology, and disagreement cases. It does not cover the accepted
      unreserved-host percent encoding, equivalent IPv6 literal, or Windows file/local
      repository spellings, so candidate repository reuse remains promotable.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain all current cases and extend canonical identity negatives."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: unresolved
    evidence: |-
      Output commitment identity, distinct paths/bytes, bundle commit/tree, external
      orchestrator, typed comparison, pointer/tolerance/order, exact default-port/DNS/
      dot-segment/terminal-git aliases, and valid disagreement cases pass. The three
      decisive broader repository aliases still validate, so the requested canonical
      repository property remains incomplete.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Add unreserved host percent encoding, canonical IP literals, and local/file URI equivalence."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001, TST-VPLAN-INDEPENDENCE-001, and TST-VPLAN-INDEPENDENCE-002"
  predicted_outcome: |-
    A complete remediation will preserve the current positive clean-room campaign and
    all repaired REREVIEW-5 cases while returning deterministic errors for malformed
    percent escapes and treating unreserved-percent hosts, canonical IP literals, and
    equivalent local/file repository spellings as candidate reuse.
  predicted_failure_mode: |-
    Relying on `urllib.parse.unquote` without validating every percent triplet will
    continue to accept malformed identities. Normalizing path escapes but not reg-name
    percent encoding, IP-address text, or file/drive syntax will continue to accept
    different strings for the same repository endpoint.
  confidence_statement: |-
    High. Every decisive counterexample was a complete schema-valid campaign built
    with real candidate/protocol commits and a real external Git bundle, frozen before
    validation, and accepted with an empty error list. Confidence concerns executable
    contract behavior only, not POPGP physics or external identity truth.

recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    Approval is fail closed. RESPONSE-6 repairs every exact REREVIEW-5 mutation and all
    older scientific, custody, outcome, dependency, evidence, freeze, review-chain,
    protocol, and comparison controls remain green. Nevertheless, the public validator
    accepts a malformed percent escape and Tier E still accepts demonstrably equivalent
    candidate repository identities. VPLAN-SCHEMA-001 and VPLAN-INDEPENDENCE-001
    therefore remain blocking; their three active requested tests remain unresolved.
    No Tier R, G, or E result or external scientific validation is established here.
```

## Frozen identity, history, and complete diff

The dedicated reviewer worktree was exact and clean before review execution:

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-6` |
| `git rev-parse HEAD` | 0 | `50df04b05e6991598a9ca0d0ffac45f8b0e0643e` |
| `git rev-parse "HEAD^{tree}"` | 0 | `8a0e7ce46e8ecf0853666f18bd69688bf6e633c2` |
| `git status --short --branch` | 0 | branch header only; no changes |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | baseline is an ancestor |
| `git merge-base --is-ancestor 5f31d810c1bbbfacc319f85f4ee67429373a2159 HEAD` | 0 | RESPONSE-6 fix commit is an ancestor |
| `git diff --check 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | no output |
| `git diff --stat 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | 39 files; 12,229 insertions; 147 deletions |

The exact prior review ref and response ref both exist. The prior-review blob at
`e264b03b...` is byte-identical to the blob in the candidate. As in earlier rounds,
the candidate contains equivalent review commit `938cb0a7...`: it has the same parent,
tree, subject, and review blob as `e264b03b...`, although the named reviewer-branch
commit itself is not a literal ancestor. The immutable ref remains locally resolvable.

The complete baseline-to-candidate diff, the complete 2,605-line validator, complete
2,683-line contract suite, every viability schema/template/requirements object, all
named scientific and reproducibility matrices, and the complete preserved review and
response chain through RESPONSE-6 were read. Builder summaries were not used as proof.

## Authoritative execution

Every command in README, CI, and the launch runbook ran against the frozen candidate.
The campaign-contract suite was also run separately as requested.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; fresh environment; 60 locked packages installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; braces and environments balanced; no Markdown remnants |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `18 passed in 432.07s` |
| `uv run pytest -q` | 0 | `177 passed in 470.70s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic passed |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic/Kubo--Mori/Richardson/spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `git diff --exit-code` and `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | no generated diff and no untracked files |

`pdflatex` and `nvcc` were unavailable. They are optional rather than authoritative CI
commands, so no PDF/native success is claimed. No hidden campaign, private evaluator,
unaffiliated group, native hardware result, continuum result, or empirical data were
available.

## Independent adversarial evidence

All decisive campaigns lived under disposable `TemporaryDirectory` repositories, used
real candidate/protocol commits and external Git bundles, were hash-consistent, and
called the public `validate_campaign(..., repo_root=...)` boundary. Mutations intended
as honest declarations were applied before packet-rule and protocol freezing.

| Mutation/check | Observed result |
|---|---|
| Exact REREVIEW-5 four repository aliases | all rejected as candidate reuse |
| Agreement numeric `1`/`1.0`; Boolean candidate/external measured values | all rejected |
| NUL/newline/DEL, double-encoded NUL, bad text/range port, malformed bracket, missing host | all deterministic invalid-identity errors; no exception |
| Invalid percent escape `https://example.com/%ZZ/repo` | **accepted** |
| Percent-encoded unreserved hostname candidate alias | **accepted** |
| Equivalent compressed/expanded IPv6 candidate alias | **accepted** |
| Equivalent Windows local path/file-URI candidate alias | **accepted** |
| Leading-zero IPv4 and `git+ssh` port-22 spellings | **accepted; portability risks not needed for the finding** |
| Query, fragment, percent-encoded path, SCP, repeated separator, chained terminal `.git` aliases | rejected as candidate reuse |
| Six nested mapping/list/scalar substitutions across contract, provenance, commitment, comparison | all rejected by strict equality |
| Honestly frozen 16 MiB malformed Git bundle | controlled clone error; no exception |

The malformed bundle demonstrates controlled failure at that size, not a general
evidence-byte ceiling. The validator hashes local evidence by streaming and bounds Git
clone by 30 seconds, but it has no explicit receipt-byte ceiling; this remains an
operator-side resource limitation rather than a promoted finding in this round.

## Recommendation

Changes requested: **two unresolved blocking findings**. The exact RESPONSE-6 remedies
are real, but fail-closed approval requires malformed percent escapes to be rejected and
the broader canonical repository identities above to reconcile to candidate reuse.
Green CI and same-operator agent agreement do not demonstrate POPGP scientific
viability or external independence.
