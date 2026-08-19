# VIA-000 clean-session execution brief

Use this brief to launch the independent seats for the first packet of
`POPGP-VIABILITY-R1-2026-08`. Each seat must run in a fresh task/session and must
record its actual model identity and version. A role label is not permission to invent
model metadata.

The packet records model identity/version as `unknown`, because the protocol designer
did not have an operator-supplied exact runtime identifier. Each artifact must retain
that honest limitation or trigger a pre-holdout protocol amendment; it must not guess.

## Frozen identities

- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Protocol snapshot: `2894b408527e5c6457eeddcfb1ae657c9a205220`
- Activated campaign handoff: `502cafea22d581c06861a6c55eeaed72f74eccb0`
- Campaign: `reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml`
- Packet: `reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml`
- Primary protocol: `protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json`

Before doing role-specific work, every seat must read the campaign, packet, primary
protocol, `docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md`,
`docs/governance/AGENT_REVIEW_WORKFLOW.md`, and
`docs/governance/REVIEWER_IDENTITY.md` completely. Verify the candidate, protocol,
and handoff commits and leave the implementation unchanged unless the seat is the
builder.

The calibration memo is builder context and must not be given to a clean reviewer,
falsifier, or runner before that seat commits its initial result. Those seats may find
it later while inspecting the public tree; if so, disclose that exposure and do not
copy its conclusions as evidence.

## Required order

1. Custodian verifies the two sealed VIA-000 commitments without revealing contents.
2. Falsifier commits a public attack plan covering all ten frozen mutation families.
3. The campaign maintainer records lifecycle `attacked`, `holdout_started: true`, and
   the attack-plan receipt; the validator must remain clean.
4. Reproduction runner executes the exact protocol in fresh exact-SHA Linux and
   Windows environments, archives raw output, writes `raw-results.json`, and commits
   its output hash before any reveal.
5. Custodian verifies the output commitment, reveals the committed manifests, and
   records the reveal bytes and timestamp. The runner must not rerun or replace output
   after reveal.
6. Statistical auditor and falsifier independently audit the frozen results and
   mutations.
7. Claim auditor creates the schema-valid independent-review artifact without editing
   implementation.
8. Adjudicator evaluates only hashed receipts and frozen Boolean rules, records the
   packet outcome, and runs the campaign validator. A maintainer separately authorizes
   any terminal campaign decision.

Do not combine steps 2, 4, 5, or 8 in one task/session. Do not expose the sealed seed,
holdout bytes, or final labels to the falsifier, builder, or runner before the runner's
output commitment.

## Custodian prompt

```text
Act as evaluator_custodian for POPGP campaign POPGP-VIABILITY-R1-2026-08,
packet VIA-000. Your frozen seat identity is codex-via000-custodian and your
session ID is popgp-viability-r1-2026-08-via000-custodian-session.

Use a fresh isolated task. Read the required files named in VIA-000_EXECUTION_BRIEF.md.
Privately verify the raw-byte SHA-256 values and packet IDs of:
C:/src/POPGP-custody/POPGP-VIABILITY-R1-2026-08/VIA-000-hidden-holdout.json
C:/src/POPGP-custody/POPGP-VIABILITY-R1-2026-08/VIA-000-secret-seed.json
against the packet commitments. Do not print, quote, summarize, or reveal their
contents. Report only whether each hash matches, the exact commands used, your actual
model/operator/session/orchestrator metadata, and any access limitation. Do not edit
the campaign yet and do not reveal either manifest until a valid runner output
commitment exists.
```

## Falsifier prompt

```text
Act as falsifier for POPGP campaign POPGP-VIABILITY-R1-2026-08, packet VIA-000.
Your frozen seat identity is codex-via000-falsifier and your session ID is
popgp-viability-r1-2026-08-via000-falsifier-session-5.

Use a fresh isolated task at the activated handoff listed above. Read the required
public files named in VIA-000_EXECUTION_BRIEF.md. Do not request or inspect sealed manifests, final labels,
or builder conclusions. Independently construct executable attacks for all ten frozen
mutation families and add boundary attacks for undeclared state, OS-only success,
semantic drift, PDF failure, dirty regeneration, and evidence available only in a
summary. Specifically attack ignored virtual-environment customize and executable
`.pth` startup hooks including self-cleaning allowed-file variants, isolated/frozen
non-editable uv execution, and the base-interpreter
preflight/postflight boundary. Do not repair the implementation.
Create `attacks/VIA-000-ATTACK-PLAN-5.md`, record exact commands and expected
rejection conditions, commit only that artifact,
leave the worktree clean, and return its commit, path, SHA-256, actual identity fields,
and whether any exposure boundary was crossed.
```

## Reproduction-runner prompt

```text
Act as reproduction_runner for POPGP campaign POPGP-VIABILITY-R1-2026-08,
packet VIA-000. Your frozen seat identity is codex-via000-runner and your session ID
is popgp-viability-r1-2026-08-via000-runner-session.

Start only after the packet is validator-clean in attacked/holdout-started state.
Use fresh exact-SHA clones of scientific candidate
9a29e05f803666bf0e3a28417ea399e3e26769fc. Do not inspect the sealed holdout or seed.
Run every command in protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json on Linux and
Windows as specified, repeat the PDF command twice, execute the ten committed mutation
tests corresponding exactly to the mutation plan, and retain raw stdout/stderr,
environment, exit status, duration, artifact hashes, PDF metadata, Git cleanliness,
and residue checks. No native/CUDA result is in scope.

Write a strict JSON raw-results document containing Boolean capabilities named
evidence-contract, cross-platform-reproduction, and mutation-rejection, plus Boolean
failed and blocked. Write a separate output-commitment JSON binding the raw-results
receipt ID and SHA-256, with committed_by codex-via000-runner and a UTC timestamp.
Commit the evidence and commitment before requesting any reveal. Do not rerun, replace,
or reinterpret output after reveal. Return only immutable refs, paths, hashes, exact
command outcomes, and disclosed identity/access metadata.
```

## Statistical-auditor prompt

```text
Act as statistical_auditor for POPGP campaign POPGP-VIABILITY-R1-2026-08,
packet VIA-000, using session
popgp-viability-r1-2026-08-via000-statistics-session. Start from the immutable runner
receipts after output commitment. Verify types, counts, platform coverage, exact
candidate identity, semantic-tolerance use, mutation rejection, PDF metadata, command
completeness, and absence of post-output threshold selection. Recompute hashes and
selected comparisons from raw evidence. Create and commit only a statistical-audit
artifact. State actual model/operator/session/orchestrator metadata and all exposure
limitations; do not remediate or adjudicate.
```

## Claim-auditor / independent-reviewer prompt

```text
Act as claim_auditor and independent-reviewer for POPGP campaign
POPGP-VIABILITY-R1-2026-08, packet VIA-000, using a fresh task/session identified as
popgp-viability-r1-2026-08-via000-claims-session.

Read the required governance and docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md plus
schemas/viability/independent-review-v2.schema.json. Audit scientific candidate
9a29e05f803666bf0e3a28417ea399e3e26769fc and the immutable VIA-000 evidence/attack/
statistical receipts. Independently rerun targeted checks and counterexamples. Verify
that a VIA-000 pass would establish only evidence integrity, not mechanism viability,
GR, Lorentz recovery, continuum behavior, native scalability, or external empirical
validation. Give stable finding and requested-test IDs, complete typed identity,
independence, and hidden-access declarations, and exact evidence refs.

Create a schema-valid initial independent-review artifact, commit only that artifact,
leave the worktree clean, and return its full commit:path ref and recommendation. Do
not edit implementation or mark builder assertions resolved without reproduction.
```

## Adjudicator prompt

```text
Act as adjudicator for POPGP campaign POPGP-VIABILITY-R1-2026-08, packet VIA-000,
using session popgp-viability-r1-2026-08-via000-adjudicator-session. Begin only after
the output commitment, custodian reveal, attack/mutation results, statistical audit,
claim diff, and independently reconciled review chain are immutable.

Recompute every receipt hash and evaluate the frozen popgp-bool-v2 rules without
changing thresholds or repairing evidence. Apply failed over blocked over passed
precedence as encoded by the validator. Record a valid terminal result only if the
rules are exclusive and complete; otherwise record an invalid round with a pending
outcome. Create the adjudication receipt, update only lifecycle/receipts/review-chain/
adjudication/campaign decision fields allowed after protocol freeze, run
scripts/check_viability_campaign.py, commit the result, and return the exact validator
output and commit. Do not claim Tier R while any required packet remains pending.
```
