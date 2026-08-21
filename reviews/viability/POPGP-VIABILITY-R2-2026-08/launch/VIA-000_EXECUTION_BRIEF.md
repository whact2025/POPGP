# VIA-000 R2 clean-session execution brief

Use this brief to launch separate seats for `POPGP-VIABILITY-R2-2026-08`. Every seat
must use a fresh task/session, record its actual identity metadata, and preserve the
R1 valid/failed result. No seat may invent model/version metadata.

## Frozen identities

- Scientific candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Failed R1 comparison baseline: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Independent implementation approval: `f7bffc98d851ad2c333c5b01e4221b655908de7e`
- Independent protocol-boundary approval: `610701ae04d1ae644da62814a71765b4ba8c0ede`
- Protocol snapshot: `PENDING_R2_PROTOCOL_REFREEZE`
- Activated campaign handoff: `PENDING_R2_ACTIVATION`
- Campaign: `reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml`
- Packet: `reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml`
- Primary protocol: `protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json`
- Supporting runner: `protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1`
- Mutation runner: `protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-MUTATION-RUNNER.py`
- Frozen assembler: `protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py`
- Raw-results schema: `protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json`

Before role-specific work, read the campaign, packet, all five protocol artifacts,
all activated pre-holdout amendments,
`docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md`,
`docs/governance/AGENT_REVIEW_WORKFLOW.md`, and
`docs/governance/REVIEWER_IDENTITY.md` completely. Verify the exact candidate,
protocol, and handoff commits. Do not access the untracked builder handoff memo.

## Required order

1. Custodian verifies all sealed R2 commitments without revealing contents.
2. A fresh falsifier independently executes all eighteen frozen mutation families and
   commits only its attack-plan receipt.
3. Only after the falsifier reports the complete gate safe may a maintainer record
   lifecycle `attacked` and `holdout_started: true`.
4. A separate reproduction runner executes the frozen clean and mutation protocols on
   exact-candidate Windows and Ubuntu, archives raw output, and commits a distinct
   output commitment before reveal.
5. Custodian verifies the commitment and authorizes reveal. The runner must not rerun
   or replace output after reveal.
6. Separate statistical, claim, and independent-review seats audit the immutable
   receipts.
7. A non-builder adjudicator evaluates only frozen Boolean rules and hash-verified
   receipts. A maintainer separately authorizes a terminal campaign decision.

Do not combine the falsifier, runner, custodian, or adjudicator seats in one task.
Builder calibration and prior reviewer conclusions are not falsifier/runner evidence.
The exact-SHA GitHub Actions workflow is the authenticated producer. Each accepted
platform summary and evidence manifest must have a GitHub OIDC/Sigstore attestation
for the frozen repository, signer workflow, and protocol source commit. Self-contained
hashes or caller-supplied identity text are insufficient. The assigned runner and
independent reviewer remain operationally trusted seats; the protocol still does not
claim protection against a malicious or colluding GitHub control plane or repository
administrator able to authorize the exact signer workflow.

## Custodian verification prompt

```text
Act only as evaluator_custodian for POPGP campaign POPGP-VIABILITY-R2-2026-08,
packet VIA-000. Use the frozen custodian identity/session in the packet. In a fresh
isolated task, privately verify the raw-byte SHA-256 values and packet IDs of all
fourteen committed R2 hidden holdout/seed files under
C:/src/POPGP-custody/POPGP-VIABILITY-R2-2026-08. Do not print, quote, summarize, or
reveal their contents. Report only per-file hash match, byte count, actual identity
metadata, and access limitations. Do not edit the campaign and do not reveal anything
until a valid runner output commitment exists.
```

## Falsifier prompt

```text
Act only as falsifier for POPGP campaign POPGP-VIABILITY-R2-2026-08, packet VIA-000,
using identity codex-via000-r2-falsifier and session
popgp-viability-r2-2026-08-via000-falsifier-session-1. Start from the activated handoff
above in a fresh isolated worktree. Read every required public file, but do not access
custody files, hidden manifests, the untracked builder handoff memo, final labels,
seeds, or builder conclusions.

Independently execute all eighteen frozen mutation families. Include exact historical
R1/R2 attacks and broadened equivalents for semantic raw/derived contradictions,
localized visuals, graph/MI/edge provenance, raw/effective source and zero-mode policy,
mass/gauge/source model, startup hooks, executable bytecode/dependency tamper,
index/filter/ignored/symlink concealment, transient self-restoring source, PDF/output
residue, platform-only success, and receipt/commitment chronology. Treat the builder's
tests as hypotheses. Do not repair implementation or change the protocol.

Create attacks/VIA-000-R2-ATTACK-PLAN-1.md under the campaign, record exact commands,
raw outcomes, expected rejection conditions, exposure boundaries, and an unequivocal
safe/unsafe recommendation. Commit only that artifact, leave normal and ignored state
clean, push a dedicated branch, and return commit/tree/path/SHA-256/identity metadata.
If any mutation is accepted or the clean control cannot execute, mark the protocol
unsafe and do not authorize holdout.
```

## Reproduction-runner prompt

```text
Act only as reproduction_runner for POPGP campaign POPGP-VIABILITY-R2-2026-08,
packet VIA-000, using the frozen runner identity/session. Begin only after the packet is
validator-clean in attacked/holdout-started state and the falsifier artifact authorizes
the complete gate. Do not inspect sealed holdouts or seeds.

Materialize all five protocol artifacts from the frozen protocol snapshot. Execute the
exact Windows and Ubuntu commands from the primary protocol in fresh paths, then run
the frozen mutation runner against the exact protocol source commit. Retain every raw stdout/stderr stream,
environment/source manifest, exit status, duration, PDF artifact, visual/JSON result,
mutation suite/result, derived mutation receipt, producer-attestation bundle, and Git
boundary result. Use only the frozen assembler to combine the two complete attested
platform fragments. Do not supply or accept separately authored mutation files.
The assembler must refuse a missing/nonzero/contradictory input and emit no commitment.
R2 raw results always record blocked=false; an unavailable or incomplete attempt is
invalid and cannot be assembled. Commit the assembler's distinct output-commitment
receipt binding the raw-results SHA-256 before requesting reveal. Do not rerun,
replace, or reinterpret output after reveal.
```

## Adjudicator prompt

```text
Act only as adjudicator for POPGP campaign POPGP-VIABILITY-R2-2026-08, packet VIA-000,
in the frozen adjudicator session. Begin only after output commitment, custodian reveal,
attack/mutation results, statistical audit, claim diff, and a reconciled independent
review chain are immutable. Recompute every receipt hash and evaluate the frozen
popgp-bool-v2 expressions without changing thresholds or repairing evidence. A valid
round has exactly one passed/failed/blocked outcome; protocol or evidence invalidity
leaves the round invalid/pending. Commit only adjudication/lifecycle/receipt/review-
chain/campaign-decision fields allowed after freeze and return exact validator output.
Do not claim Tier R while any required packet remains pending.
```
