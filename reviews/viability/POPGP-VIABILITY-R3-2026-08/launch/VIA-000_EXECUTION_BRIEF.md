# VIA-000 R3 draft execution brief

Status: design handoff for independent review only. Do not dispatch or start holdout.

## Fixed scientific identities

- Candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Comparison baseline: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- R2 invalid attempt record:
  `../attempts/VIA-000-R2-INVALID-ATTEMPT-1.md` in the immutable R2 campaign
- R3 protocol snapshot: pending independent review and final freeze
- R3 content-addressed tag: pending; it must be
  `popgp-via000-r3-protocol-<final-protocol-snapshot-commit>`

## Draft surfaces

- Campaign: `reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml`
- Packet: `reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml`
- Primary protocol: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json`
- Dispatch guard: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py`
- Runner: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1`
- Mutation runner: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py`
- Assembler: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py`
- Raw-results schema:
  `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json`
- Hosted workflow: `.github/workflows/via000-r3-protocol.yml`

## Required order after independent approval

1. A maintainer creates the final protocol snapshot commit and a tag whose exact name
   embeds that same commit.
2. The drafted campaign is activated only after its packet/manifest bindings point to
   that snapshot and the authoritative campaign validator passes.
3. A fresh custodian privately verifies all fourteen carried-forward sealed R2
   commitments and records only hash/byte-count/packet-ID outcomes.
4. A fresh falsifier executes the clean control and all eighteen mutation families.
5. Only after custody and falsifier gates pass may a maintainer record `attacked` and
   set `holdout_started: true`.
6. A separate reproduction runner manually dispatches the workflow from the exact
   content-addressed tag with `protocol_snapshot_commit` equal to the tag suffix.
7. Both matrix jobs must share `github.run_id` and `github.run_attempt`; all signed
   summaries and manifests must bind that run and the exact source commit.
8. The runner invokes the frozen assembler with the same tag, commit, run ID, and
   attempt. Any mismatch must leave no output directory or commitment.
9. Only the custodian may authorize reveal after a valid immutable commitment.
10. Fresh statistical, claim, independent-review, and adjudication seats complete the
    frozen governance sequence.

Never dispatch from a campaign branch, activation/handoff commit, or mutable lifecycle
HEAD. Never substitute calibration output for a campaign result. Never reuse the R2
invalid output package.
