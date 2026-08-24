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
- R3 signed authorization tag: pending; it must be an SSH-signed annotated tag named
  `popgp-via000-r3-authorization-<sha256-of-canonical-record>`, point to an immutable
  authorization commit, and bind campaign/packet/manifest bytes that authorize the
  exact protocol snapshot.
- Authorization signer: intentionally absent. Activation is blocked until a separate
  reviewed amendment freezes exactly one Ed25519 public key.

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

1. Independent review accepts the remediation and a separate signer-key amendment
   freezes exactly one Ed25519 authorization public key.
2. A maintainer creates the final protocol snapshot commit and a lightweight tag whose
   exact name embeds that same commit.
3. A distinct authorization commit freezes campaign, packet, and protocol-manifest
   bytes that all bind that snapshot. The designated signer creates the canonical,
   content-addressed, signed annotated authorization tag pointing at that commit.
4. The campaign is activated only after both tag identities and the authoritative
   campaign validator pass.
5. A fresh custodian privately verifies all fourteen carried-forward sealed R2
   commitments and records only hash/byte-count/packet-ID outcomes.
6. A fresh falsifier executes the clean control and all eighteen mutation families.
7. Only after custody and falsifier gates pass may a maintainer record `attacked` and
   set `holdout_started: true`.
8. A separate reproduction runner manually dispatches the workflow from the exact
   content-addressed protocol tag and supplies the exact signed authorization ref.
   The workflow captures that ref as one tag object ID and uses only that object for
   parsing, peeling, signature verification, and retained identity. All such Git
   operations use `--no-replace-objects`, a scrubbed Git environment/config, and the
   absolute system Git/SSH verifier programs; a changed ref or substituted object
   aborts before platform execution.
9. Both matrix jobs must share `github.run_id` and `github.run_attempt`; all signed
   summaries and manifests must bind that run and the exact source commit.
10. The runner invokes the frozen assembler with the same protocol and authorization
    refs, run ID, and attempt. The assembler derives the commit only from authorized
    immutable bytes, independently pins the authorization tag object, and runs a
    manifest-verified validator bundle extracted through no-replacement Git reads
    from the snapshot. Any mismatch must leave no output directory or commitment.
11. Only the custodian may authorize reveal after a valid immutable commitment.
12. Fresh statistical, claim, independent-review, and adjudication seats complete the
    frozen governance sequence.

Never dispatch from a campaign branch, activation/handoff commit, or mutable lifecycle
HEAD. Never substitute calibration output for a campaign result. Never reuse the R2
invalid output package.
