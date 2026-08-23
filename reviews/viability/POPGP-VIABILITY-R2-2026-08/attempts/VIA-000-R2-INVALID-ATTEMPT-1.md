# VIA-000 R2 invalid reproduction attempt 1

Status: invalid and stopped; not a scientific pass, failure, or blockage.

## Frozen context

- Campaign: `POPGP-VIABILITY-R2-2026-08`
- Packet: `VIA-000`
- Scientific candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Frozen protocol snapshot: `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`
- Attacked/holdout-started handoff: `5dfee2355e06aeed45c7dae662e890a6dc6452e8`
- Custody state throughout: sealed and unrevealed

## Immutable hosted attempt

GitHub Actions run `32468904587` executed from the attacked handoff. Both platform
jobs completed successfully:

| Platform | Job | Artifact | Archive digest |
| --- | --- | --- | --- |
| Ubuntu | `96731349174` | `9442167217` | `sha256:752c5630afae8f8d4a6a15597de53d6c365d6170ce44a6b569433e6a1cac5c47` |
| Windows | `96731348975` | `9442330955` | `sha256:057526a7d3ebbedad3f4ed7187f6a9c59fd65a86d60a45a4f6a2d290e7a8f052` |

The downloaded ZIP bytes matched those digests. All four platform-summary and
evidence-manifest subjects verified under GitHub OIDC/Sigstore against the repository,
frozen workflow, SLSA v1 predicate, GitHub-hosted runner, and source commit
`5dfee2355e06aeed45c7dae662e890a6dc6452e8`.

Each platform reported all sixteen commands successful, 366 tests passed, six
generators completed, twelve visuals retained, eighteen mutation families rejected
across 145 mutation nodes, and an eleven-page PDF.

## One-shot assembly and rejection

The frozen assembler ran exactly once and emitted an external, unattached package:

- raw-results SHA-256: `59bfca6fdd94cbfdab78125b3ba26b8d562e9cbd217f8da656060ac6cb35366a`
- output-commitment SHA-256: `43f652cea7fee1d009464ad9d872c9a7c1b5272b281533baad132a00bffa875f`
- commitment time: `2026-08-21T10:02:43.8578205+00:00`

Authoritative full-campaign validation then rejected the package. The packet requires
every raw result and producer attestation to bind protocol source commit
`9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`, while the workflow passed
`${{ github.sha }}` and therefore bound the attacked handoff
`5dfee2355e06aeed45c7dae662e890a6dc6452e8`. The frozen assembler accepted its
caller-supplied handoff identity, but the public campaign validator correctly rejected
the packet-level contradiction.

## Fail-closed disposition

- No invalid raw-results or output-commitment receipt was attached to the packet.
- No campaign commit or push was made from the runner seat.
- No custody reveal was requested or performed.
- No output was rerun, replaced, or reinterpreted after commitment.
- `VIA-000` remains `attacked`, `holdout_started: true`, unrevealed, and pending.
- The invalid package remains external to Git at
  `C:/src/POPGP-VIABILITY-R2-2026-08-RUN-32468904587-ASSEMBLED` for audit only.

This is a protocol/workflow identity defect. Under the frozen governance rules, the
attempt is invalid rather than scientifically failed or blocked, and R2 cannot be
repaired or rerun in place. A newly preregistered round is required.
