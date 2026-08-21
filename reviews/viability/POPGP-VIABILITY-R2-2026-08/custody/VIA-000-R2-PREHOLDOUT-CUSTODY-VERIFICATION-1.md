# VIA-000 R2 pre-holdout custody verification 1

Status: **PASS**

Verified at: `2026-08-21T09:32:15.3688446Z`

## Frozen public bindings

- Campaign: `POPGP-VIABILITY-R2-2026-08`
- Packet set: `VIA-000`, `VIA-010`, `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300`, `VIA-400`
- Audited public handoff commit: `6eac4889c7d660ec140559dba604f67d03bb04a1`
- Audited public handoff tree: `2c034bbc43c6d2efaaefe29fcf2a70d35d2a9624`
- Independent protocol-boundary approval: `610701ae04d1ae644da62814a71765b4ba8c0ede`
- Protocol snapshot: `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`
- Activated campaign handoff: `3bf74151b8f848ab8b6d44bcf84946ab6a890bde`
- SAFE post-refreeze falsifier artifact commit: `6eac4889c7d660ec140559dba604f67d03bb04a1`
- Campaign bytes SHA-256: `b7fa9b0362312ee3dd732723cf02e0dd12aa2bce63e082ddea85003dd2af2eba`
- Execution-brief bytes SHA-256: `c1d0994a20a5f53344e168a633c79be6b8f6a447c074ae44181297579a2f12bc`
- SAFE attack-plan bytes SHA-256: `583ab5701ce9fbd9a1e0896a0d77aba80389b70de971e511e95432c9c61b205f`

Commit ancestry was verified in the required approval -> snapshot -> activation ->
public-handoff direction. The public campaign validator returned:
`Viability campaign contract is valid.`

## Custodian identity

| Field | Actual value |
|---|---|
| Seat | `evaluator_custodian` |
| Agent identity | `codex-via000-r2-custodian` |
| Model identity | `OpenAI Codex (GPT-5)` |
| Model version | `not-exposed` |
| Operator | `NVIDIA.COM\rfuoco` |
| Session ID | `popgp-viability-r2-2026-08-via000-custodian-session` |
| Orchestrator | `codex-desktop` |
| Organization | `not-exposed` |
| Access level | `hidden-manifest-custody` |

The same frozen custodian identity and session metadata were present in all seven
packets. They did not reuse the builder, falsifier, reproduction-runner, or
adjudicator identity/session.

## Private commitment verification

Each check read only the raw bytes of the packet-referenced sealed source, computed
SHA-256 using the frozen `raw-bytes-v1` rule, and parsed only the top-level packet ID.
No sealed filename, URI, content, label, seed value, preimage, or private evaluator
material is recorded here.

| Packet | Sealed artifact role | Byte count | SHA-256 matches packet | Packet ID matches |
|---|---|---:|---|---|
| `VIA-000` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-000` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-010` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-010` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-100` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-100` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-150` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-150` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-200` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-200` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-300` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-300` | secret-seed manifest | 1,078 | yes | yes |
| `VIA-400` | hidden-holdout manifest | 1,093 | yes | yes |
| `VIA-400` | secret-seed manifest | 1,078 | yes | yes |

Verified: **14 of 14**. Mismatches: **0**.

## Exposure and authority boundary

- Final labels seen: `false`
- Secret-seed manifest raw bytes accessed for commitment and packet-ID verification:
  `true`
- Secret seed value revealed or recorded: `false`
- Private evaluator material seen: `false`
- Sealed content printed, quoted, summarized, or persisted: `false`
- Candidate experiments run: `false`
- Raw campaign result or output commitment created: `false`
- Reveal performed or authorized: `false`
- Campaign or packet lifecycle edited: `false`
- `holdout_started` changed: `false`

## Recommendation

The frozen pre-holdout custody gate is **PASS**. Together with the exact-handoff SAFE
falsifier artifact, this authorizes a separate maintainer to record the permitted
`attacked` / `holdout_started: true` lifecycle transition. This custody artifact does
not itself start the holdout, authorize reveal, run the candidate, create an output
commitment, or adjudicate a packet.
