# POPGP mechanism-viability campaign R3

This directory is the drafted third attempt to establish Tier R under
[`VIABILITY_DEMONSTRATION_PLAN.md`](../../../docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md).
R3 preserves the exact R2 scientific candidate
`5be3c38a0822d49953d0933f14ccab32ca12c896`, candidate tree
`6ad387f9f4e0bab7f97df1bb54a03177887f0707`, and comparison baseline
`9a29e05f803666bf0e3a28417ea399e3e26769fc`.

R1 remains a valid failed E3 result. R2 remains an immutable attacked,
holdout-started, unrevealed, pending round whose sole hosted attempt was invalid because
its attestations bound lifecycle handoff `5dfee2355e06aeed45c7dae662e890a6dc6452e8`
instead of protocol snapshot `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`.
R3 does not repair, rerun, reveal, or reinterpret either predecessor.

## Draft state

- Campaign decision: pending.
- All seven Tier-R packets: drafted, unrevealed, `holdout_started: false`, and not run.
- No R3 attack, raw result, output commitment, reveal, audit, or adjudication exists.
- The protocol requires independent adversarial review before any final protocol
  snapshot, preregistration, falsifier run, custody transition, or holdout execution.

## Identity-safe hosted execution

R3 removes automatic push execution. The sole hosted workflow is
`.github/workflows/via000-r3-protocol.yml`, and it accepts only a manual dispatch from
the lightweight content-addressed tag
`refs/tags/popgp-via000-r3-protocol-<protocol-snapshot-commit>`. The caller supplies a
separate SSH-signed annotated authorization tag whose name embeds the SHA-256 of its
canonical record. That record points to an immutable authorization commit whose
campaign, packet, and manifest blobs independently authorize the same snapshot.
The source-tag suffix and resolution, authorized packet `protocol_commit`,
`github.sha`, checkout HEAD, runner and mutation `protocol_source_commit`, Sigstore
source digest, raw-results identity, assembler-derived identity, and validator
expectation must all be the same lowercase 40-hex commit. Linux and Windows fragments
must also carry one shared GitHub Actions run ID and attempt. A branch/lifecycle HEAD,
later self-consistent tag, moved/substituted authorization, mutable validator source,
wrong-source attestation, or cross-run mixture fails before output commitment.

The checked-in allowed-signers file is deliberately comment-only. Activation is
blocked until a separately reviewed amendment freezes exactly one Ed25519 public key,
after which an authorized maintainer may create the binding commit and signed tag.

## Custody carry-forward boundary

No hidden byte was created, read, copied, or changed during R3 drafting. The packets
carry forward only the fourteen public R2 immutable URI/SHA-256 commitment pairs.
Those commitments are not yet accepted as R3 custody evidence. Before R3 may advance
to preregistration or holdout, a fresh R3 evaluator/custodian session must privately
verify all fourteen sealed raw-byte hashes and packet IDs, without revealing content,
and record a new public carry-forward verification artifact. Any missing or mismatched
sealed byte requires new custody material and a new draft; it cannot be waived.

## Validation

From the exact draft-handoff checkout, run:

```powershell
uv run --frozen python -m scripts.check_viability_campaign `
  reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml
```

A green result establishes only a coherent drafted protocol. It is not independent
approval, reproduction evidence, a scientific outcome, or Tier-R viability.
