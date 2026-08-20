# POPGP mechanism-viability campaign R2

This directory is the executable record for the second attempt to establish Tier R
under [`VIABILITY_DEMONSTRATION_PLAN.md`](../../../docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md).
R2 does not replace or reinterpret R1. The R1 `VIA-000` round remains an immutable,
valid, failed E3 result.

The R2 scientific candidate is
`5be3c38a0822d49953d0933f14ccab32ca12c896` with tree
`6ad387f9f4e0bab7f97df1bb54a03177887f0707`. Its comparison baseline is the failed R1
candidate `9a29e05f803666bf0e3a28417ea399e3e26769fc`.

## Current state

- Campaign decision: pending.
- `VIA-000`: preregistered, not attacked, holdout not started, unrevealed, and not run.
- `VIA-010`, `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300`, `VIA-400`: drafted and
  dependency-gated.
- Fourteen fresh R2 hidden holdout/seed commitments are bound in the seven packets.
  They were created by the custodian outside Git. The protocol designer and builder
  received hashes and byte counts only; no hidden bytes were revealed.
- The implementation remediation was independently approved with zero blockers in
  `f7bffc98d851ad2c333c5b01e4221b655908de7e`. That approval licenses protocol freeze;
  it is not a packet outcome or a Tier-R result.
- R2 retains the two-platform, PDF, mutation, semantic, visual, startup, source-byte,
  graph/source/configuration-provenance, output-commitment, and custody gates that
  falsified or hardened R1.

## Frozen execution boundary

The primary protocol is
[`VIA-000.json`](../../../protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json). Its
supporting runner is
[`VIA-000-RUNNER.ps1`](../../../protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1).
The runner creates a fresh exact-candidate clone, a complete external locked
environment and cache boundary, and hash-retained raw evidence. It executes the clean
Linux/Windows sequence only. The separate falsifier must first implement and commit
attacks for all eighteen frozen mutation families; the reproduction runner then
executes those attacks without seeing hidden manifests.

The packet cannot advance past preregistration until a fresh falsifier approves the
complete gate. It cannot advance to reproduced until a separate runner creates and
commits raw results plus an output commitment. Only the custodian may then reveal.

## Validation

After the protocol snapshot commit has been activated in `CAMPAIGN.yaml` and every
packet, run:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml
```

A green validator at preregistration establishes only a coherent immutable protocol.
It does not establish evidence integrity, the physical mechanism, or Tier R.
