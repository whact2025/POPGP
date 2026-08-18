# POPGP mechanism-viability campaign R1

This directory is the executable campaign record for the first attempt to establish
Tier R (mechanism viability) under
[`VIABILITY_DEMONSTRATION_PLAN.md`](../../../docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md).

The scientific candidate is immutable commit
`e6f8dc5a55032f92ecfd5a18505705fd9387f1d4` with tree
`a5c44686e72ed9c2f99988b8136078151e59416d`. The comparison baseline is
`70c867552279b74d5ce1a7bc5c50d5a980cf81e6`.

The campaign is deliberately fail closed. `VIA-000` is preregistered first. The six
dependent Tier-R packets have frozen hypotheses, rules, budgets, and attacks but stay
in `drafted` lifecycle state until their machine-declared prerequisites pass. A green
`VIA-000` establishes reproducibility and evidence integrity only; it does not establish
the physical mechanism or Tier R.

## Current state

- Campaign decision: `pending`.
- `VIA-000`: preregistered; holdout not started.
- `VIA-010`, `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300`, `VIA-400`: drafted.
- Sealed holdout/seed commitments exist outside the repository at the immutable URIs
  and SHA-256 values recorded in each packet. Their contents have not been used as
  calibration evidence.
- A builder-owned pre-freeze calibration is retained in
  [`calibration/VIA-000-2026-08-13.md`](calibration/VIA-000-2026-08-13.md). It is not a
  decisive campaign receipt and cannot satisfy the E3 evidence floor.
- A builder-owned Blackwell/native calibration is retained in
  [`calibration/VIA-300-Blackwell-2026-08-13.md`](calibration/VIA-300-Blackwell-2026-08-13.md).
  It confirms executable CUDA hardware and native `sm_120` code, but it is not an
  independent VIA-300 receipt and does not establish Tier R.

## Validation

After the protocol snapshot commit is recorded in `CAMPAIGN.yaml` and every packet,
the authoritative command is:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Only hashed receipts, reconciled independent review artifacts, custody reveal records,
and a validator-clean adjudication can move a packet or campaign to a terminal outcome.
