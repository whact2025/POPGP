# POPGP mechanism-viability campaign R1

This directory is the executable campaign record for the first attempt to establish
Tier R (mechanism viability) under
[`VIABILITY_DEMONSTRATION_PLAN.md`](../../../docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md).

The scientific candidate is immutable commit
`9a29e05f803666bf0e3a28417ea399e3e26769fc` with tree
`358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`. The comparison baseline is
`70c867552279b74d5ce1a7bc5c50d5a980cf81e6`.

The campaign is deliberately fail closed. `VIA-000` is preregistered first. The six
dependent Tier-R packets have frozen hypotheses, rules, budgets, and attacks but stay
in `drafted` lifecycle state until their machine-declared prerequisites pass. A green
`VIA-000` establishes reproducibility and evidence integrity only; it does not establish
the physical mechanism or Tier R.

## Current state

- Campaign decision: `pending`.
- `VIA-000`: preregistered; holdout not started. Clean falsifiers exposed an
  undeclared-import/untracked-file blind spot, ignored virtual-environment customize
  hooks, an executable `.pth` variant, and an editable self-cleaning `.pth`, recorded in
  [`attacks/VIA-000-ATTACK-PLAN-1.md`](attacks/VIA-000-ATTACK-PLAN-1.md) and
  [`attacks/VIA-000-ATTACK-PLAN-2.md`](attacks/VIA-000-ATTACK-PLAN-2.md), and
  [`attacks/VIA-000-ATTACK-PLAN-3.md`](attacks/VIA-000-ATTACK-PLAN-3.md), and
  [`attacks/VIA-000-ATTACK-PLAN-4.md`](attacks/VIA-000-ATTACK-PLAN-4.md).
  Pre-holdout protocol refreezes add all attacks and require a new clean falsifier
  before execution.
- `VIA-010`, `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300`, `VIA-400`: drafted.
- Sealed holdout/seed commitments exist outside the repository at the immutable URIs
  and SHA-256 values recorded in each packet. Their contents have not been used as
  calibration evidence.
- A builder-owned pre-freeze calibration is retained in
  [`calibration/VIA-000-2026-08-13.md`](calibration/VIA-000-2026-08-13.md). It is not a
  decisive campaign receipt and cannot satisfy the E3 evidence floor.
- A builder-owned Blackwell/native calibration is retained in
  [`calibration/VIA-300-Blackwell-2026-08-13.md`](calibration/VIA-300-Blackwell-2026-08-13.md).
  Its native test and benchmark evidence is superseded by the hardened remediation
  record in
  [`calibration/VIA-300-Blackwell-remediation-2026-08-17.md`](calibration/VIA-300-Blackwell-remediation-2026-08-17.md).
  The remediation confirms executable CUDA hardware and validated native `sm_120`
  code, but remains builder-owned calibration, not an independent VIA-300 receipt,
  and does not establish Tier R.

## Validation

After the protocol snapshot commit is recorded in `CAMPAIGN.yaml` and every packet,
the authoritative command is:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Only hashed receipts, reconciled independent review artifacts, custody reveal records,
and a validator-clean adjudication can move a packet or campaign to a terminal outcome.
