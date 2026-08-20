# VIA-000 R2 independent falsification attack plan 1

## Recommendation

**UNSAFE — DO NOT AUTHORIZE HOLDOUT.**

The frozen supporting runner cannot execute a clean Windows control with the
frozen `uv` prerequisite. It compares the complete `uv --version` banner with
the bare string `0.11.11`, so the correctly installed frozen version is
rejected before clone, environment sync, the 366 tests, the six examples,
mutation execution, artifact validation, or PDF construction.

Independently, a frozen authoritative-validator test demonstrates that a
summary-only document with pass/fail/capability Booleans can stand in for raw
results. It contains no platform records, command records, retained stdout or
stderr, test/example/mutation counts, source/environment manifests, visual
evidence, or PDF evidence, yet `validate_campaign(...)` returns no errors. That
is an accepted equivalent in both mutation family 17 (missing platform/raw
evidence) and mutation family 18 (result available only in a summary).

Either finding is sufficient under the frozen decision rule. This review does
not authorize a scientific outcome, lifecycle transition, or holdout start.

## Review identity and frozen inputs

- Campaign: `POPGP-VIABILITY-R2-2026-08`
- Packet: `VIA-000`
- Phase: pre-holdout
- Seat: `codex-via000-r2-falsifier`
- Session: `popgp-viability-r2-2026-08-via000-falsifier-session-1`
- Repository remote: `https://github.com/whact2025/POPGP`
- Activated handoff: `94bfe7eaae58753647a7bf8b9030c97e7fb18a58`
- Handoff tree: `2e1920aaef2790f4cfcd7705f04a40d596d54de6`
- Protocol snapshot: `da5dd8a45420bf20a075d867f06d61d79b4df8eb`
- Activation commit: `81a2ee5a41c0bd0013decac41b1255b1e8e5f20a`
- Scientific candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Failed R1 comparison baseline: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Review branch: `campaign/via000-r2-falsifier-1`
- Isolated review worktree: `C:/src/POPGP-via000-r2-falsifier-1`

The handoff resolved to the declared tree and parent:

```text
94bfe7eaae58753647a7bf8b9030c97e7fb18a58
tree 2e1920aaef2790f4cfcd7705f04a40d596d54de6
parent 81a2ee5a41c0bd0013decac41b1255b1e8e5f20a
```

The scientific-candidate disposable clone resolved to the declared commit and
tree and was clean before attack execution.

## Access declaration

I read only tracked public campaign, protocol, packet, governance, schema,
validator, checker, generator, test, and historical-review material needed to
construct and evaluate the attacks. I did not access custody material, sealed
manifests, seeds, hidden labels, private evaluator material, the untracked
handoff memo, or untracked builder/custody memos. Builder-authored tests,
reviews, and CI claims were treated as hypotheses and rerun or inspected rather
than accepted as authority. No scientific candidate source was repaired, and
no threshold, protocol, lifecycle field, or `holdout_started` field was
modified.

The tracked public materials read in full included the campaign and packet,
`PROTOCOL_MANIFEST.json`, both R2 protocol artifacts and their receipt copies,
the R2 launch brief, `VIABILITY_DEMONSTRATION_PLAN.md`, reviewer workflow and
identity, the relevant schemas and authoritative validator, reproduction and
validation checkers, generators and tests, and the seven public R1 attack plans
plus the relevant public R1/R2 reviews.

## Environments and exact controls

The attack clone, environment, caches, evidence, and PDF paths were all outside
the review worktree:

```text
C:/src/POPGP-via000-r2-falsifier-1-attacks/base
C:/src/POPGP-via000-r2-falsifier-1-attacks/environment-base
C:/src/POPGP-via000-r2-falsifier-1-validator-env
C:/src/POPGP-VIABILITY-R2-2026-08-WINDOWS
```

The disposable candidate clone was detached at
`5be3c38a0822d49953d0933f14ccab32ca12c896`, tree
`6ad387f9f4e0bab7f97df1bb54a03177887f0707`. A fresh locked non-editable
Python 3.11.15 environment installed 60 packages. Caches were external. Its
retained manifest digests were:

```text
environment manifest sha256 2c33ad5bfbbe3ed0c7de8009ac3c10f3dca5f705e6856f2afff404f3835778b0
source manifest sha256      0274b37ff4f75b8e96881113fed0139beaaca931d8b79e426187448cb1369027
```

The trusted reproduction-boundary checker was extracted from the candidate Git
object database outside the candidate checkout. The isolated semantic control
used the base interpreter with `-I -S` through
`scripts/run_without_startup_hooks.py` and exited 0 with:

```text
Validation artifact contracts and required visual outputs are valid.
```

Observed prerequisites were:

```text
uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
Python 3.12.10 (base interpreter)
Python 3.11.15 (locked attack and validator environments)
PowerShell 7.6.4
pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
```

## Clean Windows runner control

The exact frozen command was executed:

```powershell
pwsh -NoProfile -File protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1 -WorkspaceRoot C:/src/POPGP-VIABILITY-R2-2026-08-WINDOWS -PlatformFamily windows-x86_64
```

It exited 1 in approximately 1.6 seconds at runner line 114:

```text
expected uv 0.11.11, observed uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
```

The runner creates the workspace/evidence/cache/PDF directories before the
version check, but it did not clone the candidate or run environment sync, 366
tests, six examples, the artifact gate, mutation families, or the two-pass PDF
build. Therefore the clean Windows control is infeasible under the frozen
runner. The Ubuntu command cannot cure the missing required Windows result, and
the protocol requires both platform families.

## Mutation execution method

The following exact isolated command ran the semantic, visual, source,
environment, and repository-boundary mutation modules in the disposable exact
candidate environment:

```powershell
C:/src/POPGP-via000-r2-falsifier-1-attacks/environment-base/Scripts/python.exe -I -S -X pycache_prefix=C:/src/POPGP-via000-r2-falsifier-1-attacks/pycache scripts/run_without_startup_hooks.py --repo-root C:/src/POPGP-via000-r2-falsifier-1-attacks/base --module pytest -- -q -p no:cacheprovider tests/unit/test_validation_artifact_contract.py tests/unit/test_reproduction_boundary.py
```

Result:

```text
194 passed in 109.83s
```

This was not treated as proof by test name. The parametrized mutations and
their checker paths were inspected, and the covered families were broadened
across all affected records/arrays where the public modules already exposed
elementwise or configuration variants.

The receipt, output-commitment, reveal-chronology, campaign-binding, and
truth-table contract sweep used the same isolated bootstrap and these seven
targeted authoritative-validator tests:

```text
test_schema_contract_rejects_cross_field_and_receipt_mutations
test_structured_receipts_and_governance_provenance_fail_closed
test_protocol_content_and_budget_are_frozen_before_holdout
test_blind_custody_rejects_leaks_role_reuse_and_manifest_mutations
test_outcome_truth_table_is_deterministic
test_holdout_cannot_start_until_dependencies_pass
test_campaign_is_bound_to_frozen_git_and_protocol_content
```

Result:

```text
7 passed in 180.82s
```

## All 18 frozen mutation-family outcomes

`REJECTED` means the corrupt variant was refused by the relevant complete
public checker/boundary path. `ACCEPTED` means at least one coherent corrupt or
missing-evidence equivalent survived the authoritative campaign validator.

| # | Frozen family and independently executed variants | Outcome | Evidence |
|---:|---|---|---|
| 1 | Raw/derived semantic contradictions: unregistered and removed identities, pass flip, demotion and headline contradiction | REJECTED | Semantic recomputation and registered-check identity tests in the 194-test run failed closed. |
| 2 | Required array type/shape changes, non-finite values, and stale summaries | REJECTED | Type, shape, finiteness, and independently recomputed-summary variants failed closed. |
| 3 | Stable configuration and decision-threshold changes retaining prior Booleans | REJECTED | Configuration bindings and 25 decision-margin/threshold cases failed closed. |
| 4 | Localized visuals: compact feature, one-pixel change, dashed/thin curve, rendered text/label, real annotation, structured/local corruption | REJECTED | Calibrated pixel and structure variants, including removed/moved curves and annotation changes, failed closed. |
| 5 | Grid raw `phi`: every-element perturbations plus scale, sign, reverse, roll, shift, and uniform transforms retaining summaries | REJECTED | Lowest-level grid operands were recomputed; retained range/extrema/mean/pass summaries did not mask mutations. |
| 6 | Simple-source raw law operands: entropy, modular energy, and relative entropy, including elementwise and within-tolerance variants | REJECTED | Slope, affine, solver-ratio, and Boolean preservation did not bypass raw-operand recomputation. |
| 7 | Many-body raw law operands: local/global energy, KMS/first-law, endpoint/control, order sensitivity, aliases, and precision-floor variants | REJECTED | Local/global and control/order identities were independently recomputed and corrupt variants failed closed. |
| 8 | Graph weight provenance: diagonal, negative sign, asymmetry, non-edge/new edge, and correlated source variants across clock records | REJECTED | Graph support, symmetry, sign, diagonal, and independent-provenance checks rejected the variants even when residuals were preserved. |
| 9 | Mutual-information provenance: diagonal, asymmetry, weight changes, and coherent edge-support changes across clock records | REJECTED | Independent MI structure, kernel, symmetry, and edge-support checks rejected the variants. |
| 10 | Raw/effective source, removed background, chemical-potential/raw source term, zero-mode policy, and nonzero-sum/correlated transforms | REJECTED | Raw/effective/background identities, zero-mode rules, and zero-sum conditions were recomputed across the clock records. |
| 11 | Configured mass (`mu`), gauge/normalization, source-model/source binding, diagnostic center, and point strength with correlated downstream fields | REJECTED | Mass, normalization/gauge, center, strength, and raw-source/configuration variants reached the authoritative configuration/source checks and failed closed. |
| 12 | Root `sitecustomize.py`, `PYTHONPATH` and other blocked environment variables, including startup state intended to self-clean | REJECTED | Base-interpreter and blocked-environment boundary cases failed before trusted execution. |
| 13 | Ignored environment `sitecustomize.py`, `usercustomize.py`, executable `.pth`, editable and persistent/self-deleting/self-restoring startup state | REJECTED | `-I -S` bootstrap plus complete environment-manifest variants rejected persistent and self-cleaning carriers. |
| 14 | Modified installed dependency, executable bytecode/import cache, ignored cache, and post-snapshot environment substitution | REJECTED | Fresh locked environment-manifest comparison and importable-cache enumeration failed closed. |
| 15 | `assume-unchanged`, `skip-worktree`, staged/index state, filter concealment, ignored/untracked source, rename/delete, and symlink/mode substitution | REJECTED | Literal Git-object/index/mode boundary variants failed closed rather than trusting normal status output. |
| 16 | Transient tracked-source mutation restoring/deleting itself before final status | REJECTED | Per-command pre/post source-boundary checks caught the byte-restoring mutation before accepting command output. |
| 17 | Repository-local PDF/output residue, dirty generated artifacts, platform-only/missing platform evidence, and pre-existing evidence behavior | **ACCEPTED** | Boundary tests rejected local residue, but the authoritative campaign validator accepted a summary-only result with no Linux or Windows raw records, commands, counts, manifests, visuals, or PDF evidence. The clean Windows runner also aborted before producing evidence. |
| 18 | Summary-only result, raw-result replacement after commitment, reveal before commitment, and broken receipt/hash/chronology bindings | **ACCEPTED** | Targeted receipt/hash/chronology variants were rejected, but `test_outcome_truth_table_is_deterministic` supplies only summary Booleans/capabilities as “raw results” and still obtains `validate_campaign(...) == []`. One surviving variant makes the family unsafe. |

## Accepted summary-only equivalent

The frozen validator test helper writes a document structurally equivalent to:

```json
{
  "metric": 1,
  "passed": true,
  "failed": false,
  "blocked": false,
  "capabilities": {
    "<each required capability name>": true
  }
}
```

It omits the evidence required by the primary protocol: there are no separate
Linux and Windows raw results, exact candidate/tree records, commands, exits,
stdout/stderr, 366-test result, six generator results, 18 mutation results,
source/environment manifests, 12 visual results, raster comparisons, PDF
engine/passes/output, or pre-reveal output commitment. Nevertheless the frozen
truth-table test calls the authoritative validator and expects an empty error
list while using this document for packet raw results. Changing a leaf packet
outcome is also validated by the same summary-only fixture.

The packet schema binds receipt metadata such as id, kind, path, digest, and
media type, but does not schema-validate raw-result contents against the
complete runner/evidence contract. The validator derives the truth table from
typed summary Booleans and does not prove the frozen measurement procedure was
executed. Thus passing receipt hashes and chronology tests does not close the
accepted missing-evidence channel.

The supporting runner also does not itself aggregate the two platform results
into the complete packet raw-result/commitment/reveal contract. Even absent the
version bug, the frozen gate is split between prose/manual steps and a validator
that accepts less than the required retained evidence.

## Protocol, schema, and hash audit

The authoritative campaign validator accepted the tracked campaign before this
review artifact was added:

```text
Viability campaign contract is valid.
```

The frozen packet-rule SHA-256 recomputed as:

```text
d47049565cd5ef7f4f5baf7baf6e5bb619352608412f1def87f5b55562492598
```

The requirements blob at the protocol snapshot recomputed as:

```text
632528e8c4b19d746253719e308b3a676b5a19cffc3a734a670d1c878c161d20
```

All ten protocol-manifest `git-blob-sha256` bindings matched the declared
values:

| Bound artifact | Recomputed SHA-256 |
|---|---|
| Claims matrix | `33d0e656131b6aea52c3a644f59f91c302b777014839ceb13e5f6f93758e8e6d` |
| Gate registry | `7af3e9dec3a172831c34a26bc28f496b38857f7499dc04440d773b0a7726fbb2` |
| Campaign schema | `13ba43394a9ae4162d47fc44662232e9c51b73d3c0a32515ab81562f3007fbca` |
| Independent re-review schema | `8b0a6e224abbf09a6b33890acdb628c15c82d083d2ae8bf7273ee2a26d01e567` |
| Independent review schema | `032eedcdd5651bc8deaedf06ec20805aec028eaafad8ba0eea9d8acdcc76e8a1` |
| Packet schema | `9278b165eafcab04bd0989d489db3fe4fe881ade0c04e7f77cad5e9d69f5217d` |
| Primary protocol schema | `d3360469bf7eb5b501da9eb42350df17fa946d5c9e598b20534240fcb7d8f801` |
| Protocol manifest schema | `78f3c6a0dcf59a7ed134e5df2743cc7b9cdace8cc7b312af171a8c9e0381f10d` |
| Review-response schema | `0719e843fced8109e37f522e538be9905b7f43fcb42a06a3eb50239258785a1d` |
| Authoritative campaign validator | `f3cebec20a326db062351fb25a01f01ad36bd96284262ea354f8d69349d21298` |

Each binding was compared as the complete 64-hex value by the public
`--git-blob-sha256` audit.
The two copies of each R2 protocol artifact were byte-identical:

```text
primary protocol and receipt copy sha256 0af64cbc713790d10243feee5fad590c6063027fb771633c49045206b51388fb
runner and receipt copy sha256           b627852b880ffaaa4e716ddb90591fd0701fe0afc9e58c29f4b0ffcf44cb1d8a
protocol manifest checkout sha256        0cf25a4f4f9004b897238f6366cfe7d690f50021784a852258154bff7b6b3391
```

The protocol snapshot resolved to tree
`f003fd438bfae9a67c26645d0616322cb8aa43b1` with parent
`f7bffc98d851ad2c333c5b01e4221b655908de7e`; activation resolved to tree
`f66d9fd2525161a460ba581f5e7cdb5c302a9252` with parent
`da5dd8a45420bf20a075d867f06d61d79b4df8eb`.

## Limitations and non-results

- This seat had a Windows host only. No Linux control was represented as run.
- The exact Windows runner aborted at its frozen `uv` check, so this review
  cannot report a runner-produced 366-test, six-example, 18-mutation, visual,
  artifact-boundary, or two-pass PDF result.
- An attempted run of the entire
  `tests/unit/test_viability_campaign_contract.py` exceeded a 604.1-second tool
  limit and was terminated; it is a non-result. The seven frozen tests directly
  relevant to families 17/18 were then run explicitly and all passed.
- No holdout receipts or private outcome material were accessed. No scientific
  result is inferred from public implementation artifacts or attack outcomes.
- The 194 public checker/boundary tests establish rejection only for the exact
  variants inspected and executed. They do not repair the accepted summary-only
  evidence channel or the infeasible clean runner.

## Final decision

The clean-control requirement is not met, and at least one mutation survives in
each of frozen families 17 and 18. Under the frozen primary protocol, the only
permitted recommendation is:

**UNSAFE — DO NOT AUTHORIZE HOLDOUT.**

`lifecycle` and `holdout_started` must remain unchanged.
