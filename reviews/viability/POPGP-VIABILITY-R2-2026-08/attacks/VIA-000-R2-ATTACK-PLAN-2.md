# VIA-000 R2 post-refreeze independent falsification attack plan 2

## Recommendation

**SAFE for the frozen pre-holdout falsifier gate.**

The exact refrozen Windows clean control executed to completion, all 366 tests and six
generators passed, the semantic/visual/source/environment/PDF boundary completed, and
the final candidate checkout was clean in both normal and ignored-inclusive status.
The separately frozen mutation runner then reconciled 145 observed parametrized test
nodes against the exact registered selector/count contract and derived 18 complete
family receipts. Every family was rejected; no corrupt or missing-evidence equivalent
was accepted in this pass.

This recommendation authorizes no lifecycle edit, holdout start, private-data access,
runner output commitment, reveal, adjudication, or scientific conclusion. It says only
that the refrozen public protocol passed the required fresh falsifier gate. The future
reproduction-runner gate still requires complete, independently attested Ubuntu and
Windows platform fragments and the frozen assembler/validator sequence.

## Frozen identity and review boundary

- Campaign: `POPGP-VIABILITY-R2-2026-08`
- Packet: `VIA-000`
- Phase observed: `preregistered`
- `holdout_started` observed: `false`
- Falsifier seat: `codex-via000-r2-falsifier`
- Session: `popgp-viability-r2-2026-08-via000-falsifier-session-1`
- Model identity/version recorded by the frozen packet: `unknown` / `unknown`
- Operator: `fuocor`
- Orchestrator: `codex-desktop`
- Access level: `public-calibration-only`
- Repository remote: `https://github.com/whact2025/POPGP`
- Reactivation handoff: `4e98bb9e943d14d7847bd3fda949efe42d52bd0d`
- Handoff tree: `9caf2e307473eb4f3fc1074e7afef9304a1b310f`
- Activation commit: `3bf74151b8f848ab8b6d44bcf84946ab6a890bde`
- Immutable protocol snapshot: `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`
- Scientific candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Failed R1 comparison baseline: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Isolated worktree: `C:/src/POPGP-via000-r2-falsifier-2`
- Branch: `campaign/via000-r2-falsifier-2`

The Git identities were resolved directly:

```text
protocol=9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9
protocol_tree=6a2431f3dd15e1ee0e0427d2ff34773b6677fd51
protocol_parent=610701ae04d1ae644da62814a71765b4ba8c0ede

activation=3bf74151b8f848ab8b6d44bcf84946ab6a890bde
activation_tree=d0a0ffea0456e4ccaf9d934b63fb2d857fcdea35
activation_parent=9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9

handoff=4e98bb9e943d14d7847bd3fda949efe42d52bd0d
handoff_tree=9caf2e307473eb4f3fc1074e7afef9304a1b310f
handoff_parent=3bf74151b8f848ab8b6d44bcf84946ab6a890bde
```

## Access and independence declaration

I used only tracked public campaign, packet, protocol, governance, schema, validator,
checker, test, amendment, and historical attack/review material. Prior tests, reviews,
hosted calibration, and builder claims were treated as attack hypotheses, not as
evidence for this recommendation. The evidence below comes from this fresh worktree
and this seat's fresh executions.

I did not access custody files, hidden or sealed manifests, final labels, secret seeds,
private evaluator material, campaign/output raw results, the untracked handoff memo,
or any untracked builder/custody memo. No scientific candidate, protocol, checker,
threshold, packet, campaign, lifecycle, or holdout field was changed. No prior output
was imported into this pass.

```yaml
exposure:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
external_scientific_validation: false
```

The tracked public material read included the campaign and VIA-000 packet; all five
VIA-000 protocol artifacts and their tracked receipt copies; all four activated
pre-holdout amendments; the execution brief; protocol manifest; demonstration plan;
reviewer workflow and identity rules; reproduction, semantic, raw-evidence, assembler,
campaign, schema, generator, and validator code/tests; Plan 1; and the public amendment
review chain needed to reconstruct earlier counterexamples.

## Freeze and validator audit

The authoritative public validator, invoked from a fresh locked non-editable Python
3.11.15 environment, returned:

```powershell
uv run --isolated --frozen --no-editable python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml
```

```text
Viability campaign contract is valid.
```

The first file-path form (`python scripts/check_viability_campaign.py`) was a command
invocation non-result under `--no-editable`: it failed closed because that form places
`scripts/` rather than the repository root on `sys.path`. The module form above is the
successful authoritative result; no code or environment was repaired.

The VIA-000 packet-rule SHA-256 matched exactly:

```text
98bbe33c2c5f2a2a321675d12b47e074e83f92ee3463a131f398c81c21d3367b
```

The requirements and protocol-manifest Git-blob SHA-256 values at the immutable
protocol snapshot matched the campaign:

```text
requirements-v2.json 632528e8c4b19d746253719e308b3a676b5a19cffc3a734a670d1c878c161d20
PROTOCOL_MANIFEST.json d30752abc304f0ecf55544d8584a0a5fb739a11e26af845b925f60c94117b2ad
```

All ten manifest contract bindings matched their exact Git blobs:

| Contract file | Git-blob SHA-256 |
|---|---|
| `docs/scientific_hardening/CLAIMS_MATRIX.md` | `33d0e656131b6aea52c3a644f59f91c302b777014839ceb13e5f6f93758e8e6d` |
| `docs/scientific_hardening/GATE_TEST_REGISTRY.md` | `7af3e9dec3a172831c34a26bc28f496b38857f7499dc04440d773b0a7726fbb2` |
| `schemas/viability/campaign-v2.schema.json` | `13ba43394a9ae4162d47fc44662232e9c51b73d3c0a32515ab81562f3007fbca` |
| `schemas/viability/independent-rereview-v2.schema.json` | `8b0a6e224abbf09a6b33890acdb628c15c82d083d2ae8bf7273ee2a26d01e567` |
| `schemas/viability/independent-review-v2.schema.json` | `032eedcdd5651bc8deaedf06ec20805aec028eaafad8ba0eea9d8acdcc76e8a1` |
| `schemas/viability/packet-v2.schema.json` | `9278b165eafcab04bd0989d489db3fe4fe881ade0c04e7f77cad5e9d69f5217d` |
| `schemas/viability/primary-protocol-v1.schema.json` | `d3360469bf7eb5b501da9eb42350df17fa946d5c9e598b20534240fcb7d8f801` |
| `schemas/viability/protocol-manifest-v2.schema.json` | `78f3c6a0dcf59a7ed134e5df2743cc7b9cdace8cc7b312af171a8c9e0381f10d` |
| `schemas/viability/review-response-v2.schema.json` | `0719e843fced8109e37f522e538be9905b7f43fcb42a06a3eb50239258785a1d` |
| `scripts/check_viability_campaign.py` | `9b529c287454b929e4c63aa33f25bf9171f9836ff346ff72cfc2bb5056b78db9` |

Each frozen protocol artifact matched its tracked receipt copy byte-for-byte at the
protocol snapshot:

| Artifact | Git-blob SHA-256 |
|---|---|
| Primary protocol | `9bca5f0b852627e588591e133820232be86e84e827b126b299b7e927bbfb1131` |
| Clean runner | `a7c50da6f3e372cfeb9d2d359618bf405e622daa3733a03398aceba0c7a90e49` |
| Mutation runner | `159493f53921c324c43ec76d96731841ed71e65fb5fcedb99fc016a9d4536b0e` |
| Assembler | `c0a186d20d1d4b9947ad06220fb27a27ee9ea4f595ad4d63efda77b919bdfbc7` |
| Raw-results schema | `aae1720287ed077ea911e4ac79fd9fd755f7139a599df6f851ed58346d924899` |

## Exact clean control

The prior Plan-1 aborted scratch directory at the frozen Windows path was moved intact
to `C:/src/POPGP-VIABILITY-R2-2026-08-WINDOWS-PLAN1-INVALID`, making the exact frozen
workspace path absent without treating prior output as input. The following exact
command was then executed from the fresh Plan-2 worktree:

```powershell
$env:POPGP_PROTOCOL_SOURCE_COMMIT='9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9'
pwsh -NoProfile -File protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1 -WorkspaceRoot C:/src/POPGP-VIABILITY-R2-2026-08-WINDOWS -PlatformFamily windows-x86_64 -ProtocolSourceCommit $env:POPGP_PROTOCOL_SOURCE_COMMIT
```

Outcome: exit 0 in 1,414 seconds. The Plan-1 `uv` banner counterexample is repaired at
the exact boundary: the runner accepted
`uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)` by parsing and binding the
semantic-version field while retaining the complete banner.

The retained candidate identities were exact:

```text
candidate_commit=5be3c38a0822d49953d0933f14ccab32ca12c896
candidate_tree=6ad387f9f4e0bab7f97df1bb54a03177887f0707
protocol_source_commit=9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9
platform_family=windows-x86_64
```

The retained aggregate control was:

```text
366 passed in 1091.53s (0:18:11)
test_count=366
example_count=6
visual_count=12
command_count=16
nonzero_command_count=0
commands_passed=true
semantic_contract_passed=true
visual_contract_passed=true
source_boundary_passed=true
environment_boundary_passed=true
pdf_passed=true
```

All six required generators ran once. The artifact boundary passed after copying the
18 required generated artifacts (12 visuals), the generated paths were restored from
the exact candidate, and the final candidate state had zero normal and zero
ignored-inclusive entries.

The retained boundary/PDF identities were:

```text
environment_manifest_sha256=8291d31fd890fbd93e3b72fa00fc0b44adb5244917510d4f948c11548d3e72c9
source_manifest_sha256=88806215a7d6678383a38f18cb8fdd5ea70d80e5972a3eb9a120d786c419e8ca
pdf_engine=pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
pdf_sha256=fe8ac59aadf05862c4ac7a6aae710afa4cc9d5be1d1cf78a2c4d849ec14815e0
pdf_byte_count=535368
pdf_page_count=11
final_status_byte_count=0
```

The complete command ledger was zero-exit:

| ID | Frozen executable contract | Exit | Seconds |
|---|---|---:|---:|
| `001-clone` | `git-clone` | 0 | 2.032 |
| `002-checkout` | `git-checkout` | 0 | 0.638 |
| `003-sync` | `uv-sync-frozen-no-editable` | 0 | 29.031 |
| `004-ruff` | `trusted-python-ruff` | 0 | 21.674 |
| `005-check-tex` | `trusted-python-check-tex` | 0 | 20.356 |
| `006-pytest` | `trusted-python-pytest` | 0 | 1114.831 |
| `007-chain` | `trusted-python-chain-generator` | 0 | 46.260 |
| `008-grid` | `trusted-python-grid-generator` | 0 | 22.552 |
| `009-gravity` | `trusted-python-gravity-generator` | 0 | 23.742 |
| `010-source-law` | `trusted-python-source-law-generator` | 0 | 25.439 |
| `011-many-body` | `trusted-python-many-body-generator` | 0 | 24.856 |
| `012-ca` | `trusted-python-ca-generator` | 0 | 24.315 |
| `013-artifact-boundary` | `trusted-python-artifact-boundary` | 0 | 26.570 |
| `014-pdflatex-1` | `pdflatex-pass-1` | 0 | 4.412 |
| `015-pdflatex-2` | `pdflatex-pass-2` | 0 | 1.368 |
| `016-environment-verify` | `trusted-python-environment-verify` | 0 | 8.250 |

## Exact frozen mutation run

The separately frozen command was executed against the clean-control workspace:

```powershell
$env:POPGP_PROTOCOL_SOURCE_COMMIT='9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9'
python -I -S protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-MUTATION-RUNNER.py --workspace-root C:/src/POPGP-VIABILITY-R2-2026-08-WINDOWS --platform-family windows-x86_64 --protocol-source-commit $env:POPGP_PROTOCOL_SOURCE_COMMIT
```

Outcome: exit 0 in 706.6 seconds. The runner itself executed the exact frozen
`uv run --isolated --frozen --no-editable python -m pytest -vv -p no:cacheprovider`
selector list. Its retained execution window and hashes were:

```text
started_at=2026-08-21T05:20:06.055573Z
finished_at=2026-08-21T05:31:52.003034Z
mutation_suite_exit_code=0
mutation_suite_stdout_sha256=83244c1e7bf6ab6f378353edb2360a48bf0e4910c1f5fc5c3dd9f79d25e5407d
mutation_suite_stderr_sha256=82cf0007cae578114429e28d597ae7a4990b7be405473dd2853e36d031685304
mutation_suite_result_sha256=599c9a74694e11f7ac1be453ca829bf5e8eb82eaac9daeb5ba91e4777a6723de
unique_passed_nodes=145
mutation_receipts=18
mutation_count=18
mutations_rejected=true
overall_passed=true
```

## All 18 frozen mutation-family outcomes

`REJECTED` means the registered corrupt variants reached the frozen oracle and the
exact expected parametrized node count passed. The complete 366-test control also ran
the broadened raw-evidence, assembler, platform, portability, tolerance, status,
custody, chronology, and freeze controls that surround the registered selectors.

| Family | Oracle | Passed nodes | Outcome | Executed mutations and broadened equivalents |
|---:|---|---:|---|---|
| 1 | `semantic-check-identity` | 1 | **REJECTED** | Raw/derived contradiction and pass-Boolean flip; the full suite also covered removed, demoted, headline, and unregistered check identities. |
| 2 | `semantic-type-shape-finiteness` | 7 | **REJECTED** | Required array shape/type, non-finite diagnostic, malformed raw clock solver operands, and stale derived summaries. |
| 3 | `semantic-config-threshold` | 1 | **REJECTED** | Stable configuration change retaining a prior Boolean; the full suite recomputed every decision-bearing float margin. |
| 4 | `visual-locality` | 9 | **REJECTED** | Localized structured corruption, compact/one-pixel feature changes, dashed/thin curve movement/removal, rendered label/text, and real annotation removal. |
| 5 | `grid-raw-phi` | 8 | **REJECTED** | Threshold crossing, raw-grid scale/sign/reverse/roll/shift, correlated uniform shift, and correlated scale with stale summaries/residuals. |
| 6 | `simple-source-raw` | 5 | **REJECTED** | Simple-source entropy, modular-energy, relative-entropy, slope, affine, and solver-ratio raw operands. |
| 7 | `many-body-raw` | 14 | **REJECTED** | Many-body local/global energy, KMS/first-law, endpoint/control, order aliases, all decision-bearing raw elements, and correlated global shift. |
| 8 | `clock-graph-weight` | 24 | **REJECTED** | Weight diagonal, sign, asymmetry, new/non-edge support, and correlated source updates across all clock records. |
| 9 | `clock-mutual-information` | 16 | **REJECTED** | MI diagonal/asymmetry/weight and coherent edge-support changes across all clock records. |
| 10 | `clock-source-provenance` | 28 | **REJECTED** | Raw/effective source, removed background, zero-mode/zero-sum policy, normalization and correlated finite-graph transforms. |
| 11 | `clock-config-source` | 3 | **REJECTED** | Diagnostic raw source, configured point strength, and center; mass, gauge/normalization and source-model consistency were traversed by the bound clock-source checks. |
| 12 | `startup-root-environment` | 1 | **REJECTED** | Root `sitecustomize.py`, `PYTHONPATH` and blocked startup environment, including self-cleaning state at snapshot creation. |
| 13 | `startup-environment-hooks` | 10 | **REJECTED** | Ignored environment `sitecustomize.py`, `usercustomize.py`, executable `.pth`, and persistent/self-delete/byte-restoring carriers under the isolated bootstrap. |
| 14 | `environment-dependency-bytecode` | 2 | **REJECTED** | Installed dependency modification and ignored executable bytecode/import-cache substitution. |
| 15 | `repository-index-byte-boundary` | 5 | **REJECTED** | `assume-unchanged`, `skip-worktree`, clean filters, staged/ignored/delete/rename states, and symlink/mode substitution against literal Git bytes. |
| 16 | `transient-source-boundary` | 1 | **REJECTED** | Byte-restoring tracked source mutation before command execution/final status. |
| 17 | `platform-evidence-pdf-boundary` | 7 | **REJECTED** | Missing/failed platform fragment, semantic-before-commit, producer-attestation failure, Linux typed symlinks and bounded expanded-node parsing, Windows `core.autocrlf=false` clone contract, strict PDF parsing, and opposed Ubuntu/Windows raster drift. The complete suite also covered calibrated numeric tolerance, empty status separators, repository-local residue and exact platform/command/identity contracts. |
| 18 | `receipt-commitment-chronology` | 3 | **REJECTED** | Canonical assembler commitment/custody round trip, structured receipt/governance chronology, blind-custody role/hash/reveal mutations. The complete suite also rejected Plan-1 summary-only results, coherent fake/stale evidence, raw blockage, missing contracts, post-hash byte changes, invalid manifest/freeze bindings, partial assembly, and commitment-before-semantic-gate violations. |

## Prior Plan-1 counterexamples and broadened contract controls

The two Plan-1 blockers were replayed at their amended boundaries:

1. **Full `uv` banner:** accepted correctly by the exact clean runner while retaining
   exact semantic version `0.11.11`; the complete control then finished.
2. **Summary-only “raw results”:** the complete 366-test control executed
   `test_raw_evidence_contract_rejects_dummy_or_stale_results`; the minimal summary
   fails the frozen Draft 2020-12 schema before its Booleans can drive an outcome.

The broadened evidence/transport controls were also present in the executed complete
suite and registered family-17/18 matrix:

- a coherent internally rehashed package is invalid without verified producer
  attestation, and attestation failure propagates to capability failure;
- top-level/per-platform commit and tree, platform set, command ID/map/count, blockage,
  artifact roles, source/environment manifests and mutation details are independently
  reconciled rather than accepted from summary fields;
- assembler inputs missing a platform or containing a failed command produce nonzero
  exit and no output directory/commitment;
- invalid pseudo-PDF bytes and platform-opposed rasters are rejected by independent
  parsing/direct comparison;
- semantic validation occurs before the assembler atomically emits the raw result and
  canonical custody commitment;
- receipt/output commitment/reveal ordering, hashes, roles and identities fail closed;
  and
- packet rule, requirements, manifest contract blobs, all five protocol/receipt pairs,
  protocol/activation/handoff ancestry, and the authoritative validator hash matched
  the refrozen public identities.

## Exposure boundaries and limitations

- This falsifier host was Windows x86-64. The exact Windows clean command executed;
  no local Ubuntu runner was available and the Ubuntu clean command was not
  represented as executed here. The registered Linux symlink/node-boundary and direct
  cross-platform controls did execute. A real campaign result must still supply both
  externally attested platform fragments.
- This seat did not access existing GitHub platform fragments, campaign raw results,
  output commitments, or revealed material. It therefore did not perform a live
  Sigstore verification or assemble a real two-platform campaign package. The
  registered producer-attestation negative control passed, and the verifier/assembler
  command bindings were audited in the frozen public source. Live attestation and
  two-platform assembly remain mandatory reproduction-runner gates.
- Structural unit fixtures replace the external attestation primitive so internal
  typed/semantic closure can be attacked deterministically. They do not establish an
  independent trust root; the protocol explicitly trusts the exact GitHub Actions
  control plane and excludes malicious/colluding trusted principals.
- The clean full-suite run emits quiet pytest output, while exact per-node proof comes
  from the separately retained verbose mutation suite. Broadened controls outside the
  registered 145-node subset are supported by the complete 366-test exit plus source
  inspection, not by separate family receipts.
- No native/scalable/CUDA claim was in scope. No scientific or hidden-holdout result
  is inferred from these public falsifier controls.

## Final gate decision

The exact refrozen clean control completed, every registered mutation receipt is
complete, every broadened public counterexample exercised in this pass failed closed,
and no surviving mutation was found.

**SAFE for the frozen pre-holdout falsifier gate.**

This artifact intentionally stops before lifecycle or `holdout_started` transition.
