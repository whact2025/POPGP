# VIA-000 public attack plan 1

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-1`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` / `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this as a Codex agent based on
  GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign-assigned operator; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The model fields are not inferred from the seat label. `unknown` is retained because
the exact runtime identifier and version were not exposed.

## Frozen identities verified before attack construction

- Activated handoff / attack-plan parent:
  `197219bbb006b4cd1f8b9f992cadedae8d09a341`
- Activated-handoff tree: `96b5d54277bf2ffa82cc297a012870f40c9eb932`
- Activated-handoff parent: `792d2d2737dcf48cf14cbb3af78a64ba2daad1f6`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Protocol snapshot: `792d2d2737dcf48cf14cbb3af78a64ba2daad1f6`
- Frozen VIA-000 rule SHA-256:
  `7f888d469d840d9c04c81b4f3f6276d9a9f57b6859ffe447febb2d8b0f0f1f2a`
- Primary-protocol raw-byte SHA-256:
  `cd55611ffa11e16ded3a336eeb3a6fa07b5136e43e3742d9db13a152c9bb7e44`
- Worktree / branch: `C:\src\POPGP-via000-falsifier-1` /
  `campaign/via000-falsifier-1`

The handoff, candidate, candidate tree, and protocol were resolved from Git objects,
not accepted from branch names.

## Access, independence, and prereveal declarations

- `final_labels_seen: false`
- `secret_seed_seen: false`
- `private_evaluator_seen: false`
- Custody repository/path accessed: `false`
- Sealed manifest bytes or contents accessed: `false`
- `C:\src\POPGP\POPGP_Codex_Handoff.md` accessed: `false`
- Builder calibration memo contents accessed: `false`
- Builder/reviewer conclusions received before hypotheses: `false`
- Builder/reviewer conclusions used as evidence: `false`
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable` because the packet records the exact
  builder and falsifier model identities as `unknown`.
- External scientific validation: `false`

I read the required public launch, campaign, packet, protocol, plan, governance, and
both validator files completely before choosing attacks. A later directory listing
showed the public directory name `calibration`; no file in it and no memo content or
conclusion was opened. No holdout or seed is requested or inferred.

## Disposable execution harness and cleanup

Run attacks only in a new exact-candidate clone outside the frozen worktree:

```powershell
$D = 'C:\src\POPGP-via000-disposable'
if (Test-Path -LiteralPath $D) { throw "refusing existing path: $D" }
git clone --quiet --no-hardlinks 'C:\src\POPGP' $D
git -C $D checkout --quiet --detach 9a29e05f803666bf0e3a28417ea399e3e26769fc
if ((git -C $D rev-parse HEAD) -ne '9a29e05f803666bf0e3a28417ea399e3e26769fc') {
  throw 'wrong disposable candidate'
}
function Edit-Json([string]$Path, [string]$Code) {
@'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding="utf-8"))
exec(sys.argv[2], {"d": d})
p.write_text(json.dumps(d, indent=2, sort_keys=False)+"\n", encoding="utf-8")
'@ | python - $Path $Code
}
```

For every tracked target below, cleanup is the exact command
`git -C $D restore --source=9a29e05f803666bf0e3a28417ea399e3e26769fc --staged --worktree -- <target>`.
For a future receipt, create `$R.attack-backup` before mutation, restore it with
`Move-Item -Force "$R.attack-backup" $R`, and recompute its receipt hash. Remove only
explicitly named untracked attack files. Before recursively deleting a whole clone,
resolve its path and require exact equality with `$D`; never target a source/workspace
root, frozen worktree, or custody path.

`$R` below means the hash-registered receipt with ID `raw-results` (the launch brief
names `raw-results.json`). If the runner chooses another path, resolve `$R` from that
receipt entry without changing its ID. `/platforms`, `/commands`, `/counts`, and
`/pdf` are required evidence targets even though no machine schema for them is frozen
today; absence of those records is itself attack `VIA000-A15`.

## Frozen mutation-family attacks

### VIA000-A01 — add an unregistered noninformational check

- Target: `examples/physics_qg/source_law/results/validation.json:$.checks`; append
  `{"name":"unregistered_required_probe","passed":true}`.
- Command: `Edit-Json $p 'd["checks"].append({"name":"unregistered_required_probe","passed":True})'`,
  where `$p=Join-Path $D 'examples/physics_qg/source_law/results/validation.json'`, then
  run `python scripts/check_validation_artifacts.py` from `$D`.
- Expected rejection: exit `1`, changed check-array length and unregistered
  noninformational identity.
- Lethal if accepted: arbitrary green identities can enter evidence without explicit
  registration.
- Cleanup: restore the exact target JSON from the candidate.
- Qualification result: rejected with both expected diagnostics.

### VIA000-A02 — flip a required check outcome

- Target: `source_law/results/validation.json:$.checks[0].passed`, check
  `relative_entropy_is_quadratic`, from `true` to `false` while headline stays true.
- Command: `Edit-Json $p 'assert d["checks"][0]["name"]=="relative_entropy_is_quadratic"; d["checks"][0]["passed"]=False'`,
  then run the artifact checker.
- Expected rejection: exit `1`, exact pass drift and `overall_pass=True` inconsistent
  with the noninformational conjunction.
- Lethal if accepted: a failed required gate survives under a green headline.
- Cleanup: restore the exact target JSON.
- Qualification result: rejected as expected.

### VIA000-A03 — change a measurement array shape

- Target: `examples/physics_qg/source_law/results/validation.json:`
  `$.measurements.relative_entropy`, frozen length `16`.
- Command: set `$p` to that JSON and run
  `Edit-Json $p 'assert len(d["measurements"]["relative_entropy"])==16; d["measurements"]["relative_entropy"].pop()'`,
  then the artifact checker.
- Expected rejection: exit `1`, measurement-array length `16 -> 15` at the exact path.
- Lethal if accepted: dimensional corruption retains an apparently numeric payload.
- Cleanup: restore the exact target JSON.
- Qualification result: rejected as expected.

### VIA000-A04 — change stable configuration beyond tolerance

- Target: `chain_1d/results/validation.json:$.config.beta`, `1.0 -> 1.01`.
- Command: `Edit-Json $p 'd["config"]["beta"]=1.01'`, then the artifact checker.
- Expected rejection: exit `1`, stable-input drift beyond
  `rel_tol=1.77636e-15, abs_tol=0`.
- Lethal if accepted: evidence silently uses a different physical configuration.
- Cleanup: restore the exact target JSON.
- Qualification result: rejected as expected.

### VIA000-A05 — make a required visual empty and untracked

- Target: declared tracked visual
  `examples/physics_qg/source_law/results/source_scaling.png`.
- Command:

```powershell
$v='examples/physics_qg/source_law/results/source_scaling.png'
git -C $D rm --cached --quiet -- $v
[System.IO.File]::WriteAllBytes((Join-Path $D $v), [byte[]]@())
Push-Location $D; python scripts/check_validation_artifacts.py; Pop-Location
```

- Expected rejection: exit `1` with both `not tracked by git` and `empty`.
- Lethal if accepted: a required visual has neither immutable provenance nor content.
- Cleanup: restore this exact target with `--staged --worktree`.
- Qualification result: both defects rejected.

### VIA000-A06 — move a sensitive diagnostic beyond its frozen threshold

- Target: `source_law_many_body/results/validation.json`, both
  `$.measurements.quadratic_coefficient_relative_error` and
  `$.checks[0].value.quadratic_coefficient_relative_error`; frozen value
  `3.812167720149195e-08`, frozen scientific maximum `0.0005`.
- Command: set both fields to `0.001` with
  `Edit-Json $p 'd["measurements"]["quadratic_coefficient_relative_error"]=0.001; d["checks"][0]["value"]["quadratic_coefficient_relative_error"]=0.001'`,
  then the artifact checker.
- Expected rejection: exit `1` at both paths; named diagnostic `abs_tol=0.0002` is
  exceeded and the frozen scientific maximum is violated.
- Lethal if accepted: a check remains true while its decisive diagnostic fails.
- Cleanup: restore the exact target JSON.
- Qualification result: both paths rejected.

## Boundary and evidence attacks

### VIA000-A07 — undeclared executable state / extra residue

- Target: disposable root `sitecustomize.py` plus undeclared `PYTHONPATH=.`.
- Command:

```powershell
$shim=Join-Path $D 'sitecustomize.py'
Set-Content -LiteralPath $shim -NoNewline -Value "import os`nos.environ['POPGP_UNDECLARED_STATE']='active'`n"
$prior=$env:PYTHONPATH
try {
  $env:PYTHONPATH='.'; Push-Location $D
  python -c "import os; print(os.environ.get('POPGP_UNDECLARED_STATE'))"
  git diff --exit-code
  python scripts/check_validation_artifacts.py
  git status --porcelain=v1 --untracked-files=all
  Pop-Location
} finally { $env:PYTHONPATH=$prior }
```

- Expected rejection: runner environment/residue audit rejects the untracked
  executable and undeclared import path; `evidence-contract` cannot be true.
- Lethal if accepted: the shim prints `active` while frozen cleanliness and artifact
  commands return `0` and a passing receipt is allowed.
- Cleanup: restore `PYTHONPATH`, remove exactly `$shim`, then require empty full
  porcelain status.
- Independently reproduced accepted mutation: `active` was observed while
  `git diff --exit-code` and the artifact checker both returned `0`; only full
  `git status` exposed `?? sitecustomize.py`. This is a public cleanliness blind spot,
  not a packet/campaign outcome.

### VIA000-A08 — Linux-only or Windows-only success

- Target: `$R:/platforms/windows-x86_64` (and symmetric Linux variant) while
  `/capabilities/cross-platform-reproduction=true`.
- Command: back up `$R`, then
  `Edit-Json $R 'd.setdefault("platforms",{}).pop("windows-x86_64",None); d["capabilities"]["cross-platform-reproduction"]=True'`.
- Expected rejection: missing/nonzero platform raw evidence makes the capability false
  or the run failed; adjudication must withhold a pass.
- Lethal if accepted: a summary Boolean passes with only one OS represented.
- Cleanup: restore `$R.attack-backup` and its SHA-256.

### VIA000-A09 — semantic drift preserving JSON shape

- Target: many-body JSON, both
  `$.measurements.exact_kubo_mori_quadratic_coefficient` and its duplicate under
  `$.checks[0].value`; preserve every key/type/array length but reverse the sign.
- Command: `Edit-Json $p 'd["measurements"]["exact_kubo_mori_quadratic_coefficient"]*=-1.0; d["checks"][0]["value"]["exact_kubo_mori_quadratic_coefficient"]*=-1.0'`,
  then the artifact checker.
- Expected rejection: numeric semantic drift at both paths.
- Lethal if accepted: shape/type-only checking accepts sign-reversed science under an
  unchanged green check.
- Cleanup: restore the exact many-body JSON.

### VIA000-A10 — PDF command failure

- Target: Windows hashed run log and `$R:/pdf/command`, `/pdf/exit_code`; frozen command
  is `pdflatex -interaction=nonstopmode -halt-on-error docs/framework.tex`.
- Command: from `$D`, run
  `pdflatex -interaction=nonstopmode -halt-on-error docs/does-not-exist.tex *> via000-pdf-command-attack.log`
  and require `$LASTEXITCODE -ne 0`.
- Expected rejection: nonzero command; evidence contract false; hashed log preserves
  command, stdout, stderr, and status.
- Lethal if accepted: absent/replaced/failing PDF command is hidden by a summary pass.
- Cleanup: remove only the named attack log and TeX residue in the disposable clone.

### VIA000-A11 — wrong PDF pass count

- Target: `$R:/pdf/passes` and `/pdf/pass_count`, expected exactly two executions.
- Command: back up `$R`, then
  `Edit-Json $R 'd["pdf"]["passes"]=d["pdf"]["passes"][:1]; d["pdf"]["pass_count"]=1'`.
- Expected rejection: `1 != 2`; PDF/evidence capability false.
- Lethal if accepted: a one-pass or summary-only build is accepted.
- Cleanup: restore `$R.attack-backup` and hash.

### VIA000-A12 — PDF output or metadata corruption

- Target: disposable `docs/framework.pdf` and `$R:/pdf/{size_bytes,sha256,engine,metadata}`;
  frozen engine `pdfTeX-1.40.29-TeX-Live-2026`.
- Command:

```powershell
$pdf=Join-Path $D 'docs/framework.pdf'
Copy-Item -LiteralPath $pdf -Destination "$pdf.attack-backup"
Copy-Item -LiteralPath $R -Destination "$R.attack-backup"
[System.IO.File]::WriteAllBytes($pdf, [byte[]]@())
Edit-Json $R 'd["pdf"]["engine"]="unknown-engine"; d["pdf"]["size_bytes"]=0; d["pdf"]["metadata"]={}'
```
- Expected rejection: empty PDF, wrong engine, hash/size mismatch, and absent metadata.
- Lethal if accepted: command exit substitutes for a nonempty hash-bound PDF with
  frozen engine/pass metadata.
- Cleanup: restore both backups exactly and verify hashes.

### VIA000-A13 — dirty tracked regeneration

- Target: `source_law/results/validation.json`; append a blank line so decoded JSON is
  equivalent while tracked bytes are dirty.
- Command: `Add-Content -LiteralPath $p -Value ''`, then run both the artifact checker
  and `git diff --exit-code` from `$D`.
- Expected rejection: semantic checker may return `0`, but Git diff returns `1` and the
  combined evidence contract fails.
- Lethal if accepted: only semantic success is reported and tracked dirt is ignored.
- Cleanup: restore the exact JSON.
- Qualification result: checker `0`, Git diff `1`, as expected.

### VIA000-A14 — untracked required output without emptiness

- Target: nonempty required `source_law/results/source_scaling.png`; remove only its
  index entry using `git -C $D rm --cached --quiet -- $v`, then run the checker.
- Expected rejection: `required visual is not tracked by git` even though nonempty.
- Lethal if accepted: required output exists only as untracked workspace residue.
- Cleanup: restore the exact target with `--staged --worktree`.

### VIA000-A15 — evidence only in an agent summary

- Target: `$R` and packet receipt set; delete platform, command, artifact, count,
  duration, environment, mutation, and PDF records, leaving only bound Booleans.
- Command:

```powershell
Copy-Item -LiteralPath $R -Destination "$R.attack-backup"
$summary='{"capabilities":{"evidence-contract":true,"cross-platform-reproduction":true,"mutation-rejection":true},"failed":false,"blocked":false}' + "`n"
[System.IO.File]::WriteAllText($R, $summary, [System.Text.UTF8Encoding]::new($false))
```
- Expected rejection: receipt/adjudication invalid because no hashed raw evidence
  supports the Booleans; prose cannot cure it.
- Lethal if accepted: `check_viability_campaign.py` binds five summary Booleans and
  permits a pass without inspectable raw receipts. Run this against the complete
  future receipt chain.
- Cleanup: restore `$R.attack-backup` and hash.

### VIA000-A16 — wrong candidate, tree, or protocol identity

- Target: `$R:/candidate_commit`, `/candidate_tree`, `/protocol_commit` and matching
  hashed environment/log fields.
- Command: back up `$R`, then execute
  `Edit-Json $R 'd["candidate_commit"]="70c867552279b74d5ce1a7bc5c50d5a980cf81e6"; d["candidate_tree"]="0000000000000000000000000000000000000000"; d["protocol_commit"]="555dda8741b09dab762db7a8faeb5361a8b3e1ac"'`.
- Expected rejection: each must exactly match the frozen identities; mismatch makes the
  run invalid, not failed or blocked.
- Lethal if accepted: evidence from another tree/protocol is bound to VIA-000.
- Cleanup: restore `$R.attack-backup` and hash.

### VIA000-A17 — missing frozen command

- Target: `$R:/commands`; delete the record whose exact command is `uv sync --frozen`.
  Repeat independently for all 13 frozen command strings and for both PDF executions.
- Command: back up `$R`, then
  `Edit-Json $R 'd["commands"]=[x for x in d["commands"] if x.get("command")!="uv sync --frozen"]'`.
- Expected rejection: exact command-set comparison reports the omission; evidence
  contract false regardless of other green commands.
- Lethal if accepted: a skipped dependency sync or other required command is hidden.
- Cleanup: restore `$R.attack-backup` and hash.

### VIA000-A18 — wrong counts

- Target: `$R:/counts`; expected tests `187`, examples `6`, mutations `6`, PDF passes
  `2`.
- Command: back up `$R`, then
  `Edit-Json $R 'd["counts"]={"tests":186,"examples":5,"mutations":5,"pdf_passes":1}'`.
- Expected rejection: all exact mismatches; also execute each mismatch independently.
- Lethal if accepted: incomplete execution passes through a headline Boolean.
- Cleanup: restore `$R.attack-backup` and hash.

### VIA000-A19 — nonfinite numeric value

- Target: `source_law/results/validation.json`, first
  `$.measurements.relative_entropy` literal `5.952349724225314e-11`.
- Command:

```powershell
$p=Join-Path $D 'examples/physics_qg/source_law/results/validation.json'
@'
import sys
from pathlib import Path
p=Path(sys.argv[1]); s=p.read_text(encoding="utf-8")
assert "5.952349724225314e-11" in s
p.write_text(s.replace("5.952349724225314e-11", "NaN", 1), encoding="utf-8")
'@ | python - $p
Push-Location $D; python scripts/check_validation_artifacts.py; Pop-Location
```

  Repeat future receipt tests with `Infinity`, `-Infinity`, overflow, and nonzero
  underflow.
- Expected rejection: exit `1`, `invalid strict JSON: non-finite JSON constant 'NaN'`.
- Lethal if accepted: nonfinite evidence reaches comparisons/adjudication.
- Cleanup: restore the exact JSON.
- Qualification result: rejected as expected.

### VIA000-A20 — duplicate check identity

- Target: `source_law/results/validation.json:$.checks`; deep-copy and append the first
  check without renaming it.
- Command: `Edit-Json $p 'import copy; d["checks"].append(copy.deepcopy(d["checks"][0]))'`,
  then the artifact checker.
- Expected rejection: changed array length and duplicate check name.
- Lethal if accepted: duplicate greens inflate or ambiguously overwrite evidence.
- Cleanup: restore the exact JSON.
- Qualification result: both diagnostics observed.

### VIA000-A21 — post-output threshold selection

- Target: many-body JSON
  `$.config.maximum_kubo_mori_coefficient_relative_error`, frozen `0.0005`.
- Command: after reading the result, run
  `Edit-Json $p 'observed=abs(d["measurements"]["quadratic_coefficient_relative_error"]); d["config"]["maximum_kubo_mori_coefficient_relative_error"]=2.0*observed'`,
  then the artifact checker.
- Expected rejection: stable configuration changed after output; no pass may use the
  selected threshold.
- Lethal if accepted: observed results determine the threshold while gate success is
  retained.
- Cleanup: restore the exact many-body JSON.

## Disposable qualification results

All dynamic probes used a separate clone at exact candidate
`9a29e05f803666bf0e3a28417ea399e3e26769fc`.

| Probe | Observed result |
|---|---|
| Baseline artifact checker | exit `0`; contracts and required visuals valid |
| A01 | exit `1`; length and unregistered identity rejected |
| A02 | exit `1`; pass flip and headline inconsistency rejected |
| A03 | exit `1`; measurement-array length `16 -> 15` rejected |
| A04 | exit `1`; stable beta drift rejected |
| A05 | exit `1`; untracked and empty visual both rejected |
| A06 | exit `1`; both diagnostic copies rejected |
| A07 | shim active; Git diff `0`; checker `0`; full status showed untracked shim |
| A13 | semantic checker `0`; Git diff `1` |
| A19 | exit `1`; strict JSON rejected `NaN` |
| A20 | exit `1`; length and duplicate identity rejected |

Mutated files were restored after each probe. A clean disposable clone was retained
after the host command policy refused recursive directory removal; it contains no
attack residue and is outside the frozen worktree. This cleanup limitation neither
exposes custody data nor changes the candidate.

## Acceptance oracle

Artifact rejection is a nonzero artifact-checker exit with the specified path-level
diagnostic. Tracked dirt requires nonzero `git diff --exit-code`. Untracked residue
requires `git status --porcelain=v1 --untracked-files=all` or an equivalently complete
hashed residue check; `git diff` alone is insufficient.

For evidence attacks, every raw command/environment/artifact must be hash-bound and
independently inspectable; both platforms and exact identities must be present; counts
and PDF metadata must match the frozen protocol; all six mutations must reject; and no
summary Boolean may contradict raw evidence. Otherwise the affected capability must
be false, the run failed, or the round invalid/pending under the frozen rules. A
validator accepting a lethal condition is attack evidence, not permission for this
seat to repair source.

## Limitations and non-claims

- This is a prereveal attack plan, not runner output, statistical audit, review,
  adjudication, campaign decision, or claim promotion.
- No paired Linux/Windows run, TeX toolchain, future raw-results receipt, custody
  reveal, or final labels were available to this seat. Those attacks remain executable
  obligations for authorized later seats.
- Only the public artifact-checker surface was dynamically qualified. Future receipt
  attacks cannot be conclusively classified before immutable runner receipts exist.
- A07 proves a public cleanliness blind spot, not that a genuine run used it.
- No native/CUDA or scalability claim is in VIA-000 scope.
- This artifact assigns no packet/campaign outcome, demonstrates no Tier R result, and
  changes no implementation, protocol, packet, campaign state, schema, evidence, or
  threshold.
