# VIA-000 public attack plan 6

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-6`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this seat as a Codex agent based
  on GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-6`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_6`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The exact model fields remain `unknown`; neither the seat label nor the general
runtime description supplies an exact model identifier or version.

## Frozen identities independently verified

- Exact falsifier checkout / attack-plan parent:
  `886bb05bf0153a5c371f0ad78cfaf6247774bd47`
- Checkout tree: `1364db205b6bce011745e0f0ea1e70bb0a3d4ab7`
- Checkout parent / activated campaign handoff named by the launch brief:
  `75473e222b4b5c56b69ae39c82be1bd039b15672`
- Protocol snapshot: `35a4acf304541deabd03cac16b93df9c84d06b9d`
- Protocol-snapshot tree: `a0e15a1c0186278438bb15dcdb13bf8f22e85b3e`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `963a20b1f677b02cd6eee4164b2fe31736f9c3b402fb7ef2adc4aaeedd21c99b`
- Primary-protocol and protocol-receipt raw-byte SHA-256:
  `14e3fc144773374e41993b36f53f21c334187d49fc7147c0cf58fa0663eccf82`
- Protocol-manifest raw-byte SHA-256:
  `40e3ccf266e6bf77026ae7679b9c7975e808484a7bb9038a17cf5dda18b8dfd0`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-6` /
  `campaign/via000-falsifier-6`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-6-candidate` (detached)

All identities were resolved from Git objects. No existing source or campaign
worktree was switched or modified.

## Access, independence, and prereveal declaration

- `final_labels_seen: false`
- `secret_seed_seen: false`
- `private_evaluator_seen: false`
- Custody repository or path accessed: `false`
- Sealed manifest bytes or contents accessed: `false`
- `C:\src\POPGP\POPGP_Codex_Handoff.md` accessed: `false`
- Builder calibration memo contents or conclusions accessed: `false`
- Builder conclusions supplied before attack construction: `false`
- Public attack plans 1 through 5 read: `true`, as explicitly required
- Public pre-holdout amendments 1 through 5 read: `true`, as explicitly required
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because the exact builder and
  falsifier model identities are both recorded as `unknown`
- External scientific validation: `false`
- Restricted custody/holdout boundary crossed: `false`

I read the launch brief, all five amendments, all five prior public attack plans,
campaign, VIA-000 packet, primary protocol, viability plan, review workflow, and
reviewer-identity guidance completely. I did not open a custody location, sealed
manifest, builder memo, final label, or private evaluator material.

## Harness and clean controls

The clean and mutation harness was a new detached worktree at the exact scientific
candidate. Host metadata was:

```text
Microsoft Windows 11 Enterprise
base Python 3.12.10; sys.prefix == sys.base_prefix
uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
locked/isolated Python 3.11.15
pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
```

Before sync, normal and ignored porcelain status were empty and the exact frozen
base-interpreter preflight exited `0`. `uv sync --frozen --no-editable` exited `0`
and produced exactly one project-environment startup file:

```text
.venv/Lib/site-packages/_virtualenv.pth
  import _virtualenv
```

The uninterrupted mandatory repaired test command was:

```powershell
uv run --isolated --frozen --no-editable python -m pytest -q
```

It exited `0` with exactly:

```text
187 passed in 1123.43s (0:18:43)
```

The wrapper wall time was `1150.2s`, including isolated installation. An earlier
attempt hit a five-minute shell-wrapper timeout while its child processes continued;
those exact processes were identified and stopped before this uninterrupted result.
The timed-out attempt is not counted as evidence.

The remaining clean frozen commands produced:

| Command or gate | Observed result |
|---|---|
| exact base preflight | exit `0` |
| `uv sync --frozen --no-editable` | exit `0` |
| isolated/frozen/non-editable `ruff check .` | exit `0`; all checks passed |
| isolated/frozen/non-editable `python scripts/check_tex.py` | exit `0` |
| repaired full pytest module invocation | exit `0`; exactly `187 passed` |
| six isolated/frozen/non-editable example modules | all six exit `0` |
| isolated/frozen/non-editable artifact checker | exit `0` |
| `git diff --exit-code` after regeneration | exit `0` |
| first exact PDF pass | exit `0`; 11 pages, 531345 bytes |
| second exact PDF pass | exit `0`; 11 pages, 535368 bytes |

The final two-pass PDF SHA-256 was
`83240a8cfe9b97961e66fc33dd5b1adc408bc97023abcf6f7982bbd8ce71884a`.
The pytest module-invocation repair therefore resolves the precise clean collection
failure from Plan 5. The complete clean sequence nevertheless fails later, as
described below.

## Frozen mutation-family outcomes

The first seven families were refreshed directly in this disposable exact-candidate
worktree. Each tracked mutation was restored from the exact candidate before the next
probe. The last three startup families retain the exact executable evidence already
committed in Plan 5; they were not redundantly re-executed after the decisive clean
PDF/postflight failure was found. This provenance distinction is explicit rather than
presenting inherited observations as new execution.

### VIA000-F01 — unregistered noninformational check

Appending `unregistered_required_probe` to the source-law check array made the exact
isolated non-editable artifact checker exit `1`. It reported array length `4 -> 5`
and the unregistered identity. Result: **rejected (directly refreshed)**.

### VIA000-F02 — required pass flip

Changing `relative_entropy_is_quadratic.passed` from `true` to `false` made the
checker exit `1`. It reported the exact pass drift and contradiction with
`overall_pass=true`. Result: **rejected (directly refreshed)**.

### VIA000-F03 — measurement-array shape change

Reducing the source-law relative-entropy array from 16 values to 15 made the checker
exit `1` with the exact length diagnostic. Result: **rejected (directly refreshed)**.

### VIA000-F04 — stable configuration drift

Changing chain beta from `1.0` to `1.01` made the checker exit `1` with drift beyond
`rel_tol=1.77636e-15, abs_tol=0`. Result: **rejected (directly refreshed)**.

### VIA000-F05 — required visual empty and untracked

Removing the source-scaling visual from the index and replacing its worktree bytes
with an empty file made the checker exit `1` with both `not tracked by git` and
`empty`. Restoration reproduced SHA-256
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.
Result: **both defects rejected (directly refreshed)**.

### VIA000-F06 — sensitive diagnostic beyond threshold

Changing both canonical copies of `quadratic_coefficient_relative_error` from
`3.812167720149195e-08` to `0.001` made the checker exit `1` at both paths beyond
`abs_tol=0.0002`. Result: **rejected (directly refreshed)**.

### VIA000-F07 — root customize plus PYTHONPATH

An untracked root `sitecustomize.py` set `POPGP_M07_ROOT=active`, with process-local
`PYTHONPATH=.`. Isolated non-editable Python printed `active`; the semantic checker
and `git diff --exit-code` both returned `0`. The exact base-interpreter postflight
printed `?? sitecustomize.py`, reported `blocked environment: PYTHONPATH`, and exited
`1`. Result: **rejected by the complete frozen boundary (directly refreshed)**.

### VIA000-F08 — ignored customize hooks

Plan 5's immutable executable receipt places ignored project-environment
`sitecustomize.py` and `usercustomize.py` both before and after sync. Preflight rejects
the pre-sync ignored state; isolated non-editable execution is inert; postflight lists
and rejects the post-sync hooks. Result: **rejected (inherited exact Plan 5
execution)**.

### VIA000-F09 — executable and modified `.pth` surfaces

Plan 5's immutable executable receipt shows an extra executable `.pth` and a modified
`_virtualenv.pth` execute under project Python but remain inert under isolated
non-editable Python; the exact postflight rejects both. Result: **rejected (inherited
exact Plan 5 execution)**.

### VIA000-F10 — editable self-cleaning carrier

Plan 5 independently reproduced the exact historical exploit under editable
`uv sync --frozen` / `uv run --isolated --frozen`: the copied editable `.pth` set its
marker and restored its project carrier before the command body. Transitioning with
`uv sync --frozen --no-editable` removed the carrier; isolated frozen non-editable
execution left the marker and sentinel absent, and postflight rejected any later
extra carrier. Result: **historical exploit confirmed and amended boundary rejection
retained (inherited exact Plan 5 execution)**.

No startup-family evidence from Plans 1 through 5 is weakened by the present run. The
new decisive defect is instead in the exact clean PDF/postflight sequence.

## Decisive clean-control falsifier

### VIA000-F11 — exact PDF command leaves forbidden untracked residue

From the disposable repository root, I executed the exact required command twice:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error docs/framework.tex
```

Both passes exited `0` and produced a nonempty 11-page PDF with the frozen TeX engine.
`git diff --exit-code` still exited `0`, but the exact frozen postflight printed:

```text
?? framework.aux
?? framework.log
?? framework.out
?? framework.pdf
?? frameworkNotes.bib
```

and exited `1`. These files are ordinary untracked repository-root residue, not
ignored project-environment state. The exact frozen sequence contains no cleanup or
out-of-tree output command between PDF generation and postflight. Deleting those five
files by exact validated paths made a diagnostic postflight return `0`, but that
cleanup is not a frozen command and cannot be silently inserted by a runner.

Result: **accepted clean-control falsifier / protocol feasibility failure**. The PDF
build itself succeeds; the evidence-integrity postflight correctly rejects its own
declared command's residue. Therefore the complete clean command sequence cannot
produce a passing result as frozen.

## Static validator outcome

At exact checkout `886bb05bf0153a5c371f0ad78cfaf6247774bd47`, I ran:

```powershell
uv run --isolated --frozen --no-editable python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

It exited `0` with:

```text
Viability campaign contract is valid.
```

The packet-rule helper exited `0` and returned
`963a20b1f677b02cd6eee4164b2fe31736f9c3b402fb7ef2adc4aaeedd21c99b`.
Static schema/hash coherence does not execute `pdflatex` or the postflight, so it does
not contradict F11.

## Falsifier conclusion and holdout recommendation

The fifth amendment fixes the Plan 5 pytest feasibility defect: the exact mandatory
command passes all 187 tests. The directly refreshed artifact mutations reject, and
the prior immutable startup executions continue to show that customize, executable
`.pth`, and editable self-cleaning variants are isolated or rejected by the amended
non-editable boundary.

Nevertheless, the fifth amended handoff is **unsafe to transition to `attacked` /
`holdout_started: true`**. Its own successful PDF command creates five untracked files
that its exact postflight necessarily rejects. A runner cannot obtain a valid clean
result without adding an undeclared cleanup step or changing the PDF output location,
both of which alter the frozen execution envelope.

This is a pre-holdout protocol feasibility defect, not a scientific-candidate failure
and not packet adjudication. A sixth pre-holdout amendment should make PDF output and
retained evidence explicit while preserving final repository cleanliness—for example,
by freezing an out-of-tree output directory and hash/archive step, or by freezing an
exact cleanup/retention procedure before postflight. A seventh fresh falsifier should
execute the revised complete clean sequence before lifecycle advancement. This seat
does not implement that repair.

## Limitations and non-claims

- Only Windows was dynamically exercised; Linux runner evidence does not exist.
- F08 through F10 are inherited exact executable evidence from immutable Plan 5, not
  newly repeated execution in this session; F11 made further redundant repetition
  immaterial to the unsafe recommendation.
- No immutable runner raw-results receipt, output commitment, custody reveal, final
  label, statistical audit, or adjudication was available.
- The PDF result is a clean protocol-command/postflight contradiction, not a failure
  of the scientific hypothesis or a packet outcome.
- This artifact changes no implementation, protocol, packet, campaign lifecycle,
  threshold, measurement, mutation rule, or custody state; it assigns no packet
  outcome and establishes no Tier R, mechanism, gravitational, native/CUDA, or
  external-validation claim.
