# VIA-000 public attack plan 5

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-5`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this seat as a Codex agent based
  on GPT-5, but does not expose an operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-5`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_5`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The exact model fields remain `unknown`; neither the seat label nor the general
runtime description is an exact served-model identifier or snapshot.

## Frozen identities independently verified

- Exact falsifier checkout / attack-plan parent:
  `abb04704774990e962ce2b05dcb826c171475802`
- Checkout tree: `11001b8d086cbc26f9356f41f769bc697e329fea`
- Checkout parent / activated campaign handoff:
  `e55619a90f99b10e781d115a62c6221cbccd3456`
- Activated-handoff tree: `bac68576314eaee76b00d41e35093aea18f53ddf`
- Protocol snapshot: `85587d061d20b518a8612293fca70f3764e0e1ec`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `5c753eff4794d34a9fa18be276942d76b7df417a9a728577cde5395d2fbf4100`
- Primary-protocol raw-byte SHA-256:
  `3f810ff466bc5506ac203ddecead3a43bf274b87aad9422d284534313853653c`
- Protocol-receipt raw-byte SHA-256: the same
  `3f810ff466bc5506ac203ddecead3a43bf274b87aad9422d284534313853653c`
- Protocol-manifest raw-byte SHA-256:
  `6a3f721c6c06f78e920dc70e0a4c0f4f48e7966ae5438eea5a1a3c50eb3fb98b`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-5` /
  `campaign/via000-falsifier-5`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-5-candidate` (detached)

All commits and trees were resolved from Git objects. No existing source or campaign
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
- Public attack plans 1 through 4 read: `true`, as explicitly required
- Public pre-holdout amendments 1 through 4 read: `true`, as explicitly required
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because the exact builder and
  falsifier model identities are both recorded as `unknown`
- External scientific validation: `false`
- Restricted custody/holdout boundary crossed: `false`

I read the launch brief, all four amendments, all four prior public attack plans,
campaign, VIA-000 packet, primary protocol, viability plan, review workflow, and
reviewer-identity guidance completely. I did not open a custody location, sealed
manifest, builder memo, final label, or private evaluator material.

## Harness, platform, and clean controls

The mutation harness was a new detached worktree at the exact scientific candidate.
Before sync, normal and ignored porcelain status were empty and the exact frozen
base-interpreter preflight exited `0`. Host metadata was:

```text
Windows-11-10.0.26200-SP0
base Python 3.12.10; sys.prefix == sys.base_prefix
uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
locked/isolated Python 3.11.15
```

The first clean `uv sync --frozen --no-editable` exited `0` and produced exactly:

```text
.venv/Lib/site-packages/_virtualenv.pth
  import _virtualenv
```

The clean semantic artifact checker exited `0`. Every tracked mutation below was
restored from exact candidate
`9a29e05f803666bf0e3a28417ea399e3e26769fc`; named ignored or untracked attack files
and bytecode were removed by exact path before the next probe.

## Ten frozen mutation-family executions

### VIA000-E01 — unregistered noninformational check

Appended `unregistered_required_probe` to
`source_law/results/validation.json:$.checks` and ran the exact isolated,
lock-frozen, non-editable checker. Exit `1` reported length `4 -> 5` and the
unregistered identity. Result: **rejected**.

### VIA000-E02 — required pass flip

Changed `source_law/results/validation.json:$.checks[0].passed` from `true` to
`false`. Exit `1` reported the pass change and contradiction with
`overall_pass=true`. Result: **rejected**.

### VIA000-E03 — measurement shape change

Removed the final element from
`source_law/results/validation.json:$.measurements.relative_entropy`, changing its
length from 16 to 15. Exit `1` reported the exact array-length change. Result:
**rejected**.

### VIA000-E04 — stable configuration drift

Changed `chain_1d/results/validation.json:$.config.beta` from `1.0` to `1.01`.
Exit `1` reported drift beyond `rel_tol=1.77636e-15, abs_tol=0`. Result:
**rejected**.

### VIA000-E05 — required visual empty and untracked

Removed the index entry for
`examples/physics_qg/source_law/results/source_scaling.png` and replaced the
worktree file with zero bytes. Exit `1` reported both `not tracked by git` and
`empty`; status showed a staged deletion plus an untracked replacement. Exact
restoration reproduced SHA-256
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.
Result: **both defects rejected**.

### VIA000-E06 — sensitive diagnostic beyond threshold

Changed both canonical copies of
`quadratic_coefficient_relative_error` in the many-body validation artifact from
`3.812167720149195e-08` to `0.001`. Exit `1` reported sensitive-diagnostic drift at
both paths beyond `abs_tol=0.0002`. Result: **rejected at both paths**.

### VIA000-E07 — root customize plus PYTHONPATH

An untracked root `sitecustomize.py` set `POPGP_D07_ROOT=active`; process-local
`PYTHONPATH=.` was set. The isolated uv Python printed `active`, while the semantic
checker and `git diff --exit-code` both returned `0`. The exact base-interpreter
postflight printed `?? sitecustomize.py`, reported
`blocked environment: PYTHONPATH`, and exited `1`. Result: **rejected by the frozen
combined gate**.

### VIA000-E08 — ignored customize hooks before and after sync

Before sync, ignored project-environment `sitecustomize.py` and `usercustomize.py`
were both insulated from base Python `-I`; the exact preflight exposed `!! .venv/`
and exited `1`.

After a clean non-editable sync, the same two files were placed in project
site-packages. Project Python executed the site marker, while
`uv run --isolated --frozen --no-editable python` reported both markers `None` and
used a temporary uv-cache prefix. The exact postflight listed both hook paths and
exited `1`. Result: **both placements rejected**.

### VIA000-E09 — executable and modified `.pth` surfaces

An extra `via000_extra.pth` executed under project Python, but its marker was `None`
under isolated non-editable Python. Postflight listed the extra file and exited `1`.

Appending an executable marker line to `_virtualenv.pth` likewise controlled project
Python but not isolated non-editable Python. Postflight listed the changed allowed
file and exited `1`. A clean control restored exact content `import _virtualenv`.
Result: **extra and modified variants rejected**.

### VIA000-E10 — editable self-cleaning carrier and non-editable boundary

I first reproduced the exact prior counterexample under its vulnerable editable
precondition. After `uv sync --frozen`, the allowed file contained:

```text
C:\src\POPGP-via000-falsifier-5-candidate
import os,pathlib; os.environ["POPGP_D09_SELFCLEAN"]="active"; pathlib.Path(".venv/Lib/site-packages/_editable_impl_popgp.pth").write_text(str(pathlib.Path.cwd().resolve())+"\n",encoding="utf-8")
```

`uv run --isolated --frozen python` copied and executed the payload in a temporary
environment, printed marker `active`, and restored the project carrier to its benign
one-line form before the command body. This independently reproduces the exact D10
attack rather than accepting Plan 4's claim.

Recreating the payload and then transitioning with
`uv sync --frozen --no-editable` removed `_editable_impl_popgp.pth`. The amended
`uv run --isolated --frozen --no-editable python` printed marker `None` and used a
temporary environment. The amended postflight would reject any remaining editable
carrier because only `_virtualenv.pth` is allowed.

A stricter placement exposed an important but nonlethal nuance. If an attacker adds
an editable-named file after the project is already in non-editable state, a no-op
`uv sync --frozen --no-editable` may leave that extra file in project site-packages.
However, isolated non-editable uv did not copy or execute it: the marker remained
`None` and an instrumented persistent sentinel was absent. Exact postflight listed
both `.pth` files and exited `1`. Thus sync alone does not sanitize arbitrary later
files, but the complete frozen gate still rejects this placement.

Result: **the exact old self-cleaning attack is reproduced, then rejected by the
complete amended non-editable/isolated/postflight boundary**.

## Exact frozen uv-command probe and clean-control failure

To detect cached or copied D10 execution, an equivalent carrier additionally wrote a
root sentinel before self-cleaning. After the non-editable transition, every frozen
uv command was run with the exact `--isolated --frozen --no-editable` flags. The
sentinel and editable carrier remained absent for every command.

| Frozen uv command | Exit | D10 sentinel | Editable carrier |
|---|---:|---|---|
| `ruff check .` | 0 | absent | absent |
| `python scripts/check_tex.py` | 0 | absent | absent |
| `pytest -q` | **2** | absent | absent |
| `python -m examples.physics_qg.chain_1d` | 0 | absent | absent |
| `python -m examples.physics_qg.grid_2d` | 0 | absent | absent |
| `python -m examples.physics_qg.gravity_well` | 0 | absent | absent |
| `python -m examples.physics_qg.source_law` | 0 | absent | absent |
| `python -m examples.physics_qg.source_law_many_body` | 0 | absent | absent |
| `python -m examples.physics_qg.ca_model` | 0 | absent | absent |
| `python scripts/check_validation_artifacts.py` | 0 | absent | absent |

All six example regenerations were byte-clean: `git diff --exit-code` returned `0`.
The semantic checker returned `0`. The exact clean postflight returned `0` with only
`_virtualenv.pth`; full normal status and Git diff were empty.

The pytest failure is a decisive **clean-control protocol feasibility defect**. With
no mutation carrier, hook, environment substitution, repository dirt, or D10 sentinel
present, collection failed before the required count:

```text
ERROR tests/unit/test_validation_artifact_contract.py
  ModuleNotFoundError: No module named 'scripts'
ERROR tests/unit/test_viability_campaign_contract.py
  ModuleNotFoundError: No module named 'scripts'
2 errors during collection
```

The two tests import `scripts.check_validation_artifacts` and
`scripts.check_viability_campaign`. The frozen `pytest` console entry point in a
wheel-installed non-editable temporary environment does not place the repository root
on `sys.path`; the former editable project `.pth` had supplied that path. Direct
`python -c` is not an equivalent control because its empty `sys.path[0]` resolves the
working directory. The exact frozen command is `pytest -q`, and that exact command
fails cleanly.

This does not reopen the D10 startup exploit: all startup markers remained absent.
It instead refutes the required premise that the amended command sequence has a
passing clean non-editable control. I did not repair or rewrite the command.

## Static validator result

At exact checkout `abb04704774990e962ce2b05dcb826c171475802`, before this
artifact-only commit, I ran:

```powershell
uv run --isolated --frozen --no-editable python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Observed exit `0`:

```text
Viability campaign contract is valid.
```

The packet-rule helper returned
`5c753eff4794d34a9fa18be276942d76b7df417a9a728577cde5395d2fbf4100`.
Static schema/hash coherence does not execute the clean pytest command and therefore
does not contradict the dynamic failure.

## Falsifier conclusion and holdout recommendation

All ten frozen mutation families were independently exercised. E01 through E09 are
rejected. E10 reproduces the prior editable self-cleaning exploit, and the complete
amended non-editable/isolated/postflight boundary rejects every tested editable,
customize, extra/modified `.pth`, and persistent carrier variant. No replacement
accepted startup mutation was found.

Nevertheless, the fourth amended handoff is **unsafe to transition to `attacked` /
`holdout_started: true`**. The exact clean frozen command
`uv run --isolated --frozen --no-editable pytest -q` exits `2` during collection, so
the protocol cannot produce its required 187-test clean result. A holdout runner
would fail for a known protocol-execution reason before generating valid evidence.
This is a pre-holdout protocol/packaging feasibility defect, not a scientific-candidate
failure and not packet adjudication.

Any repair requires another pre-holdout refreeze and a fresh falsifier. One plausible
surface to test is invoking pytest through the frozen Python module entry point or
packaging the test-imported validator modules, but this artifact does not select or
implement a repair. The next clean seat must rerun the exact repaired command and all
ten mutations.

## Limitations and non-claims

- Only Windows was dynamically exercised; Linux runner evidence does not exist.
- No immutable runner raw-results receipt, output commitment, custody reveal, final
  label, statistical audit, or adjudication was available.
- The controlled startup payloads only set benign markers, wrote a sentinel, or
  restored their carrier. They do not allege that an honest prior run used an attack.
- The pytest result proves a clean protocol command failure, not a failure of the
  scientific hypothesis or a packet outcome.
- This artifact changes no implementation, protocol, packet, campaign lifecycle,
  threshold, measurement, or custody state; it assigns no packet outcome and
  establishes no Tier R, mechanism, gravitational, native/CUDA, or external-validation
  claim.
