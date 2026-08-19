# VIA-000 public attack plan 3

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-3`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this seat as a Codex agent based
  on GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-3`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_3`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The exact model fields remain `unknown`; neither a role label nor the general runtime
description is an exact served-model identifier or snapshot.

## Frozen identities independently verified

- Exact falsifier checkout / attack-plan parent:
  `cce782f3f8ea3695e88e049bac49a3e97bd1c054`
- Checkout tree: `e33d91d6f3ecf43165643871298eedbe9b1b6eba`
- Checkout parent: `9f0d51000b10ec705a6e24067398e530e98dbd60`
- Activated campaign handoff named by the launch brief:
  `9f0d51000b10ec705a6e24067398e530e98dbd60`
- Activated-handoff tree: `91f78692b7adf41e0065f0a1b7f5e7b7a5b91e15`
- Protocol snapshot: `5ae56f9f91dd00ef38a5e351438f5962d51750e6`
- Protocol-snapshot tree: `b38a75948dc40df742698064df0cc99dd873e8f9`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `47705e5244470fa8c946a734536783f9b8c4f3863eb2c369f65031a1c7b84a4b`
- Primary-protocol raw-byte SHA-256:
  `db1ef7ccd9732506f39e65a95e71a145da6a3c7c0406ac6e1f90be8da7366684`
- Protocol-receipt raw-byte SHA-256: the same
  `db1ef7ccd9732506f39e65a95e71a145da6a3c7c0406ac6e1f90be8da7366684`
- Protocol-manifest raw-byte SHA-256:
  `0613c60809d11586d2a0da8900d4d4c66de1418ec8391622a70993fa1f5e2f56`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-3` /
  `campaign/via000-falsifier-3`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-3-candidate` (detached)

The commits and trees were resolved from Git objects rather than accepted from branch
names. The artifact worktree was created from the exact hardened handoff with:

```powershell
git worktree add -b campaign/via000-falsifier-3 `
  C:\src\POPGP-via000-falsifier-3 `
  cce782f3f8ea3695e88e049bac49a3e97bd1c054
```

The mutation worktree was independently created at the scientific candidate:

```powershell
git -C C:\src\POPGP worktree add --detach `
  C:\src\POPGP-via000-falsifier-3-candidate `
  9a29e05f803666bf0e3a28417ea399e3e26769fc
```

## Access, independence, and prereveal declaration

- `final_labels_seen: false`
- `secret_seed_seen: false`
- `private_evaluator_seen: false`
- Custody repository or path accessed: `false`
- Sealed manifest bytes or contents accessed: `false`
- `C:\src\POPGP\POPGP_Codex_Handoff.md` accessed: `false`
- Builder calibration memo contents or conclusions accessed: `false`
- Builder conclusions supplied before attack construction: `false`
- Public attack plans 1 and 2 read: `true`, as explicitly required by the hardened
  handoff
- Public pre-holdout amendments 1 and 2 read: `true`, as explicitly required
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because the exact builder and
  falsifier model identities are both recorded as `unknown`
- External scientific validation: `false`
- Restricted exposure boundary crossed: `false`

I read the launch brief, both public amendments, both public prior attack plans,
campaign, VIA-000 packet, primary protocol, viability plan, review workflow, and
reviewer-identity guidance completely. I did not open a custody location, sealed
manifest, builder memo, final label, or private evaluator material.

## Clean harness and exact boundary commands

Before locked sync, the disposable candidate had no tracked, untracked, or ignored
status. The exact frozen preflight was:

```powershell
python -I -c "import os,subprocess,sys; ps=[subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True),subprocess.run(['git','status','--porcelain=v1','--untracked-files=normal','--ignored'],capture_output=True,text=True)]; [sys.stdout.write(p.stdout) for p in ps]; [sys.stderr.write(p.stderr) for p in ps]; blocked=sorted(k for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_PROJECT_ENVIRONMENT') if os.environ.get(k)); sys.stderr.write(('blocked environment: '+','.join(blocked)+'\n') if blocked else ''); nested=sys.prefix!=sys.base_prefix; sys.stderr.write('base interpreter required\n' if nested else ''); raise SystemExit(any(p.returncode for p in ps) or any(p.stdout for p in ps) or bool(blocked) or nested)"
```

Observed clean-preflight exit: `0`. The host base interpreter was Python `3.12.10`,
with `sys.prefix == sys.base_prefix`. `uv sync --frozen` then returned `0` and created
the frozen scientific environment with Python `3.11.15`.

The exact frozen postflight was:

```powershell
python -I -c "import os,pathlib,subprocess,sys; p=subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True); sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); blocked=sorted(k for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_PROJECT_ENVIRONMENT') if os.environ.get(k)); sys.stderr.write(('blocked environment: '+','.join(blocked)+'\n') if blocked else ''); nested=sys.prefix!=sys.base_prefix; sys.stderr.write('base interpreter required\n' if nested else ''); env_path=pathlib.Path('.venv'); invalid_env=not env_path.is_dir() or env_path.is_symlink(); sys.stderr.write('regular .venv directory required\n' if invalid_env else ''); hooks=sorted(str(q) for name in ('sitecustomize.py','usercustomize.py') for q in env_path.rglob(name)) if not invalid_env else []; sys.stdout.write(''.join(h+'\n' for h in hooks)); raise SystemExit(p.returncode or bool(p.stdout) or bool(blocked) or nested or invalid_env or bool(hooks))"
```

The mutation checker was always executed as:

```powershell
uv run python scripts/check_validation_artifacts.py
```

Each tracked mutation was restored from exact candidate
`9a29e05f803666bf0e3a28417ea399e3e26769fc` before the next probe. Every untracked or
ignored attack file and generated attack bytecode was removed by exact path.

## Eight frozen mutation-family executions

### VIA000-C01 — unregistered noninformational check

Mutation: append `{"name":"unregistered_required_probe","passed":true}` to
`source_law/results/validation.json:$.checks`.

Observed checker exit: `1`.

```text
$.checks: array length changed from 4 to 5
non-informational check set is not explicitly registered
(missing=[], unregistered=['unregistered_required_probe'])
```

Result: **rejected**.

### VIA000-C02 — required pass flip

Mutation: change
`source_law/results/validation.json:$.checks[0].passed` from `true` to `false`.

Observed checker exit: `1`.

```text
$.checks[0].passed: changed from True to False
overall_pass=True but non-informational conjunction is False
```

Result: **rejected**.

### VIA000-C03 — measurement-array shape change

Mutation: remove the final element of
`source_law/results/validation.json:$.measurements.relative_entropy`, changing length
`16` to `15`.

Observed checker exit: `1`.

```text
$.measurements.relative_entropy: array length changed from 16 to 15
```

Result: **rejected**.

### VIA000-C04 — stable configuration drift

Mutation: change `chain_1d/results/validation.json:$.config.beta` from `1.0` to
`1.01`.

Observed checker exit: `1`.

```text
$.config.beta: stable input drift exceeds rel_tol=1.77636e-15, abs_tol=0
(1.0 -> 1.01, abs=0.01, rel=0.00990099)
```

Result: **rejected**.

### VIA000-C05 — required visual empty and untracked

The original visual SHA-256 was
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.
The exact mutation sequence was:

```powershell
$v='examples/physics_qg/source_law/results/source_scaling.png'
$full=(Resolve-Path -LiteralPath $v).Path
git rm --cached --quiet -- $v
[System.IO.File]::WriteAllBytes($full,[byte[]]@())
uv run python scripts/check_validation_artifacts.py
git status --porcelain=v1 --untracked-files=all -- $v
```

Observed checker exit: `1`; status showed a staged deletion and untracked replacement.

```text
required visual is not tracked by git
required visual is empty
```

Exact restoration reproduced the original SHA-256. Result: **rejected**.

### VIA000-C06 — sensitive diagnostic beyond threshold

Mutation: set both the measurement and check-value copies of
`quadratic_coefficient_relative_error` in the many-body validation artifact from
`3.812167720149195e-08` to `0.001`.

Observed checker exit: `1`.

```text
$.checks[0].value.quadratic_coefficient_relative_error: sensitive diagnostic drift
exceeds rel_tol=0, abs_tol=0.0002
$.measurements.quadratic_coefficient_relative_error: sensitive diagnostic drift
exceeds rel_tol=0, abs_tol=0.0002
```

Result: **rejected at both paths**.

### VIA000-C07 — root sitecustomize plus PYTHONPATH

An untracked repository-root `sitecustomize.py` set
`POPGP_UNDECLARED_STATE=active`. With process-local `PYTHONPATH=.` the sequence was:

```powershell
$prior=$env:PYTHONPATH
try {
  $env:PYTHONPATH='.'
  uv run python -c "import os; print(os.environ.get('POPGP_UNDECLARED_STATE'))"
  git diff --exit-code
  uv run python scripts/check_validation_artifacts.py
  python -I -c <exact-postflight-body-above>
} finally {
  $env:PYTHONPATH=$prior
}
```

Observed:

```text
active
ordinary exit=0
git diff exit=0
artifact checker exit=0
?? sitecustomize.py
blocked environment: PYTHONPATH
postflight exit=1
```

Result: **rejected by the amended base-interpreter postflight**. The root source and
generated `__pycache__/sitecustomize.cpython-311.pyc` were removed by exact path.

### VIA000-C08 — ignored virtual-environment startup hooks, both placements

#### Pre-sync placement

I removed the disposable `.venv`, then preplanted both:

```text
.venv/Lib/site-packages/sitecustomize.py
.venv/Lib/site-packages/usercustomize.py
```

Each file set a distinct benign environment marker. Before the exact preflight,
`python -I` reported both markers `None` and `sys.prefix == sys.base_prefix`; the
preplanted hooks therefore did not control the base interpreter. The exact preflight
returned `1` and ignored status exposed:

```text
!! .venv/Lib/site-packages/sitecustomize.py
!! .venv/Lib/site-packages/usercustomize.py
```

Result: **rejected before locked sync**.

#### Post-sync placement

After deleting the preplant and recreating `.venv` with a clean
`uv sync --frozen`, I injected
`.venv/Lib/site-packages/sitecustomize.py`. It set
`POPGP_POSTSYNC_SITE=active`.

Observed:

```text
uv run python: POPGP_POSTSYNC_SITE=active; nested=True; exit=0
base python -I: POPGP_POSTSYNC_SITE=None; base_interpreter=True; exit=0
postflight printed .venv\Lib\site-packages\sitecustomize.py; exit=1
```

The executable hook controlled ordinary protocol Python but could not control the
base-interpreter check that rejected it. A separate post-sync `usercustomize.py` probe
did not execute under this virtual environment, but the exact postflight still listed
it and returned `1`.

Result: **the named B08 sitecustomize/usercustomize variants are rejected in both
required placements**.

## Decisive equivalent startup attack

### VIA000-C09 — ignored executable `.pth` survives postflight

Python processes executable `import` lines in `.pth` files under site-packages during
startup. The frozen postflight scans only files named `sitecustomize.py` and
`usercustomize.py`; normal porcelain status also suppresses ignored `.venv` files.

After a clean locked sync, I added exactly:

```text
# .venv/Lib/site-packages/via000_attack.pth
import os; os.environ["POPGP_PTH_STATE"] = "active"
```

The executed sequence was:

```powershell
uv run python -c "import os,sys; print('pth_state='+str(os.environ.get('POPGP_PTH_STATE'))); print('nested='+str(sys.prefix!=sys.base_prefix))"
python -I -c "import os,sys; print('pth_state='+str(os.environ.get('POPGP_PTH_STATE'))); print('base_interpreter='+str(sys.prefix==sys.base_prefix))"
uv run python scripts/check_validation_artifacts.py
git diff --exit-code
python -I -c <exact-postflight-body-above>
git status --porcelain=v1 --ignored --untracked-files=all |
  Select-String 'via000_attack\.pth'
```

Observed:

```text
pth_state=active
nested=True
ordinary exit=0
pth_state=None
base_interpreter=True
base probe exit=0
Validation artifact contracts and required visual outputs are valid.
artifact checker exit=0
git diff exit=0
postflight exit=0
!! .venv/Lib/site-packages/via000_attack.pth
```

Result: **accepted mutation**. The base interpreter is correctly insulated from the
hook, but the frozen postflight does not inventory the executable `.pth` surface, so
it cannot reject the same undeclared startup behavior that the protocol says is an
exact failure. This is not hypothetical: ordinary `uv run python` executed the benign
payload before the process body while every frozen post-sync gate returned zero.

The attack `.pth` was deleted by exact path. The clean postflight then returned `0`;
normal Git status was empty and an ignored-status residue search found no attack,
`sitecustomize`, or `usercustomize` file.

## Equivalent environment and boundary probes

Each frozen blocked environment variable was independently set to `VIA000_ATTACK`
for one base-interpreter postflight and then restored. Exact exits were:

| Probe | Exit | Diagnostic |
|---|---:|---|
| `PYTHONPATH` | 1 | `blocked environment: PYTHONPATH` |
| `PYTHONHOME` | 1 | `blocked environment: PYTHONHOME` |
| `VIRTUAL_ENV` | 1 | `blocked environment: VIRTUAL_ENV` |
| `UV_PROJECT_ENVIRONMENT` | 1 | `blocked environment: UV_PROJECT_ENVIRONMENT` |

Running the postflight through `uv run python -I` returned `1` with both
`blocked environment: VIRTUAL_ENV` and `base interpreter required`. The clean base
postflight returned `0` after restoration.

Two further public boundary controls were independently executed:

- Sign-reversing both canonical copies of
  `exact_kubo_mori_quadratic_coefficient`, without changing JSON shape, returned
  checker exit `1` with diagnostic drift at the measurement and check paths.
- Adding a single whitespace byte after the opening JSON brace preserved semantic
  checker exit `0`, while `git diff --exit-code` returned `1`. This confirms the
  tracked-dirt half of the combined gate.
- `pdflatex -interaction=nonstopmode -halt-on-error` against a nonexistent TeX source
  returned `1` and produced no PDF. The installed engine reported
  `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)`. Temporary output was outside the
  repository and removed by an exact validated path.

The Linux/Windows-only-success, wrong PDF count/metadata, missing command, wrong
identity/count, and summary-only-evidence attacks from plan 1 remain executable
receipt-level obligations. They cannot be dynamically adjudicated before the runner
creates an immutable raw-results receipt. Their rejection oracle remains exact:
missing either platform, any frozen command, any required raw support, either PDF
pass, exact count `187/6/8/2`, or any frozen identity makes the evidence capability
false or the round invalid; a prose/Boolean summary cannot cure missing hashed raw
evidence.

## Static validator result

At exact checkout `cce782f3f8ea3695e88e049bac49a3e97bd1c054`, after
`uv sync --frozen`, I ran:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Observed exit: `0`.

```text
Viability campaign contract is valid.
```

This confirms the frozen documents and hashes are statically coherent. The validator
does not execute the dynamic startup mutations and therefore does not contradict C09.

## Falsifier conclusion and holdout recommendation

All eight frozen mutation families were independently executed. C01–C07 reject, and
amendment 2 closes the named B08 pre-sync and post-sync
`sitecustomize.py`/`usercustomize.py` placements. Nevertheless, the hardened handoff
is **unsafe to transition to `attacked` / `holdout_started: true`** because C09 is a
reproducible accepted mutation in the same ignored-startup/evidence-integrity threat
model.

The frozen hypothesis and statistical analysis require rejection of undeclared import
state. An executable `.pth` line changes ordinary protocol Python startup while the
artifact checker, Git diff, normal status, and exact base-interpreter postflight all
pass. Advancing would assert a predicate the frozen command sequence cannot establish.
This is a pre-holdout protocol defect, not a scientific-candidate failure or packet
adjudication.

Before holdout, a new amendment must bind the expected executable site-packages
startup surface after locked sync—for example by a frozen inventory/hash of `.pth`
files and their bytes, plus the named customize hooks—and reject additions or changes
using the base interpreter. It must account for the legitimate lock-created
`_editable_impl_popgp.pth` and `_virtualenv.pth` rather than banning `.pth` by filename
alone. A fourth fresh falsifier should challenge that refreeze before lifecycle
transition. This artifact does not implement that repair.

## Limitations and non-claims

- No Linux runner, immutable runner raw-results receipt, output commitment, custody
  reveal, final label, or adjudication was available to this seat.
- The integrity checker used the frozen `python -I` command, which resolved to host
  Python `3.12.10`; the locked scientific environment was Python `3.11.15`. The
  protocol does not assert that the base integrity checker itself must use the locked
  minor version.
- C09 proves that the present gate accepts a controlled startup mutation; it does not
  allege that an honest run already contained that file.
- This artifact changes no implementation, protocol, packet, campaign lifecycle,
  threshold, measurement, or custody state; it assigns no packet outcome and
  establishes no Tier R, mechanism, gravitational, native/CUDA, or external-validation
  claim.
