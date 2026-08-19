# VIA-000 public attack plan 2

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-2`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes the seat as a Codex agent based
  on GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-2`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_2`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The model fields are deliberately `unknown`: neither the seat name nor the general
runtime description supplies an exact model identifier or version.

## Frozen identities independently verified

- Falsifier checkout / attack-plan parent:
  `1dca4381956cb3fb4d5d644c29a79130da6c7901`
- Checkout tree: `44eedc26f4f81326790c02174864b62483fa3bfb`
- Checkout parent: `4244a75bce127b3c5f3173e58b3c4e997792ae16`
- Activated campaign handoff named by the launch brief:
  `4244a75bce127b3c5f3173e58b3c4e997792ae16`
- Activated-handoff tree: `579b38be9e4ca3f984846b137b74f7ef66415ce2`
- Protocol snapshot: `bc206970b60ef4123a46500a404743488a536010`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `cde2d916ee1c09fd49126815a090ee207a7b259284da5d8b33dd3b6d96e28341`
- Primary-protocol raw-byte SHA-256:
  `72ad5dc9b9fa23fb3cd1b46f1fdeef6a6391f1f5a904e5db45cbac5a5030ee70`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-2` /
  `campaign/via000-falsifier-2`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-2-candidate` (detached at the scientific
  candidate)

The commits and trees were resolved from Git objects, not accepted from branch names.

## Access and prereveal declaration

- `final_labels_seen: false`
- `secret_seed_seen: false`
- `private_evaluator_seen: false`
- Custody repository/path accessed: `false`
- Sealed manifest bytes or contents accessed: `false`
- `C:\src\POPGP\POPGP_Codex_Handoff.md` accessed: `false`
- Builder calibration memo contents or conclusions accessed: `false`
- Builder conclusions supplied before attack construction: `false`
- Public `VIA-000-ATTACK-PLAN-1.md` read: `true`, as required by the amended handoff
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because both exact model identities
  are recorded as `unknown`
- External scientific validation: `false`
- Restricted boundary crossed: `false`

I read the launch brief, campaign, VIA-000 packet, primary protocol, viability plan,
review workflow, reviewer-identity guidance, pre-holdout amendment, and public first
attack plan completely. I did not inspect a custody path, sealed manifest, final label,
builder memo, or private evaluator material.

## Harness and clean controls

The mutation harness was a separate detached Git worktree at the exact candidate. The
artifact worktree remained at the amended handoff and received no implementation or
protocol edits. The harness was initialized with:

```powershell
git -C C:\src\POPGP worktree add --detach `
  C:\src\POPGP-via000-falsifier-2-candidate `
  9a29e05f803666bf0e3a28417ea399e3e26769fc
uv sync --frozen
```

The amended isolated checker body was executed both with system Python and through
the locked environment:

```powershell
$check = "import os,subprocess,sys; p=subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True); sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); blocked=bool(os.environ.get('PYTHONPATH')); sys.stderr.write('PYTHONPATH must be unset\n' if blocked else ''); raise SystemExit(p.returncode or bool(p.stdout) or blocked)"
python -I -c $check
uv run python -I -c $check
```

Clean control results before mutation were:

```text
clean_preflight_exit=0
baseline_checker_exit=0
clean_postflight_exit=0
Validation artifact contracts and required visual outputs are valid.
```

The same three checks returned zero after all probes and cleanup. Normal full
porcelain status and `git diff --exit-code` were empty/zero in both worktrees.

## Seven frozen mutation-family executions

Every JSON mutation below was applied only in the disposable worktree and reversed
before the next probe. The exact checker command was:

```powershell
uv run python scripts/check_validation_artifacts.py
```

### VIA000-B01 — unregistered noninformational check identity

Mutation: append
`{"name":"unregistered_required_probe","passed":true}` to
`examples/physics_qg/source_law/results/validation.json:$.checks`.

Observed exit: `1`.

```text
$.checks: array length changed from 4 to 5
non-informational check set is not explicitly registered
(missing=[], unregistered=['unregistered_required_probe'])
```

Result: **rejected**.

### VIA000-B02 — required pass flip

Mutation: change
`source_law/results/validation.json:$.checks[0].passed` from `true` to `false`.

Observed exit: `1`.

```text
$.checks[0].passed: changed from True to False
overall_pass=True but non-informational conjunction is False
```

Result: **rejected**.

### VIA000-B03 — measurement-array shape change

Mutation: remove the final value from
`source_law/results/validation.json:$.measurements.relative_entropy`, reducing the
frozen length from 16 to 15.

Observed exit: `1`.

```text
$.measurements.relative_entropy: array length changed from 16 to 15
```

Result: **rejected**.

### VIA000-B04 — stable configuration drift

Mutation: change
`chain_1d/results/validation.json:$.config.beta` from `1.0` to `1.01`.

Observed exit: `1`.

```text
$.config.beta: stable input drift exceeds rel_tol=1.77636e-15, abs_tol=0
(1.0 -> 1.01, abs=0.01, rel=0.00990099)
```

Result: **rejected**.

### VIA000-B05 — required visual empty and untracked

The original visual SHA-256 was
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.
The executed mutation and check were:

```powershell
$v = 'examples/physics_qg/source_law/results/source_scaling.png'
git rm --cached --quiet -- $v
[System.IO.File]::WriteAllBytes((Resolve-Path -LiteralPath $v), [byte[]]@())
uv run python scripts/check_validation_artifacts.py
git status --porcelain=v1 --untracked-files=all -- $v
```

Observed exit: `1`; status showed the staged deletion and untracked replacement.

```text
required visual is not tracked by git
required visual is empty
```

Result: **rejected**. Exact-candidate restoration reproduced the original SHA-256 and
left no index or worktree diff.

### VIA000-B06 — sensitive diagnostic beyond threshold

Mutation: set both
`source_law_many_body/results/validation.json:$.measurements.quadratic_coefficient_relative_error`
and the duplicate under `$.checks[0].value` from
`3.812167720149195e-08` to `0.001`.

Observed exit: `1`.

```text
$.checks[0].value.quadratic_coefficient_relative_error: sensitive diagnostic drift
exceeds rel_tol=0, abs_tol=0.0002
$.measurements.quadratic_coefficient_relative_error: sensitive diagnostic drift
exceeds rel_tol=0, abs_tol=0.0002
```

Result: **rejected** at both paths.

### VIA000-B07 — amended root sitecustomize/PYTHONPATH attack

An untracked repository-root `sitecustomize.py` set
`POPGP_UNDECLARED_STATE=active`. With `PYTHONPATH=.` the executed sequence was:

```powershell
$prior = $env:PYTHONPATH
try {
  $env:PYTHONPATH = '.'
  uv run python -c "import os; print(os.environ.get('POPGP_UNDECLARED_STATE'))"
  git diff --exit-code
  uv run python scripts/check_validation_artifacts.py
  python -I -c $check
  uv run python -I -c $check
} finally {
  $env:PYTHONPATH = $prior
}
```

Observed:

```text
active
A07_ordinary_exit=0
A07_git_diff_exit=0
A07_artifact_checker_exit=0
?? sitecustomize.py
PYTHONPATH must be unset
A07_isolated_preflight_exit=1
?? sitecustomize.py
PYTHONPATH must be unset
A07_isolated_postflight_exit=1
```

Result: **rejected by both amended isolated boundary checks**. This independently
confirms that amendment 1 closes the exact A07 counterexample from attack plan 1.

## New lethal boundary attack

### VIA000-B08 — ignored locked-environment sitecustomize survives the gate

The protocol claims exact rejection of undeclared import state and complete
tracked/untracked repository cleanliness, but both Git status commands intentionally
exclude ignored files. `uv sync --frozen` also reuses an existing `.venv` and does not
remove an otherwise unregistered `sitecustomize.py` from its site-packages directory.

I placed the following disposable file at
`.venv/Lib/site-packages/sitecustomize.py`, a Git-ignored path:

```python
import os

os.environ["POPGP_IGNORED_STATE"] = "active"
```

No `PYTHONPATH` was set. The executed sequence was:

```powershell
$prior = $env:PYTHONPATH
try {
  $env:PYTHONPATH = $null
  python -I -c $check
  uv sync --frozen
  uv run python -c "import os; print(os.environ.get('POPGP_IGNORED_STATE'))"
  uv run python scripts/check_validation_artifacts.py
  git diff --exit-code
  uv run python -I -c $check
  git status --porcelain=v1 --untracked-files=all
  git status --porcelain=v1 --ignored --untracked-files=all |
    Select-String 'sitecustomize.py'
} finally {
  $env:PYTHONPATH = $prior
}
```

Observed:

```text
A08_preflight_exit=0
A08_uv_sync_exit=0
active
A08_ordinary_exit=0
Validation artifact contracts and required visual outputs are valid.
A08_artifact_checker_exit=0
A08_git_diff_exit=0
A08_postflight_exit=0
A08_normal_porcelain=
A08_ignored_porcelain=!! .venv/Lib/site-packages/sitecustomize.py
```

This mutation is **accepted**. The ordinary protocol Python process demonstrably
executes undeclared code, yet locked sync, the artifact checker, Git diff, the amended
normal-status preflight, and the amended isolated postflight all return zero. Python
`-I` does not imply `-S`; when run through the virtual environment it still imports
`sitecustomize` from site-packages before executing the checker body. The benign probe
only set an environment variable; an adversarial shim could alter imports, subprocess
behavior, evidence, or the checker itself.

The source and its generated `sitecustomize.cpython-311.pyc` were removed by exact
path. A subsequent ordinary Python control printed `None`; clean preflight,
postflight, and artifact checks returned zero; normal porcelain status and Git diff
were clean.

## Validator result

At the exact falsifier parent, after `uv sync --frozen`, I ran:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Observed exit: `0`.

```text
Viability campaign contract is valid.
```

This validates the static campaign contract. It does not exercise or detect B08, so it
does not overturn the dynamic counterexample.

## Falsifier conclusion and holdout recommendation

All seven frozen mutation families reject, and the exact root-level
`sitecustomize.py` plus `PYTHONPATH=.` defect from attack plan 1 is repaired. The
amended protocol nevertheless remains **unsafe to advance to `attacked` /
`holdout_started: true`** because B08 is a reproducible accepted mutation inside the
same undeclared-state and evidence-integrity threat model.

The frozen hypothesis requires a run that does not depend on undeclared builder state,
and the statistical analysis says undeclared import state or repository residue is a
failure. The current command sequence cannot establish those predicates: it neither
forces recreation of the locked environment nor inventories executable ignored
environment files, and its postflight interpreter can itself load the hidden shim.
This is a protocol/evidence-integrity defect discovered before holdout, not a
scientific-candidate failure and not a packet adjudication.

Before holdout, a new protocol amendment should ensure that the environment path does
not preexist, create it outside the repository or destroy/recreate it from the lock,
inventory or hash executable startup surfaces after sync (including `sitecustomize`,
`usercustomize`, and `.pth` files), and run the integrity checker with a trusted
interpreter configuration that cannot import the environment being checked. Those are
required properties, not implementation changes made by this falsifier.

## Limitations and non-claims

- No Linux/Windows reproduction, PDF run, future raw-results receipt, custody reveal,
  final label, or output commitment was available to this seat.
- Receipt-level attacks from plan 1 remain obligations once immutable runner receipts
  exist; I did not invent future evidence to execute them now.
- B08 demonstrates a gate-level accepted mutation, not evidence that an honest prior
  run actually contained the shim.
- This artifact repairs no implementation or protocol, assigns no packet or campaign
  outcome, and establishes no Tier R, mechanism, gravitational, native/CUDA, or
  external-validation claim.
