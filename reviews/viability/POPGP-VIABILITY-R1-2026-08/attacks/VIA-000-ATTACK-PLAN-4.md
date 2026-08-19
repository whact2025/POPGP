# VIA-000 public attack plan 4

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-4`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this seat as a Codex agent based
  on GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-4`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_4`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The exact model fields remain `unknown`; neither the frozen seat label nor the
general runtime description is an exact served-model identifier or snapshot.

## Frozen identities independently verified

- Exact falsifier checkout / attack-plan parent:
  `0ad99e45c79d6bc7def7fffc1b4bb268cc8ffb95`
- Checkout tree: `3aa1cc5421aefbb61a7186458a0ecee7415226c2`
- Checkout parent / activated campaign handoff named by the launch brief:
  `502cafea22d581c06861a6c55eeaed72f74eccb0`
- Activated-handoff tree: `9df85585da387ea231ad202fa35511edb3083649`
- Protocol snapshot: `2894b408527e5c6457eeddcfb1ae657c9a205220`
- Protocol-snapshot tree: `58a35e1f426388f94c968525f5e451b69062adb1`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `78f91b4cef5913ce344bf40d89458c9d7dcb6648ca85443a60db1781f784e547`
- Primary-protocol raw-byte SHA-256:
  `37ef44edab45240d594ac77bd30df9bad97a7564decfe5881db96a341875bd4f`
- Protocol-receipt raw-byte SHA-256: the same
  `37ef44edab45240d594ac77bd30df9bad97a7564decfe5881db96a341875bd4f`
- Protocol-manifest raw-byte SHA-256:
  `3bcaff626ac314e885e28d255c25371b1b874535516ac8b3c7274549fd99266d`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-4` /
  `campaign/via000-falsifier-4`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-4-candidate` (detached)

The commits and trees were resolved from Git objects rather than accepted from branch
names. The artifact worktree was created from the exact requested handoff; all
mutations ran only in the separate detached scientific-candidate worktree.

## Access, independence, and prereveal declaration

- `final_labels_seen: false`
- `secret_seed_seen: false`
- `private_evaluator_seen: false`
- Custody repository or path accessed: `false`
- Sealed manifest bytes or contents accessed: `false`
- `C:\src\POPGP\POPGP_Codex_Handoff.md` accessed: `false`
- Builder VIA-000 calibration memo contents or conclusions accessed: `false`
- Builder conclusions supplied before attack construction: `false`
- Public attack plans 1, 2, and 3 read: `true`, as explicitly required
- Public pre-holdout amendments 1, 2, and 3 read: `true`, as explicitly required
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because the exact builder and
  falsifier model identities are both recorded as `unknown`
- External scientific validation: `false`
- Restricted custody/holdout boundary crossed: `false`

I read the launch brief, all three public amendments, all three public prior attack
plans, campaign, VIA-000 packet, primary protocol, viability plan, review workflow,
and reviewer-identity guidance completely. I did not open any calibration memo. The
public campaign README exposed an unrelated high-level statement that a builder-owned
VIA-300 calibration exists and remains nondecisive; no calibration file, numerical
result, VIA-000 builder conclusion, or such statement was used to choose or evaluate
an attack. This incidental public-tree exposure does not cross a sealed boundary, but
is disclosed rather than silently omitted.

## Harness, platform, and clean controls

The exact candidate was a new detached Git worktree. Before locked sync, both normal
and ignored porcelain status were empty. The frozen base-interpreter preflight exited
`0`. Host and tool metadata were:

```text
Windows-11-10.0.26200-SP0
base Python 3.12.10; sys.prefix == sys.base_prefix
uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
locked/isolated Python 3.11.15
```

`uv sync --frozen` exited `0` and created the declared project `.venv`. A clean
`uv run --isolated --frozen python` used a temporary prefix under the uv cache, not
the project `.venv`. The clean frozen postflight exited `0`. Its two expected files
were initially:

```text
.venv/Lib/site-packages/_editable_impl_popgp.pth
  C:\src\POPGP-via000-falsifier-4-candidate
.venv/Lib/site-packages/_virtualenv.pth
  import _virtualenv
```

The first file's initial SHA-256 was
`c5df48ad73bc40fd7381d6aaf70f91da769c9d608456b8d0a3bc69a380e8de09`;
the second's was
`69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d`.
After all probes, no attack hook or marker remained, normal porcelain status was
empty, `git diff --exit-code` returned `0`, and the exact frozen postflight again
returned `0`.

Every artifact mutation below used the frozen command:

```powershell
uv run --isolated --frozen python scripts/check_validation_artifacts.py
```

Each tracked target was restored from exact candidate
`9a29e05f803666bf0e3a28417ea399e3e26769fc` before the next probe. Every named
untracked or ignored attack file and attack bytecode was removed by exact path.

## Nine frozen mutation-family executions

### VIA000-D01 — unregistered noninformational check

Appended `{"name":"unregistered_required_probe","passed":true}` to
`source_law/results/validation.json:$.checks`.

Observed checker exit: `1`.

```text
$.checks: array length changed from 4 to 5
non-informational check set is not explicitly registered
(missing=[], unregistered=['unregistered_required_probe'])
```

Result: **rejected**.

### VIA000-D02 — required pass flip

Changed `source_law/results/validation.json:$.checks[0].passed` from `true` to
`false` while leaving `overall_pass` true.

Observed checker exit: `1`.

```text
$.checks[0].passed: changed from True to False
overall_pass=True but non-informational conjunction is False
```

Result: **rejected**.

### VIA000-D03 — measurement-array shape change

Removed the last value from
`source_law/results/validation.json:$.measurements.relative_entropy`, changing its
length from 16 to 15.

Observed checker exit: `1`.

```text
$.measurements.relative_entropy: array length changed from 16 to 15
```

Result: **rejected**.

### VIA000-D04 — stable configuration drift

Changed `chain_1d/results/validation.json:$.config.beta` from `1.0` to `1.01`.

Observed checker exit: `1` with the expected stable-input tolerance diagnostic.

```text
$.config.beta: stable input drift exceeds rel_tol=1.77636e-15, abs_tol=0
(1.0 -> 1.01, abs=0.01, rel=0.00990099)
```

Result: **rejected**.

### VIA000-D05 — required visual empty and untracked

Removed the index entry for
`examples/physics_qg/source_law/results/source_scaling.png` and replaced the
worktree file with zero bytes. Its original and restored SHA-256 was
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.

Observed checker exit: `1`; status showed a staged deletion plus untracked
replacement.

```text
required visual is not tracked by git
required visual is empty
```

Result: **both defects rejected**.

### VIA000-D06 — sensitive diagnostic beyond threshold

Set both the measurement and check-value copies of
`quadratic_coefficient_relative_error` in the many-body validation artifact from
`3.812167720149195e-08` to `0.001`.

Observed checker exit: `1` with sensitive-diagnostic drift at both JSON paths and
`abs_tol=0.0002`.

Result: **rejected at both paths**.

### VIA000-D07 — root sitecustomize plus PYTHONPATH

An untracked repository-root `sitecustomize.py` set `POPGP_D07_ROOT=active`. With
process-local `PYTHONPATH=.`:

```text
uv run --isolated --frozen python: marker=active; exit=0
artifact checker: exit=0
git diff --exit-code: exit=0
frozen base postflight: ?? sitecustomize.py; blocked environment: PYTHONPATH; exit=1
```

The uv environment isolation flag does not clear `PYTHONPATH`, but the frozen base
postflight rejects both the environment and the untracked source. Result:
**rejected by the combined frozen gate**.

### VIA000-D08 — ignored customize hooks before and after sync

I placed both `.venv/Lib/site-packages/sitecustomize.py` and
`usercustomize.py` with distinct benign markers. The base interpreter with `-I`
reported both markers `None`; the frozen preflight reported ignored `.venv` state and
exited `1`. With the same two hooks present after a clean locked sync:

```text
uv run --isolated --frozen python: both markers=None; temporary uv-cache prefix; exit=0
frozen postflight: printed both hook paths; exit=1
```

The project `.venv` customize files therefore did not control isolated Python, and
the preflight/postflight boundaries rejected both named placements. Result:
**rejected**.

### VIA000-D09 — executable and replaced `.pth` startup surfaces

#### Extra executable file

Added `.venv/Lib/site-packages/via000_extra.pth` with a benign environment marker
and sentinel action. Ordinary `uv run python` used the project `.venv`, reported the
marker `active`, and created the sentinel. After removing that sentinel, the frozen
isolated Python reported the marker `None`, used a temporary uv-cache prefix, and
created no sentinel. The frozen postflight printed all three `.pth` paths and exited
`1`. Result: **the simple extra-file attack is rejected**.

#### Static modification and full replacement of each allowed file

- Appending an executable line to `_editable_impl_popgp.pth` controlled ordinary
  Python and, importantly, also controlled isolated Python. The postflight saw the
  changed content and exited `1` while it remained changed.
- Replacing `_editable_impl_popgp.pth` completely with unexpected executable content
  made the startup-surface comparison exit `1`.
- Replacing `_virtualenv.pth` with executable marker content did not control isolated
  Python; uv generated its own clean `_virtualenv.pth` in the temporary environment.
  The project postflight exited `1`.
- A complete unexpected-content replacement of `_virtualenv.pth` also made the
  startup-surface comparison exit `1`.

These are expected rejections, but the first result shows that `--isolated` does not
make the allowed editable `.pth` inert. Inspection from inside the temporary isolated
Python showed:

```text
prefix=C:\Users\rfuoco\AppData\Local\uv\cache\builds-v0\.tmp8p5JFn
temporary _editable_impl_popgp.pth bytes:
  C:\src\POPGP-via000-falsifier-4-candidate
  import os; os.environ["POPGP_D09_EDITABLE_MOD"]="active"
marker=active
```

uv copied the project environment's allowed editable `.pth` bytes into its fresh
isolated environment.

#### Decisive self-cleaning allowed-file mutation

I replaced the allowed editable `.pth` with these two lines before frozen execution:

```text
C:\src\POPGP-via000-falsifier-4-candidate
import os,pathlib; os.environ["POPGP_D09_SELFCLEAN"]="active"; pathlib.Path(".venv/Lib/site-packages/_editable_impl_popgp.pth").write_text(str(pathlib.Path.cwd().resolve())+"\n",encoding="utf-8")
```

The executable line was copied into the temporary isolated environment. At Python
startup it set the benign marker and rewrote the project `.pth` back to the content
accepted by the frozen postflight before the requested Python body ran. Observed:

```text
uv run --isolated --frozen python:
  POPGP_D09_SELFCLEAN=active
  prefix=C:\Users\rfuoco\AppData\Local\uv\cache\builds-v0\.tmpH7Pd5p
  exit=0
project editable .pth after startup:
  C:\src\POPGP-via000-falsifier-4-candidate
git diff --exit-code: exit=0
frozen postflight: exit=0
```

I then repeated the attack against the exact frozen semantic-checker command rather
than only a probe:

```text
Validation artifact contracts and required visual outputs are valid.
artifact checker exit=0
git diff --exit-code=0
frozen postflight exit=0
```

The rewritten project `.pth` had SHA-256
`839f3e9b302d4fc9e3558b1aa6d9e9c45efc7be9e00fd10dcbb21e62b45b16d8`.
That differs from the clean sync file's initial byte hash because the postflight
compares stripped text, not exact frozen bytes; it still exited `0`.

Result: **accepted mutation**. The allowed editable startup file can control every
frozen isolated Python invocation and erase the persistent evidence that the
postflight checks. This is a time-of-check/self-erasure variant inside the exact C09
startup-surface family, not a new scientific hypothesis.

The native `ruff` executable does not itself process Python `.pth` files. Every frozen
Python entry point (`check_tex.py`, pytest, the six examples, and the semantic
checker), however, starts the same isolated Python environment whose copied editable
`.pth` executes before the command body. The exact checker reproduction above is
sufficient to show that at least one required frozen command and the evidence gate
itself can be controlled while all final gates return zero; broad reruns cannot cure
that accepted mutation.

## Equivalent environment and evidence-boundary probes

The prior public receipt-level attacks remain obligations once immutable runner
receipts exist. In particular, both platform records, exact command records, raw
stdout/stderr, exact identities, test/example/mutation/PDF counts, two PDF passes,
nonempty PDF metadata, and hash-bound support for every Boolean remain necessary. A
summary-only Boolean cannot cure missing raw evidence. No runner receipt exists yet,
so I did not invent one to mutate.

This round additionally observed that importing the editable project from a clean
isolated environment creates ignored repository `__pycache__` directories outside
`.venv`, while the frozen postflight still exits `0`. I did not rely on or promote a
bytecode exploit because D09 already supplies a direct accepted mutation. The
observation is retained as a future boundary attack: ignored executable bytecode must
not become an unmeasured substitute startup surface.

## Static validator result

At exact checkout `0ad99e45c79d6bc7def7fffc1b4bb268cc8ffb95`, before this
artifact-only commit, I ran:

```powershell
uv run --isolated --frozen python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

Observed exit: `0`.

```text
Viability campaign contract is valid.
```

The canonical packet-rule helper also returned the frozen rule SHA-256 recorded
above. The static validator proves document/hash coherence; it does not dynamically
execute C09 and therefore does not contradict the accepted self-cleaning mutation.

## Falsifier conclusion and holdout recommendation

All nine frozen mutation families were independently exercised. D01 through D08
reject. C09's extra file, static replacements, customize variants, and modification
of `_virtualenv.pth` reject as intended. Nevertheless, the third amended handoff is
**unsafe to transition to `attacked` / `holdout_started: true`** because the allowed
editable `.pth` supplies a reproducible accepted self-cleaning mutation.

The frozen hypothesis requires fail-closed rejection of every preregistered evidence
corruption and no undeclared import state. Instead, a mutation of an explicitly
allowed startup file is copied into the supposedly isolated environment, executes
before the frozen checker, restores the only persistent bytes inspected later, and
leaves the semantic checker, Git diff, and exact postflight all successful. Advancing
would assert `mutation-rejection=true` when the frozen command sequence cannot
establish it.

This is a pre-holdout protocol/evidence-integrity defect, not a scientific-candidate
failure and not packet adjudication. A repair must not trust mutable project `.venv`
editable-install bytes when constructing an isolated environment, and must bind or
check the startup input before it can execute rather than only comparing self-mutable
state afterward. A fifth fresh falsifier should challenge any new refreeze. This seat
does not implement that repair.

## Limitations and non-claims

- No Linux runner, immutable runner raw-results receipt, output commitment, custody
  reveal, final label, statistical audit, or adjudication was available.
- D09 proves controlled undeclared startup execution and a passing frozen gate. The
  benign payload changed only an environment marker and restored its own carrier; it
  does not allege that an honest prior run used the attack.
- The isolated environment is fresh in location but its editable startup bytes are
  demonstrably derived from mutable project `.venv` state on this Windows platform.
  Linux still requires runner coverage, but a Windows-accepted mutation is already
  sufficient to refute cross-platform fail-closed rejection.
- This artifact changes no implementation, protocol, packet, campaign lifecycle,
  threshold, measurement, or custody state; it assigns no packet outcome and
  establishes no Tier R, mechanism, gravitational, native/CUDA, or external-validation
  claim.
