# VIA-000 public attack plan 7

## Seat and artifact metadata

- Artifact ID: `VIA-000-ATTACK-PLAN-7`
- Campaign / packet / role: `POPGP-VIABILITY-R1-2026-08` / `VIA-000` /
  `falsifier`
- Agent identity: `codex-via000-falsifier`
- Model identity: `unknown` — the runtime describes this seat as a Codex agent based
  on GPT-5, but exposes no operator-supplied exact served-model identifier.
- Model version/snapshot: `unknown`
- Operator: `fuocor` (campaign declaration; local OS account path `rfuoco`)
- Session ID: `popgp-viability-r1-2026-08-via000-falsifier-session-7`
- Orchestrator ID: `codex-desktop`
- Agent task: `/root/via000_falsifier_7`
- Organization / access: `popgp-internal` / `public-calibration-only`
- Preparation date: `2026-08-19` (`America/New_York` operator context)

The exact model fields remain `unknown`; neither the frozen seat label nor the
general runtime description supplies an exact served-model identifier or snapshot.

## Frozen identities independently verified

- Exact falsifier checkout / attack-plan parent:
  `c9dafa18b051b261ed8ed3d8701ac3ff8bc10187`
- Checkout tree: `a143f2a7041db4d24d5330546a2f51df949bfe82`
- Checkout parent / activated campaign handoff named by the launch brief:
  `22a87abac058e3fdaa7503e6e72c1c45c78ccee2`
- Protocol snapshot: `72bfcfbb5ab5a0fee3449510f8868b8bb19be805`
- Protocol-snapshot tree: `944ed1dbc47f04d74b80f575641e90f48e17645d`
- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`
- Baseline: `70c867552279b74d5ce1a7bc5c50d5a980cf81e6`
- Frozen VIA-000 rule SHA-256:
  `e5ac4103188f652a460b4a0cc74d641d694e0043b6f2d62dcf2484b92de8acca`
- Primary-protocol and protocol-receipt raw-byte SHA-256:
  `66d9cc3325a9758c7443b29df612e2b27b8db14de3b224489dee5e2e04a718f8`
- Protocol-manifest raw-byte SHA-256:
  `8b6d3b83fc374b32ad558aacfcce319160ac955353c93a4d89919f91d8fd5d0f`
- Artifact worktree / branch: `C:\src\POPGP-via000-falsifier-7` /
  `campaign/via000-falsifier-7`
- Disposable exact-candidate worktree:
  `C:\src\POPGP-via000-falsifier-7-candidate` (detached)

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
- Public attack plans 1 through 6 read: `true`, as explicitly required
- Public pre-holdout amendments 1 through 6 read: `true`, as explicitly required
- Shared operator with builder: `true` (packet declaration)
- Shared orchestrator with builder: `true` (packet declaration)
- Shared session with builder: `false`
- Model separation from builder: `not decidable`, because the exact builder and
  falsifier model identities are both recorded as `unknown`
- External scientific validation: `false`
- Restricted custody/holdout boundary crossed: `false`

I read the launch brief, all six amendments, all six prior public attack plans,
campaign, VIA-000 packet, primary protocol, viability plan, review workflow, and
reviewer-identity guidance completely. I did not open a custody location, sealed
manifest, builder memo, final label, private evaluator material, or the local handoff
file named above.

## Harness and clean controls

The clean and mutation harness was a new detached worktree at the exact scientific
candidate. Host metadata was:

```text
Microsoft Windows 11 Enterprise 10.0.26200 (build 26200)
base Python 3.12.10; sys.prefix == sys.base_prefix
uv 0.11.11 (ed7b06001 2026-05-06 x86_64-pc-windows-msvc)
locked/isolated Python 3.11.15
pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
```

Before the counted sequence, an initial lint/TeX tool probe was discarded, its
`.venv` and `.ruff_cache` were removed by exact validated paths, and both normal and
ignored Git status were empty. The following sequence was then executed serially.

| Frozen command or gate | Observed result |
|---|---|
| exact base-interpreter preflight | exit `0`; no output |
| `uv sync --frozen --no-editable` | exit `0`; Python 3.11.15; 60 locked packages |
| isolated/frozen/non-editable `ruff check .` | exit `0`; `All checks passed!` |
| isolated/frozen/non-editable `python scripts/check_tex.py` | exit `0`; 652 lines; balanced source |
| `python -m pytest -q` with all frozen uv flags | exit `0`; exactly `187 passed in 1150.61s (0:19:10)`; wrapper 1193.7 s |
| `python -m examples.physics_qg.chain_1d` | exit `0` |
| `python -m examples.physics_qg.grid_2d` | exit `0`; retained inadmissible-decomposition warning |
| `python -m examples.physics_qg.gravity_well` | exit `0`; retained inadmissible-decomposition warning |
| `python -m examples.physics_qg.source_law` | exit `0` |
| `python -m examples.physics_qg.source_law_many_body` | exit `0` |
| `python -m examples.physics_qg.ca_model` | exit `0` |
| isolated semantic artifact checker | exit `0`; contracts and visuals valid |
| exact external-directory creation | exit `0`; directory was initially absent |
| first exact external pdfTeX pass | exit `0`; 11 pages; 531345 bytes |
| second exact external pdfTeX pass | exit `0`; 11 pages; 535368 bytes |
| exact `git diff --exit-code` | exit `0` |
| exact base-interpreter postflight | exit `0`; no normal repository residue; only expected `_virtualenv.pth` |

The two PDF passes used the literal frozen command and fixed adjacent directory
`C:\src\POPGP-VIABILITY-R1-2026-08-VIA-000-PDF`. The final PDF engine, page count,
and byte count were reported by pdfTeX itself. The host's bundled `pdfinfo.cmd`
wrapper could not resolve its backend; that auxiliary inspection failure was not a
frozen command and did not replace or weaken the pdfTeX log. The retained external
artifacts before exact cleanup were:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `framework.aux` | 8223 | `f5b8d3d2fe8a51aaa7cb929609df21ab33a7feee4d2cc4b6445d62be5a9174b4` |
| `framework.log` | 26406 | `dbe0acaa15e0c4ed16575dc2ce315b84671f137ef0f23852e07dcadd0fdb9c19` |
| `framework.out` | 13920 | `e46d0300a92cfff300c634cd1a9dfd595f73000caf93fece37704482c8502854` |
| `framework.pdf` | 535368 | `2bd837f8c22d0bc752a2b2d8e465d031e99b65a64888a0a4170ed23d6f6cc162` |
| `frameworkNotes.bib` | 104 | `61b9ca8a0eae79078bb6bcd4205c582b5f4f2ed0a16ece16a01837cc4718248d` |

This independently closes the precise clean-control PDF/postflight contradiction
reported by Plan 6: the declared PDF build leaves the candidate repository clean.

## Ten frozen mutation-family executions

Each mutation was executed only in the disposable exact-candidate worktree. Tracked
targets were restored from the exact candidate before the next family. Named attack
files, bytecode, carriers, and sentinels were removed by exact validated paths.

### VIA000-G01 — unregistered noninformational check

Appending `unregistered_required_probe` to the source-law checks made the exact
isolated non-editable checker exit `1`. It reported array length `4 -> 5` and the
unregistered identity. Result: **rejected**.

### VIA000-G02 — required pass flip

Changing `relative_entropy_is_quadratic.passed` from `true` to `false` made the
checker exit `1`, reporting the exact drift and contradiction with
`overall_pass=true`. Result: **rejected**.

### VIA000-G03 — measurement-array shape change

Reducing the source-law relative-entropy array from 16 values to 15 made the checker
exit `1` with the exact length diagnostic. Result: **rejected**.

### VIA000-G04 — stable configuration drift

Changing chain beta from `1.0` to `1.01` made the checker exit `1` with drift beyond
`rel_tol=1.77636e-15, abs_tol=0`. Result: **rejected**.

### VIA000-G05 — required visual empty and untracked

Removing the source-scaling visual from the index and replacing its worktree bytes
with an empty file made the checker exit `1` with both `not tracked by git` and
`empty`. Exact restoration reproduced SHA-256
`643b379661570729d8501cbe109d6cf68bfe9c38d39612462d5ac92ef9eca607`.
Result: **both defects rejected**.

### VIA000-G06 — sensitive diagnostic beyond threshold

Changing both canonical copies of `quadratic_coefficient_relative_error` from
`3.812167720149195e-08` to `0.001` made the checker exit `1` at both paths beyond
`abs_tol=0.0002`. Result: **rejected**.

### VIA000-G07 — root customize plus PYTHONPATH

An untracked root `sitecustomize.py` set `POPGP_G07_ROOT=active`, with process-local
`PYTHONPATH=.`. Isolated non-editable Python printed `active`; the semantic checker
and `git diff --exit-code` both returned `0`. The exact base postflight printed
`?? sitecustomize.py`, reported `blocked environment: PYTHONPATH`, and exited `1`.
Result: **rejected by the complete frozen boundary**.

### VIA000-G08 — ignored customize hooks before and after sync

With only ignored preplanted project-environment `sitecustomize.py` and
`usercustomize.py` present, base Python `-I` saw neither marker and the exact preflight
printed `!! .venv/` and exited `1`. After a clean non-editable sync, project Python
executed the site marker, while isolated non-editable Python saw neither marker and
used a temporary uv-cache prefix. Exact postflight listed both hooks and exited `1`.
Result: **both placements rejected**.

### VIA000-G09 — executable and modified `.pth` surfaces

An extra executable `via000_g09_extra.pth` set its marker under project Python but
not under isolated non-editable Python. Postflight listed both `.pth` files and exited
`1`. An executable line appended to allowed `_virtualenv.pth` likewise ran only under
project Python; isolated execution saw no marker and postflight exited `1` on the
content mismatch. Result: **extra and modified variants rejected**.

### VIA000-G10 — editable self-cleaning carrier

The historical exploit was independently reproduced under its vulnerable editable
precondition. After `uv sync --frozen`, an executable line was added to
`_editable_impl_popgp.pth`. `uv run --isolated --frozen python` copied and executed
the carrier in a temporary environment, printed marker `active`, and restored the
project carrier before the command body; Git diff stayed `0`.

The carrier was then recreated with an external sentinel action and the exact old
isolated/frozen semantic-checker command was run. The checker exited `0`, the sentinel
changed from absent to present, the project carrier self-cleaned, and Git diff again
returned `0`. This directly proves checker startup control without relying on Plan 4
or Plan 5's observation.

Finally, the carrier was recreated and the amended
`uv sync --frozen --no-editable` was run. Sync exited `0` and removed the editable
carrier. Exact isolated/frozen/non-editable Python reported the marker `None`; the
carrier remained absent; and exact postflight exited `0`. Result: **historical
exploit reproduced, then eliminated by the complete amended boundary**.

## Additional boundary attacks

- **Repository-local PDF output:** the old local pdfTeX command exited `0` and made an
  11-page PDF; Git diff remained `0`, but postflight listed the five local TeX files
  and exited `1`. Exact cleanup restored empty normal status.
- **Dirty tracked regeneration:** adding one JSON whitespace byte preserved semantic
  checker exit `0`, while `git diff --exit-code` returned `1`. Exact-candidate bytes
  were restored.
- **Pre-existing external evidence directory:** rerunning the exact creation command
  while the directory existed raised the frozen assertion and exited `1`.
- **Receipt-level evidence attacks:** OS-only success, summary-only evidence, missing
  commands, wrong identities/counts, PDF metadata corruption, and post-output
  selection remain mandatory attacks against future immutable runner receipts. No
  such receipt exists yet, so this seat did not fabricate one to mutate.

No new accepted public mutation or clean-control contradiction was found.

## Static validator outcome

At exact checkout `c9dafa18b051b261ed8ed3d8701ac3ff8bc10187`, before this
artifact-only commit, I ran:

```powershell
uv run --isolated --frozen --no-editable python scripts/check_viability_campaign.py `
  reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml
```

It exited `0` with:

```text
Viability campaign contract is valid.
```

The packet-rule helper exited `0` and returned
`e5ac4103188f652a460b4a0cc74d641d694e0043b6f2d62dcf2484b92de8acca`.

## Falsifier conclusion and holdout recommendation

The sixth amendment resolves the exact Plan 6 PDF/postflight feasibility failure.
The entire frozen Windows clean sequence succeeds, including the mandatory 187 tests,
six example regenerations, semantic contract check, initially absent external output
directory, both retained PDF passes, Git diff, and startup-surface postflight. All ten
frozen mutation families independently reject under the complete amended boundary.
The self-cleaning editable exploit remains reproducible under its old precondition but
cannot survive the required non-editable sync and isolated non-editable execution.

Recommendation: **safe to record `attacked` / `holdout_started: true` for VIA-000**,
provided this Plan 7 artifact is hash-bound as the decisive attack-plan receipt and
the campaign remains validator-clean. This recommendation authorizes no reveal and
does not substitute for the required fresh Linux and Windows reproduction-runner
receipts, output commitment, custody reveal, statistical audit, claim audit, or
adjudication.

## Limitations and non-claims

- Dynamic execution in this seat was Windows-only; Linux remains a runner obligation.
- The external PDF directory was deliberately removed after hashing, as required for
  a disposable falsifier qualification; its hashes and pdfTeX metadata are retained
  above, not promoted as the future runner receipt.
- No immutable runner raw-results receipt, output commitment, custody reveal, final
  label, statistical audit, claim audit, or adjudication was available.
- Benign startup payloads set markers, wrote an external sentinel, or restored their
  carrier. They do not allege that an honest run used an attack.
- This artifact changes no implementation, protocol, packet, campaign lifecycle,
  threshold, measurement, mutation rule, or custody state; it assigns no packet
  outcome and establishes no Tier R, mechanism, gravitational, native/CUDA, or
  external-validation claim.
