# Launching an independent agent review

This runbook turns [the POPGP review workflow](../governance/AGENT_REVIEW_WORKFLOW.md)
into concrete Git commands and copy/paste agent prompts. Use a fresh agent task/session
and, when possible, a different model from the builder.

For a multi-packet scientific campaign, first define the target tier and packet
contracts in the
[adversarial viability demonstration plan](../scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md),
then apply this review runbook to each frozen implementation candidate. Packet
adjudication does not replace independent implementation review.

Before reviewing a preregistered or adjudicated campaign packet, run the executable
contract command specified in the viability plan and retain its exact output. Schema
validity alone is insufficient; the validator also enforces receipt hashes, custody,
dependencies, outcome exclusivity, evidence floors, and review-chain closure.

## 1. Freeze and record the candidate

From the POPGP repository root, confirm the intended branch is clean and commit the
candidate before review:

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse "HEAD^{tree}"
git merge-base HEAD origin/master
```

Record the full candidate, tree, and baseline hashes. If `git status --short` is not
empty, either commit the intended candidate or stop and separate unrelated work. Never
ask a reviewer to audit an ambiguous working tree.

Run the current repository quality suite and retain exact output:

```powershell
uv sync --frozen --no-editable
uv run --frozen --no-editable ruff check .
uv run --frozen --no-editable python scripts/check_tex.py
uv run --frozen --no-editable python -m pytest -q -p no:cacheprovider
uv run --frozen --no-editable python -m examples.physics_qg.chain_1d
uv run --frozen --no-editable python -m examples.physics_qg.grid_2d
uv run --frozen --no-editable python -m examples.physics_qg.gravity_well
uv run --frozen --no-editable python -m examples.physics_qg.source_law
uv run --frozen --no-editable python -m examples.physics_qg.source_law_many_body
uv run --frozen --no-editable python -m examples.physics_qg.ca_model
uv run --frozen --no-editable python -m scripts.check_validation_artifacts --enforce-change-boundary
```

`.github/workflows/ci.yml` is authoritative. The README and this runbook intentionally
repeat its quality-command set, and a regression test requires all three to remain in
sync.

## 2. Create an isolated reviewer worktree

Choose a short task slug and a new path. Substitute the full candidate hash:

```powershell
git worktree add -b review/<task>-1 C:\src\POPGP-review-<task>-1 <candidate-40-hex>
git -C C:\src\POPGP-review-<task>-1 status --short
git -C C:\src\POPGP-review-<task>-1 rev-parse HEAD
```

The last two commands must show a clean tree and the exact candidate hash. Do not reuse
an existing review worktree without verifying its branch, HEAD, and cleanliness.

## 3. Launch a fresh reviewer task

Open a new agent task/session with no builder conclusions in its context. Provide the
frozen tree, baseline, scope, commands, and access boundary. Paste and fill this prompt:

```text
You are the independent reviewer for POPGP task <task>.

Before reviewing, read these files completely:
- docs/governance/AGENT_REVIEW_WORKFLOW.md
- docs/governance/REVIEWER_IDENTITY.md
- docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md

Frozen reviewer worktree: <absolute-review-worktree>
Candidate commit: <candidate-40-hex>
Baseline commit: <baseline-40-hex>
Context hash command: git rev-parse "<candidate-40-hex>^{tree}"
Review ID: <stable-review-id>
Access boundary: public-repository-only; no final labels, secret seeds,
private evaluator logic, credentials, or unrestricted private hardware data.

Independently inspect the complete candidate diff and relevant surrounding code,
equations, scientific premises, claims, tests, CI, generated JSON, and visual
artifacts. Run the repository quality suite and targeted counterexamples or mutation
tests where feasible. For every issue, provide a stable finding ID, exact evidence,
failure scenario, consequence, required action, verification level, severity, and
blocking status. Give every requested test a stable ID. Do not modify implementation,
tests, or documentation.

Create reviews/independent_reviewer/<review-id>.md from the independent-review
template. Commit only that review artifact on the reviewer branch, leave the worktree
clean, and return the full review commit hash plus approve/changes-requested and the
blocking count. Record the exact model identity, version, operator, access, and any
shared context honestly.
```

If a reviewer is launched as a subagent from the builder's task, disclose the shared
operator/orchestrator and inherited context. That is useful process separation but not
external scientific independence.

## 4. Remediate a changes-requested review

Push the reviewer branch so the artifact is durable. Start remediation from the full
review commit, not merely from the candidate:

```powershell
git worktree add -b codex/<task>-remediation C:\src\POPGP-<task>-remediation <review-commit-40-hex>
```

The builder reads the immutable review and
[review-response template](../templates/REVIEW_RESPONSE_TEMPLATE.md), addresses every
finding and requested test, commits fixes before the response where possible, reruns
the full suite/CI, and records the final response-containing hash in the handoff.

## 5. Launch the re-review

Create a new worktree from the exact remediation handoff:

```powershell
git worktree add -b review/<task>-rereview-1 C:\src\POPGP-review-<task>-rereview-1 <remediation-40-hex>
```

Launch the independent reviewer with this prompt:

```text
Perform independent re-review for POPGP review <review-id>.

Read completely:
- docs/governance/AGENT_REVIEW_WORKFLOW.md
- docs/governance/REVIEWER_IDENTITY.md
- docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md
- <prior-review-artifact>
- <builder-response-artifact>

Frozen re-review worktree: <absolute-rereview-worktree>
Remediation commit: <remediation-40-hex>
Prior review commit: <prior-review-commit-40-hex>
Access boundary: <declared-access-boundary>

Independently verify every prior finding and requested test, the complete remediation
diff, exact CI evidence, regressions, residual risks, and new findings. Do not accept
builder dispositions as proof and do not modify implementation files. Assign
verified-resolved, unresolved, or superseded to every finding and
verified-satisfied, unresolved, or superseded to every requested test.

Create reviews/independent_reviewer/<review-id>-REREVIEW-1.md, commit only that
artifact, leave the worktree clean, and return the full review commit hash plus the
fresh recommendation and blocking count. approve: true requires zero unresolved
blocking findings. Complete both the typed independence declaration and the prose
independence statement.
```

Repeat with incremented response/re-review round numbers if a blocker remains. Never
replace a prior artifact.

## 6. Completion checklist

Before treating the review gate as complete, verify:

- initial review, builder response, and latest re-review are committed and pushed;
- every artifact names the intended full hashes and a reproducible context hash;
- every finding and requested test has a final independent outcome;
- latest recommendation is `approve: true` with zero blockers;
- exact remediation-head CI is green;
- reviewer and builder worktrees are clean; and
- a maintainer or PI separately authorizes merge.

Completed POPGP review artifacts under `reviews/` provide concrete examples of the
format, but new reviews must use fresh IDs, hashes, evidence, and access declarations.
