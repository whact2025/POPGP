# Independent agent review workflow

This procedure defines the builder-reviewer loop for POPGP code, scientific claims,
validation artifacts, and documentation. It is designed for Codex, Claude, or another
agent system, but roles are never tied to a vendor or model name.

The objective is an auditable review of a frozen Git commit. Model agreement is review
evidence; it is not independent experimental confirmation of a physical claim.

## Roles and authority

| Role | Responsibility | Authority limit |
|---|---|---|
| Builder | Implements the candidate, tests it, and responds to findings | Cannot mark its own findings resolved |
| Independent reviewer | Audits the frozen candidate, requests tests, and verifies remediation | Does not edit implementation on a review branch or authorize merge |
| Principal investigator or maintainer | Confirms scope, external gates, and merge readiness | Cannot convert model agreement into empirical validation |

The same human may operate both agents, but that limitation must be disclosed. Prefer
a different model and a fresh task/session for the reviewer. Do not present a
same-session role-play as an independent review.

## Non-negotiable boundaries

- Every review binds to a full 40-character commit hash, never only a branch name.
- Freeze the candidate while it is under review. Any implementation change creates a
  new candidate and invalidates an unfinished review of the prior commit.
- Use separate builder and reviewer branches or worktrees. Reviewer branches contain
  review artifacts only.
- Do not give the reviewer final labels, secret seeds, private evaluator logic,
  credentials, or unrestricted private hardware profiles.
- Record the exact model identity, version, operator, access, shared context, commands,
  results, and context-hash method.
- Preserve initial reviews, responses, re-reviews, and disagreements. Correct an
  artifact with a new round; never rewrite the historical record.
- A disputed or deferred blocking finding remains blocking until independent evidence
  or an explicit maintainer override is recorded.

## Branch and artifact layout

| Work | Suggested branch | Artifact |
|---|---|---|
| Builder candidate | `codex/<task>` | implementation, tests, and documentation |
| Initial review | `review/<task>-1` | `reviews/independent_reviewer/<review-id>.md` |
| Builder remediation | `codex/<task>-remediation` | `reviews/codex/<review-id>-RESPONSE-<round>.md` |
| Re-review | `review/<task>-rereview-<round>` | `reviews/independent_reviewer/<review-id>-REREVIEW-<round>.md` |
| Disagreement | relevant role branch | `reviews/disagreements/<disagreement-id>.md` |

The first remediation branch starts from the commit containing the initial review
artifact. A later remediation round starts from the commit containing the prior
re-review. Each re-review branch starts from the exact remediation candidate it audits.

## Closed-loop procedure

### 1. Builder freezes a candidate

1. Start from the intended baseline and implement a bounded change.
2. Add unit tests and scientific regression or falsification tests as appropriate.
   Every new or changed acceptance gate must include a demonstrated failing negative
   control or mutation. Register that control in
   `docs/scientific_hardening/GATE_TEST_REGISTRY.md`; an identity check must be
   labelled as implementation consistency and cannot serve as a physics falsifier.
3. Run every command in `.github/workflows/ci.yml`, which is the authoritative quality
   suite. The README and
   [`LAUNCH_INDEPENDENT_REVIEW.md`](../reviews/LAUNCH_INDEPENDENT_REVIEW.md) reproduce
   the same command set for local execution.
4. Commit the candidate and record:
   - full candidate hash;
   - baseline hash;
   - `git rev-parse "<candidate>^{tree}"` output and command;
   - exact commands and results;
   - access restrictions and unresolved external gates.
5. Do not change that commit while review is active.

### 2. Independent reviewer audits the candidate

Create a fresh review branch/worktree at the exact candidate and use
[the independent-review template](../templates/INDEPENDENT_REVIEW_TEMPLATE.md).
The reviewer must:

- inspect the complete diff and relevant surrounding code, claims, tests, CI, and
  generated artifacts;
- reproduce important behavior and construct counterexamples or mutations where
  feasible;
- assign stable IDs to every finding and requested test;
- state evidence, consequence, required action, verification level, and blocking
  status;
- distinguish current implemented results from hypotheses and future work;
- commit only the review artifact; and
- return `approve` or `changes requested` with the blocking count.

The reviewer does not remediate implementation on the review branch.

### 3. Builder responds finding by finding

Start the remediation branch from the immutable review commit and use
[the review-response template](../templates/REVIEW_RESPONSE_TEMPLATE.md). Include one
response for every finding and every requested test, including non-blocking items.

Builder disposition and implementation status are separate:

| Field | Allowed values |
|---|---|
| `disposition` | `accepted`, `partially-accepted`, `disputed`, `deferred` |
| `implementation_status` | `implemented`, `not-implemented`, `external-action-required` |

Record rationale, changed files, full fix commits, exact verification results,
residual risk, and disagreement references. The builder may say a change is
`implemented`; only a later independent reviewer may say `verified-resolved`.

### 4. Builder freezes the remediation

After fixes and the response are committed, rerun the complete quality suite and CI.
Hand the reviewer the full response-containing remediation hash. The response cannot
contain the hash of its own commit; record that final hash in the handoff instead.
Make no further changes while re-review is active.

### 5. Independent reviewer re-reviews

Create a new re-review branch/worktree at the exact remediation hash. The reviewer:

1. assigns `verified-resolved`, `unresolved`, or `superseded` to every prior finding;
2. assigns `verified-satisfied`, `unresolved`, or `superseded` to every requested test;
3. independently verifies the builder evidence;
4. checks regressions and reports new findings with new stable IDs; and
5. issues a fresh recommendation.

`approve: true` requires zero unresolved blocking findings. If changes are requested,
repeat the response and re-review steps with new round-numbered artifacts.

### 6. Record disagreements and obtain merge authority

Use [the disagreement template](../templates/DISAGREEMENT_LOG_TEMPLATE.md) for a
disputed proposition. Freeze both positions and their predictions before running a
discriminating test where possible.

Before merge, the responsible maintainer verifies that artifacts refer to the intended
commits, no blocker remains open, required CI/external gates are green, and access
declarations are complete. Independent-review approval is necessary for this process;
it is not itself authorization to merge.

## Finding lifecycle

| State | Assigned by | Meaning |
|---|---|---|
| `reported` | Independent reviewer | A stable finding exists with evidence |
| builder disposition | Builder | Agreement or disagreement is recorded |
| `implemented` | Builder | A claimed fix awaits independent verification |
| `verified-resolved` | Independent reviewer | Required action is independently satisfied |
| `unresolved` | Independent reviewer | The finding remains open |
| `superseded` | Independent reviewer | A linked new finding replaces it |
| merge approval | Maintainer or PI | The repository change may advance |

## Minimum handoff record

Every handoff includes:

- task and review IDs;
- source and destination roles;
- exact 40-character commit and baseline hashes;
- artifact paths relevant to the round;
- context-hash command and result;
- commands already run and their exact outcomes;
- model/operator/access declarations; and
- unresolved blockers or external gates.

A branch name or “tests pass” statement alone is not a valid handoff. See
[Launching an independent review](../reviews/LAUNCH_INDEPENDENT_REVIEW.md) for
copy/paste commands and prompts.
