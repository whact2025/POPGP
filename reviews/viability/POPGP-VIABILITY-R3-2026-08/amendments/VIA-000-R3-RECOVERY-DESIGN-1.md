# VIA-000 R3 protocol-identity recovery design 1

Status: drafted for independent review; not frozen, activated, preregistered, or run.

## Preserved predecessor record

R2 remains attacked, holdout-started, unrevealed, and pending. Its one hosted attempt
completed technically on Ubuntu and Windows, but the attested source was lifecycle
handoff `5dfee2355e06aeed45c7dae662e890a6dc6452e8` while every packet required frozen
protocol snapshot `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`. The frozen assembler trusted its
caller-supplied source identity and emitted an external package; the full campaign
validator correctly rejected it. No receipt was attached, no reveal occurred, and no
rerun was permitted. This protocol defect makes the attempt invalid rather than a
scientific failure or blockage.

R3 preserves candidate `5be3c38a0822d49953d0933f14ccab32ca12c896`, tree
`6ad387f9f4e0bab7f97df1bb54a03177887f0707`, comparison baseline
`9a29e05f803666bf0e3a28417ea399e3e26769fc`, every scientific measurement and
threshold, both platforms, the PDF and literal-source boundaries, all semantic/visual
checks, and all eighteen mutation families. It does not use the invalid R2 output.

## Recovery mechanism

Automatic push execution is removed. The only admissible event is a manual
`workflow_dispatch` from
`refs/tags/popgp-via000-r3-protocol-<protocol-snapshot-commit>`. The lightweight
source tag is content-addressed by naming convention: its 40-hex suffix must equal its
resolved commit. A separate SSH-signed annotated authorization tag is named by the
SHA-256 of its canonical JSON record and points to an immutable authorization commit.
The record hashes that commit's campaign, packet, and protocol-manifest blobs; both the
campaign and packet independently bind the exact source snapshot. Before any
scientific command, the frozen dispatch guard requires:

1. data-only step-environment transport for the canonical authorization ref, with
   exact grammar/length rejection before Git, Python, or input-directed file access;
2. event name `workflow_dispatch`;
3. exact content-addressed tag ref;
4. the resolved source tag equal to its suffix;
5. `github.sha` and checkout HEAD equal to the source snapshot;
6. one captured authorization tag object for record parsing, target peeling, and
   returned identity, with replacement objects disabled and a final check that its
   ref did not change;
7. a valid SSH signature over that same captured object from the public key frozen in
   the source snapshot; and
8. exact hashes and protocol bindings in the authorized campaign, packet, and
   manifest blobs.

The runner and mutation runner retain the same commit/ref plus the shared GitHub run
ID and attempt in each platform summary. GitHub OIDC/Sigstore signs the summary and
evidence manifest with the workflow source digest. The assembler independently
resolves the content-addressed source tag, re-runs the frozen authorization guard, and
derives the source commit only from the signed authorized bytes. It compares every
supplied protocol/schema artifact with named Git blobs, verifies frozen artifact
hashes and both attestations, and requires identical platform run identities. Before
any output rename or commitment it extracts the semantic validator and every local
dependency from the authorized source snapshot, checks them against both the primary
contract and authorized manifest, supplies the real authorized packet blob, and
executes the isolated bundle under Python `-I`. The public validator repeats the
packet-commit, source, attestation, dispatch-ref, authorization, and cross-platform
run checks. Guard, assembler, extracted validator, and workflow identity queries use
an absolute system Git program with `--no-replace-objects`, scrub all inherited
`GIT_*` repository/object/config/namespace/discovery controls, and disable system and
global Git config. SSH verification also pins the absolute system `ssh-keygen`
program, so local or environment-injected signature programs cannot run.
Workflow inputs and GitHub identity values are supplied only as step environment data,
validated before use, and passed through PowerShell argument arrays. The workflow
contains no GitHub expression in any `run:` source and performs no `Get-Command` or
dispatcher-`PATH` tool discovery. Its OS branches name the reviewed Windows and Ubuntu
Git/SSH paths, while the pinned setup action supplies exact Python 3.11.15.
The GitHub control plane and exact pinned action revisions are the declared bootstrap
principal. Fixed `ubuntu-24.04` and `windows-2025` labels, setup-Python output/root,
canonical Git/SSH/Python/PowerShell paths, uv 0.11.11 derived through trusted
`sysconfig`, and the TeX Live 2026 `pdftex` path are checked before experiment output.
Every executable must be a regular non-reparse file under its exact trusted root;
PATH/PATHEXT and child-process injection state are scrubbed, hashes are rechecked
across step boundaries, and a tool-identity manifest is retained in signed evidence.
Any failure removes the complete platform workspace and prevents artifact upload.

## RR5 execution-context separation amendment

Candidate execution, PDF production, and frozen mutation verification are separate
matrix entries and therefore separate fresh GitHub-hosted VMs on both Windows and
Ubuntu. They share only immutable Git identities and signed evidence bytes; they do
not share writable tools, caches, temporary directories, configuration, environment,
or processes. The PDF stage runs directly from the exact candidate Git checkout before
any candidate Python environment or package installation exists in that VM. It clears
TeX, TEXMF, kpathsea, font, and native-loader selectors, disables shell escape, and
hashes every regular byte under the pinned TeX Live 2026 root before and after both
passes. The mutation stage creates its own clone and locked environment only after the
candidate job has ended.

Each of the six stage fragments signs `stage-summary.json` plus
`evidence-manifest.json` and binds the same repository, workflow, protocol source,
authorization, GitHub run/attempt, candidate, platform, and explicit stage ID. The
assembler accepts no cross-stage executable/configuration state, link/reparse point,
or undeclared file. It requires all three stage attestations per platform and derives
the final platform record only from the candidate results, PDF proof, and mutation
proof assigned to those stages. Failure cleans the temporary assembly and cannot emit
an output commitment.

## RR6 descendant and same-path execution-boundary amendment

The RR4 same-path replacement gap and RR6 descendant-attestation race are closed in
this draft by moving every candidate- or mutation-controlled command behind an
OS-enforced privilege boundary. On Windows the trusted runner creates a restricted
low-integrity token, creates the child suspended, assigns it to a kill-on-close Job
Object before its first instruction, and terminates and queries that complete job
tree. On Ubuntu it creates a randomly named unprivileged system account per command,
runs it in a systemd transient service with `KillMode=control-group`, proves both an
empty control group and empty UID process set, creates trusted evidence while the UID
remains allocated, rechecks the UID process set, and only then deletes the account.
Keeping the account allocated through evidence creation prevents UID reuse from
invalidating the quiescence proof. The workflow fails closed when any account or
containment operation is unavailable.

The hosted Ubuntu service manager rejected the mount-namespace properties during
RR7 proof-path bootstrap. The frozen hosted path therefore declares
`PrivateTmp=no`, `ProtectSystem=no`, and `ProtectHome=no` and does not claim mount
namespace isolation. Its security boundary is instead the fresh per-command account,
runner-owned mode-0700 protected roots, a dedicated world-writable mutable root,
`NoNewPrivileges`/SUID restrictions, unchanged closure hashes, complete control-group
teardown, empty UID process proof, and account removal. Independent review must treat
those permission, identity, and cgroup premises—not an unavailable namespace—as the
Ubuntu claim.

The untrusted identity can write only a fresh mutable staging root. Tool,
configuration, and trusted-evidence roots are non-writable to it; on Windows the
trusted evidence SACL also applies no-read-up/no-write-up so an untrusted process
cannot target evidence bytes by content. `TEMP`, `TMP`, `RUNNER_TEMP`, home/cache,
loader, Git, Python, uv, and TeX selectors exposed to the child all resolve inside
mutable staging or are scrubbed. Trusted code copies result bytes, creates manifests,
and captures attestation subjects only after termination and a zero-descendant proof.
The assembler recomputes every retained containment-result hash and requires the
platform-specific primitive, privilege separation, teardown, protected-tool and
protected-evidence flags, and zero active descendants for every contained command.

The production workflow runs a hostile self-test on both hosted OS families. Its real
child detaches a delayed grandchild and attempts direct and replace-then-restore writes
to live tool/evidence paths; the gate requires the process tree to be killed and both
subjects to remain byte-identical. The exact Windows mechanism is also executed by a
local regression. Ubuntu enforcement remains a hosted-runner gate and requires fresh
independent rereview before activation. These changes are a drafted remediation, not
an approval, refreeze, or campaign run.

## RR7 hosted containment proof-path amendment

RR7 identified that the production self-test could not be exercised safely before
campaign authorization because it lived only behind the scientific workflow's manual
lifecycle and signer guards. This draft adds a separate, non-scientific branch-push
workflow for the narrow feature/review prefixes. It checks out only the frozen proof
closure, uses the exact production containment helper with a synthetic hostile
fixture, and runs all candidate/PDF/mutation labels on both supported hosted OSes.
It cannot access the scientific candidate or baseline, custody, signer/lifecycle
state, assemble output, create a commitment, or reveal anything.

Every cell binds the repository, workflow ref, branch ref, source SHA, run/attempt,
platform, and stage; verifies helper/runner/fixture/schema/aggregator/workflow source
and receipt hashes before creating its synthetic workspace; establishes a live
detached child-of-child; and requires protected evidence/tool denial, unchanged
closure bytes, total teardown, and zero active descendants. The aggregate accepts
exactly six complete cell artifacts, recomputes containment/transcript hashes, and
rejects identity mixing or any false predicate. The workflow and all verifier bytes
are protocol artifacts and receipts. A successful hosted run is evidence for the
next independent rereview only, never approval or activation.

Thus the following identity is single-valued:

```text
authorized packet commit = source-tag suffix = resolved source tag
           = github.sha = checkout HEAD
           = runner source = mutation source = Sigstore source digest
           = assembler-derived source = raw-results source
```

## Negative controls

The registered R3 identity gate rejects push events, branch refs, wrong tag suffixes,
wrong ref resolution, wrong `github.sha`, later lifecycle HEADs, a later
self-consistent exact-tagged commit, unsigned/moved/deleted/substituted authorization,
an authorization-ref swap between parsing, peeling, and signature verification,
default or custom-namespace Git object replacement, caller-controlled Git object
directories/alternates/config/programs, protocol files or validator dependencies that
differ from no-replacement snapshot Git blobs, PowerShell quote/statement/
subexpression/newline/control payloads, option-like input, `PATH`/`PATHEXT`-shadowed
tools, substituted setup-action output, wrong roots/versions/banners/hashes, command
shims, or symlink/junction/reparse tools,
wrong-source attestations, and Ubuntu/Windows fragments from different workflow runs.
Every rejection is required before an output commitment can exist. Disposable test
repositories exercise the exact signed authorization path, invalid-tag replacement,
source/campaign/packet/manifest/validator replacement, environment/config injection,
the actual workflow shell boundary with hostile inputs and zero/one/multiple `PATH`
shims, and Git-normalized LF blobs under both `core.autocrlf=true` and `false`.

## Custody and authorization boundary

The builder accessed no custody directory, sealed manifest, secret seed, hidden label,
reveal material, or R2 invalid output package. R3 carries forward only the public R2
URI/SHA-256 commitment pairs. A fresh R3 custodian must privately verify all fourteen
sealed raw-byte hashes and packet IDs before preregistration or holdout. This draft
contains no R3 custody verification, attack receipt, raw result, output commitment,
reveal, audit, adjudication, or campaign decision.

The checked-in signer file is intentionally comment-only; no production authorization
key or tag has been created. Activation is blocked until a separately reviewed
amendment freezes exactly one Ed25519 public key. Independent review with zero blockers
is then required before a maintainer creates final snapshot/authorization tags and
activates the campaign. This document is design evidence only.
