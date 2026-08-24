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

## RR8 canonical proof-export amendment

RR8 found that Windows containment and the trusted four-subject checks completed, but
the artifact action could not discover those files inside the security-labelled output
tree. This draft leaves containment unchanged. Only after the untrusted descendant tree
is quiescent and the four live subjects are regular, single-link, bounded, and mutually
hash-bound, the trusted runner creates a fresh direct child of `RUNNER_TEMP`, applies an
ordinary medium-integrity/inherited-user boundary on Windows (mode 0700 on Ubuntu), and
writes one deterministic sorted compact JSON envelope. The envelope has exactly four
case-sensitive members, exact cell/run/source identity, and base64/length/SHA-256 for
each byte string. It is UTF-8 with LF and no BOM.

The upload action receives only the exact runner-temp `envelope.json` path. The frozen
aggregator requires six one-file artifact directories, rejects reparse/symlink/hardlink,
duplicate or case-fold-colliding keys, malformed/noncanonical JSON/base64, size or
expansion excess, member/inner hash disagreement, and cross-cell identity. It decodes
members only in memory; no archive or extraction path exists. A hosted 2×3 replay and
fresh independent rereview remain required before RR4/RR6/RR8 can be considered closed.

## RR9 canonical envelope transport amendment

RR9 records that the RR8 envelope was valid inside each Windows trusted step but the
per-cell artifact action could not discover it, while Ubuntu rejected legitimate empty
stdout/stderr at PowerShell parameter binding. The byte hasher now explicitly accepts
an empty byte array and retains the standard empty SHA-256 identity.

The proof workflow now has six explicit jobs rather than matrix-output collision
semantics. After containment teardown, workspace cleanup, and canonical-envelope
validation, each trusted runner caps the envelope at 131072 decoded bytes and 174764
single-line base64 characters, checks its fresh regular single-link GitHub output
control beneath runner temp, and writes exactly one cell-specific output without
logging it. The contained environment still omits all GitHub and runner control-plane
variables and paths.

One Ubuntu job receives six statically distinct outputs, rejects missing, equal,
overwritten, masked, truncated, newline/control-injected, oversized, corrupt, or
cross-cell values, validates every canonical envelope and inner four-member proof in
memory, then creates exactly six envelope files plus one aggregate manifest in a fresh
directory. It uploads only that consolidated directory. A dependent Ubuntu job
downloads the retained artifact and revalidates the exact seven-file set, regular and
single-link metadata, sizes, identities, envelope hashes, inner hashes, and aggregate
bytes. Hosted success and fresh independent rereview remain required before the RR4,
RR6, RR8, or RR9 blockers may be considered closed.

## RR10 digest-bound cache transport amendment

Hosted run 32714269336 demonstrated that all six containment cells were green but the
three Windows large job outputs were empty at the `needs` boundary, while GitHub printed
the three Ubuntu base64 values in the aggregate step environment preamble. RR10 removes
envelope bytes from job outputs and environments entirely.

Each explicit cell now stages one canonical envelope beneath the fixed relative path
`.via000-r3-proof-cache/<platform>/<stage>`. The untrusted identity has already been
retired and cannot write that path. A distinct trusted outer step rechecks the ordinary,
single-link, non-reparse/ADS-free subject and exact cell/run/source identity, computes
its SHA-256, and writes only the lowercase 64-hex digest to GitHub output control. It
preflights an exact key and rejects any pre-existing entry, then invokes pinned
`actions/cache/save` v6.1.0 at commit
`55cc8345863c7cc4c66a329aec7e433d2d1c52a9`. Cache-save success is explicitly not
evidence because the action can turn collision/upload failures into warnings.

The key is shorter than 512 characters and binds the fixed namespace/schema,
repository ID, workflow SHA, source SHA, run ID, run attempt, platform, stage, and full
envelope digest. No prefix or restore key exists. The Ubuntu aggregate reconstructs six
keys from six distinct digest outputs, restores to the identical six relative paths
with cross-OS archives enabled, and requires `cache-hit=true` plus byte-identical
primary and matched keys. The frozen aggregator rejects extra/missing/case-colliding,
linked, oversized, corrupt, stale, cross-cell, or digest-mismatched restored state before
creating retained output. The consolidated seven-file artifact and its dependent
redownload verification remain authoritative; cache retention is transport only.

This closure assumes the already-reviewed trusted GitHub control plane and pinned action
principals and admits no additional hostile trusted job with a cache runtime token. If
that excluded principal is added to the threat model, standard cache save cannot prove
producer provenance and requires a separately reviewed reservation-identity protocol.
Fresh hosted completion and independent rereview remain required before RR4, RR6, RR7,
RR8, RR9, or RR10 may be considered closed.

Hosted run 32718921969 then proved the six containment/staging cells but exposed two
outer-preflight portability defects before any cache save: GitHub rejected the custom
Windows shell string whose executable path contained spaces, and the shared archive
banner check observed only one delayed process exit after two version pipelines. The
bounded correction uses the supported absolute system Windows PowerShell shell for the
trusted digest, archive, and cache-preflight snippets; keeps the digest writer in that
outer process; and limits those snippets to Windows PowerShell 5 APIs. GNU tar and
Zstandard output and exit codes are now captured immediately and validated independently
against nonempty version-bearing banners. No cache key, action, containment, transport,
stage topology, or threat-boundary semantics changed. A fresh hosted completion and
independent rereview are still required.

## RR12 Windows built-in PowerShell execution amendment

RR12 invalidates every prior hosted Windows conclusion whose material script used a
custom PowerShell shell template. Runs 32722238480, 32722416771, 32722567809,
32722822767, and 32723138041 are disposable, non-authoritative diagnostics only. The
last run proved that GitHub's built-in `shell: pwsh` executes at the expected PowerShell
7 process, preserves same-step files for a following built-in shell, and propagates a
digest output. It also proved that the remaining custom dot-source staging steps could
report green without creating their cache path. No cache or artifact from those runs is
campaign evidence, and no earlier Windows proof artifact produced through a custom
shell may be reused.

All twelve trusted Windows run steps in the proof workflow and all eight trusted
PowerShell step types in the production workflow now use GitHub's built-in
`shell: pwsh`. The job PATH is constrained to the reviewed PowerShell 7, Git, and
Windows system roots. Before any material action, every Windows script checks the
current process module and `$PSHOME` against
`C:\Program Files\PowerShell\7\pwsh.exe`, enforces the PowerShell 7 version policy,
requires that no profile file exists at any applicable profile path, and requires the
exact sanitized PATH. The source regression rejects custom `-File {0}`, dot-source,
spaced executable, and Windows PowerShell templates. Ubuntu stage logic, the six-job
topology, cache keys/actions, containment helper, and scientific lifecycle guards are
unchanged; production steps use the same supported built-in PowerShell boundary on
both hosted operating systems. Every production step applies the corresponding exact
Windows or Ubuntu process, `$PSHOME`, version, no-profile-file, and sanitized-PATH
assertion before material action.

Each material producer now proves its side effect before returning, and the next
trusted boundary rechecks it before consumption: authorization record and outputs;
tool manifest, executable hashes, and outputs; containment teardown and result;
stage summaries/manifests and mutation quiescence; attestation subject outputs and
subject hashes; one ordinary staged envelope and its digest immediately before cache
save; retained attestation bundle; and failure cleanup absence. Action success alone
is not evidence. A fresh exact-source 2×3 proof, six exact cache restores, retained
seven-file Ubuntu artifact, dependent redownload verification, and independent RR12
rereview remain mandatory. The production workflow must remain undispatched while the
campaign is drafted.

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

## RR14 Windows restricted-token compatibility amendment

The bounded RR13 factor matrix isolated `LUA_TOKEN` as the cause of hosted Windows
`STATUS_DLL_INIT_FAILED`; the manual scrubbed environment was not causal. RR14 therefore
removes only `LUA_TOKEN` from the production `CreateRestrictedToken` flags and retains
`DISABLE_MAX_PRIVILEGE`, explicit `S-1-16-4096` relabeling, suspended process creation,
atomic Job Object assignment before resume, kill-on-close, explicit descendant-tree
termination, protected/mutable separation, closure checks, and zero-active proof.

The production helper now queries the restricted token before process creation and
fails unless integrity is exactly `S-1-16-4096` and the sorted enabled-privilege list is
empty or exactly `SeChangeNotifyPrivilege`. Command results, stage summaries, canonical
proof envelopes, schemas, and the assembler bind and independently validate the exact
flag list, integrity SID, canonical privilege count/list, protected-label policy, and
teardown state. The local exact hostile replay requires payload execution, mutable-write
success, protected read/write/replace denial, unchanged protected hashes, and complete
descendant teardown. A new exact-source six-cell hosted proof and independent rereview
remain required; this draft amendment is not activation or scientific evidence.

## RR15 PowerShell parse-closure amendment

The exact RR14 proof-only run reached the frozen proof runner on all six hosted cells
but PowerShell rejected the nonexistent `-cjoin` operator before containment began.
The same invalid comparison was present in the production runner and both byte-equal
receipt copies. RR15 replaces all four comparisons with one explicit predicate that
requires equal counts and then compares each non-null string at the same index using
`[StringComparison]::Ordinal`. It preserves case and order, distinguishes an empty
array from one empty string, rejects non-string/null entries, and never serializes the
arrays through a delimiter.

The pre-seal test now records the exact eight R3 protocol/receipt PowerShell paths and
uses PowerShell 7 `Parser.ParseFile` to require zero errors in every file. A malformed
temporary `-cjoin` source must fail the same gate, while separate probes reject case,
order, count, type, null, delimiter-collision, and empty-array substitutions. RR14's
token, integrity, privilege, containment, and evidence requirements are unchanged.
A fresh six-cell hosted proof and retained redownload remain mandatory.

## RR18 native root/file descriptor amendment

RR17's proof-only experiment established that the exact medium-integrity boundary is
viable and that the hosted envelope's default owner can be BUILTIN Administrators.
RR18 does not import the experiment workflow or scripts. It replaces only the managed
root ACL application with native owner plus protected-DACL application and explicitly
sets only the closed envelope's owner to the current runner SID. The root remains one
explicit runner FullControl CI/OI ACE; the file remains one inherited runner
FullControl ACE. Exact native and managed control masks, owner, DACL presence/default
state, ACE type/mask/flags, medium label, stream/link identity, and hash must agree at
all three trusted boundaries. No owner, DACL, label, or hostile-operation predicate is
relaxed. The fresh retained 2x3 hosted proof remains the acceptance gate.

## RR19 Windows cache zstd identity amendment

The RR18 replay reached and passed all six containment, canonical-envelope, and digest
boundaries, then failed closed because the frozen Windows cache preflight named a zstd
path absent from the hosted image. RR19 changes only that archive-tool identity to the
hosted `C:\tools\zstd\zstd.exe` version 1.5.7 contract. Every Windows cell places only
`C:\tools\zstd` into the already-sanitized PATH, requires unique exact command
resolution, binds the executable SHA-256, and repeats its path, ancestor, stream,
hard-link, reparse, version, resolution, and hash predicates immediately before and
after cache save. Containment, export descriptors, envelope bytes, cache keys, and the
six-job topology are unchanged. A fresh retained 2x3 replay remains mandatory.

## RR20 canonical inner JSON amendment

RR19's hosted replay passed every producer, zstd, cache-save, exact-restore, hit, and
matched-key boundary. It then failed before output because Windows ConvertTo-Json plus
Set-Content produced CRLF inside decoded proof.json; the same latent pipeline wrote
containment-result.json. RR20 replaces all three call sites with one frozen helper that
serializes a compact object to strict UTF-8 without BOM, appends one LF byte, writes
through an exclusive create or explicit replacement stream, flushes to disk, and
requires identical read-back bytes and hash. Inner aggregation now requires precisely
that terminal-LF shape. Outer envelopes and all earlier security predicates are
unchanged; a fresh retained 2x3 replay remains mandatory.

## RR21 retained artifact digest canonicalization amendment

RR20's replay produced and retained the exact seven proof-only files, but the final
verifier rejected before byte validation because upload-artifact exported bare
lowercase SHA-256 while the verifier deliberately requires the Actions API's
`sha256:<hex>` representation. RR21 inserts one trusted built-in-pwsh step immediately
after upload. It rejects noncanonical ID, digest, repository, run, or URL data, accepts
only bare lowercase 64-hex input, prefixes it exactly once, and exports only the exact
normalized ID/digest/URL. The verifier remains prefixed-only and independently binds
the URL to the current repository, run, and artifact ID. All inner/outer bytes,
containment, cache, aggregation, and pinned download checks remain unchanged.

## RR22 Ubuntu PowerShell launch-identity amendment

RR21's exact-handoff replay passed all six producer/cache cells, aggregate validation,
and upload, but the post-upload normalizer failed before canonicalization. GitHub's
built-in Ubuntu `pwsh` selector launched `/usr/bin/pwsh`, contradicting the step's
unchanged exact `/opt/microsoft/powershell/7/pwsh` process-identity predicate. RR22
changes only that shell declaration to the already-proven literal `/opt/.../pwsh
-NoLogo -NoProfile -NonInteractive -File {0}` form. Exact PSHOME, PS7, absent-profile,
sanitized-PATH, raw ID/digest/URL, output-control, and verifier predicates remain
unchanged. Run `32777599858` remains synthetic and cannot authorize lifecycle action.

## RR16 hosted export-boundary amendment

RR15's hosted run proved that all six frozen runners parse and both containment
primitives execute, but Ubuntu's conditional empty-array expression collapsed the
expected flags to null, while Windows tested a nondeterministic inherited ACL shape.
RR16 constructs the Ubuntu expectation as an explicit non-null empty object array.

After Windows teardown, trusted PowerShell now applies a protected DACL containing
only one exact current-runner SID `FullControl` ACE, fixes the owner to that SID, and
sets an inheritable exact medium `S-1-16-8192` mandatory label with `NO_WRITE_UP`.
Native SID/mask/ACE queries and SID-based DACL inspection verify the fresh root and its
single inherited envelope before digest output and again before cache save. A live low-
integrity regression requires mutable writes to succeed while export create, write,
rename, delete, reparse, and replace attempts fail. A fresh six-cell hosted proof and
retained redownload remain mandatory.

## RR24 Ubuntu PowerShell PATH-normalization amendment

The literal RR22 launcher is retained. Diagnostic run `32783816053` established that
its exact live PATH is `/opt/microsoft/powershell/7:/usr/bin:/bin`, not the already
sanitized `/usr/bin:/bin`. RR24 first requires that exact trusted runtime prefix, then
immediately resets PATH to `/usr/bin:/bin` and reasserts it before raw artifact
identity access, the absolute `stat` call, `GITHUB_OUTPUT` append, and post-append
verification. Missing, doubled, reordered, alternate, or injected prefixes reject.
MainModule, PSHOME, PowerShell 7.6.5, and the exact six launch arguments remain fixed.
Exact `-NoProfile` argv is sufficient to prove profiles were not loaded; the prior
profile-file-nonexistence assertion is removed because profile files may legitimately
exist. A fresh six-cell replay remains mandatory.

The campaign validator's shared R2/R3 raw-evidence entry point now preserves the two
protocol generations explicitly. It accepts exactly the legacy R2 platform contract
or the complete R3 staged-plus-dispatch contract, rejects partial mixtures, retains
R2 `platform-summary.json` semantics, and applies staged tool-identity requirements
only to R3. No R2 protocol, packet, receipt, or evidence byte is reinterpreted.
