# POPGP mechanism-viability campaign R3

This directory is the drafted third attempt to establish Tier R under
[`VIABILITY_DEMONSTRATION_PLAN.md`](../../../docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md).
R3 preserves the exact R2 scientific candidate
`5be3c38a0822d49953d0933f14ccab32ca12c896`, candidate tree
`6ad387f9f4e0bab7f97df1bb54a03177887f0707`, and comparison baseline
`9a29e05f803666bf0e3a28417ea399e3e26769fc`.

R1 remains a valid failed E3 result. R2 remains an immutable attacked,
holdout-started, unrevealed, pending round whose sole hosted attempt was invalid because
its attestations bound lifecycle handoff `5dfee2355e06aeed45c7dae662e890a6dc6452e8`
instead of protocol snapshot `9a0e28de5605a6d84965cbd594fa1ab0baf9a7b9`.
R3 does not repair, rerun, reveal, or reinterpret either predecessor.

## Draft state

- Campaign decision: pending.
- All seven Tier-R packets: drafted, unrevealed, `holdout_started: false`, and not run.
- No R3 attack, raw result, output commitment, reveal, audit, or adjudication exists.
- The protocol requires independent adversarial review before any final protocol
  snapshot, preregistration, falsifier run, custody transition, or holdout execution.

## Identity-safe hosted execution

R3 removes automatic push execution from the scientific campaign workflow.
`.github/workflows/via000-r3-protocol.yml` accepts only a manual dispatch from
the lightweight content-addressed tag
`refs/tags/popgp-via000-r3-protocol-<protocol-snapshot-commit>`. The caller supplies a
separate SSH-signed annotated authorization tag whose name embeds the SHA-256 of its
canonical record. That record points to an immutable authorization commit whose
campaign, packet, and manifest blobs independently authorize the same snapshot.
The workflow and assembler each capture the authorization ref once as a full tag
object ID; record parsing, target peeling, signature verification, returned identity,
and the final ref-stability check must all use that exact immutable object. Every Git
read in that chain disables replacement objects, uses an absolute system Git program,
and runs with caller-supplied Git repository, object-store, replacement, namespace,
config, SSH, and discovery variables removed. Signature verification additionally
pins the absolute system `ssh-keygen` program and an isolated allowed-signers file.
The manual authorization input crosses the GitHub/PowerShell boundary only through a
step environment value. The first shell decision validates its exact full-ref grammar
and length; later Git, Python, runner, and mutation values are data elements in
argument arrays. No GitHub expression is rendered into PowerShell source. The
workflow uses reviewed literal Git and `ssh-keygen` paths for Windows and Ubuntu plus
the exact Python 3.11.15 path produced by the pinned setup action. Fixed hosted-runner
labels, trusted roots, regular/non-reparse ancestry, exact Python/uv/TeX banners, and
SHA-256 identities bind Git, SSH, base/environment Python, uv, pdfTeX, and PowerShell.
uv is found only through trusted Python `sysconfig`; TeX is found only beneath the
pinned action root. PATH/PATHEXT and child injection state are scrubbed, every child
gets an explicit executable path, signed evidence retains the tool-identity manifest,
and a rejected execution leaves no platform workspace or uploaded artifact.
The source-tag suffix and resolution, authorized packet `protocol_commit`,
`github.sha`, checkout HEAD, runner and mutation `protocol_source_commit`, Sigstore
source digest, raw-results identity, assembler-derived identity, and validator
expectation must all be the same lowercase 40-hex commit. Linux and Windows fragments
must also carry one shared GitHub Actions run ID and attempt. A branch/lifecycle HEAD,
later self-consistent tag, static or mid-verification authorization replacement, Git
replacement object or injected Git environment/config, mutable validator source,
workflow-expression, setup-output/executable identity injection, wrong-source attestation, or cross-run
mixture fails before output commitment.

Each platform now consists of three independently hosted jobs: `candidate`, `pdf`,
and `mutation`. GitHub provisions a fresh VM for every matrix entry, so candidate
Python, caches, temporary files, configuration, and surviving processes cannot be
observed by the later PDF or mutation stages. The PDF stage clones the exact candidate
without creating a candidate Python environment, sanitizes TeX/kpathsea/font/native
loader state, disables shell escape, and retains byte-identical before/after manifests
of the complete pinned TeX Live tree. The mutation stage independently clones and
builds its environment after the candidate job has ended. Each stage signs its own
summary and evidence manifest while binding the same repository, workflow, source,
run, attempt, platform, and authorization identity. Uploaded artifacts contain only
declared evidence bytes; executables, configuration state, links/reparse points, and
undeclared files are rejected. The assembler requires all six stage attestations and
merges only their stage-authorized scientific contributions.

Inside every candidate- or mutation-controlled stage, the runner establishes a
second boundary before the first adversarial instruction. Windows uses a restricted
low-integrity token atomically assigned to a kill-on-close Job Object; Ubuntu creates
a fresh unprivileged system account for a systemd transient service, proves empty
control group and UID process set, creates evidence while retaining the UID allocation,
rechecks the UID process set, and only then deletes the account. Untrusted code
can write only mutable staging paths and cannot write the protected tool/configuration
or trusted-evidence roots. The trusted runner creates evidence and captures/rechecks
attestation subjects only after terminating the complete descendant tree and proving
zero active descendants. A production-path hostile self-test on each hosted platform
spawns a delayed detached writer and attempts live evidence/tool replace-and-restore;
any surviving write, unavailable containment primitive, or incomplete quiescence
aborts without upload or commitment. The Windows primitive is exercised locally; the
Ubuntu primitive is enforced by the hosted workflow and remains pending independent
rereview. This is still a draft, not an activated or frozen protocol.

The hosted Ubuntu image rejected the mount-namespace properties during the safe RR7
proof bootstrap. Consequently the reviewed helper explicitly uses `PrivateTmp=no`,
`ProtectSystem=no`, and `ProtectHome=no`; it does not claim mount namespace isolation.
The Ubuntu boundary depends on that per-command unprivileged identity, runner-owned
mode-0700 protected roots, a dedicated mutable root, no-new-privileges/SUID controls,
retained closure hashes, empty-cgroup and empty-UID-process proof, and account removal.
This premise is deliberately visible for independent reviewer scrutiny.

A separate `.github/workflows/via000-r3-containment-proof.yml` is a non-scientific
review gate. A tightly scoped push to `campaign/via000-r3-protocol-*` or
`review/via000-r3-protocol-*` runs the exact frozen production containment helper on
the synthetic hostile fixture in the full 2-platform × 3-stage matrix. It has only
read-only repository permission, no secrets or environments, no authorization,
lifecycle, signer, custody, candidate/baseline execution, assembly, commitment, or
reveal path. Each cell proves child-of-child teardown, protected evidence/tool
denial, unchanged closure hashes, a scrubbed control-plane environment, and zero
descendants. A frozen aggregator rejects a missing, substituted, cross-source, or
cross-run cell and emits only a small non-scientific aggregate. RR8 adds a post-teardown
transport boundary: each cell serializes the exact four validated proof subjects into
one canonical UTF-8/LF/no-BOM JSON envelope in a fresh ordinary directory directly
under trusted runner temp. Each exact case-sensitive member carries its byte length,
SHA-256, and base64 bytes. RR10 supersedes the failed RR9 large job-output channel.
Each of six explicit cells stages exactly one envelope at the same frozen workspace-
relative path used by a pinned `actions/cache` v6.1.0 save/restore pair. A separate
trusted outer step transports only the envelope's lowercase SHA-256. The full cache key
binds repository, workflow/source SHA, run/attempt, platform, stage, schema namespace,
and digest; a pre-existing exact key is rejected and no restore prefix is allowed.
Ubuntu requires six exact cache hits and primary/matched-key equality, then validates
all six envelopes in memory, writes the six exact envelopes plus one aggregate manifest,
uploads that seven-file tree, and a dependent job redownloads and revalidates it. Cache
save success is not evidence: only exact restore, byte validation, retained upload, and
redownload verification establish the synthetic proof. Missing, duplicate, nonhex,
stale, fallback, corrupt, linked, cross-cell, or cross-run bytes fail closed.
Hosted output is
review evidence only; it cannot activate R3 or establish scientific viability.

The checked-in allowed-signers file is deliberately comment-only. Activation is
blocked until a separately reviewed amendment freezes exactly one Ed25519 public key,
after which an authorized maintainer may create the binding commit and signed tag.

## Custody carry-forward boundary

No hidden byte was created, read, copied, or changed during R3 drafting. The packets
carry forward only the fourteen public R2 immutable URI/SHA-256 commitment pairs.
Those commitments are not yet accepted as R3 custody evidence. Before R3 may advance
to preregistration or holdout, a fresh R3 evaluator/custodian session must privately
verify all fourteen sealed raw-byte hashes and packet IDs, without revealing content,
and record a new public carry-forward verification artifact. Any missing or mismatched
sealed byte requires new custody material and a new draft; it cannot be waived.

## Validation

From the exact draft-handoff checkout, run:

```powershell
uv run --frozen python -m scripts.check_viability_campaign `
  reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml
```

A green result establishes only a coherent drafted protocol. It is not independent
approval, reproduction evidence, a scientific outcome, or Tier-R viability.
