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

1. event name `workflow_dispatch`;
2. exact content-addressed tag ref;
3. the resolved source tag equal to its suffix;
4. `github.sha` and checkout HEAD equal to the source snapshot;
5. a canonical content-addressed authorization record with an immutable target;
6. a valid SSH signature from the public key frozen in the source snapshot; and
7. exact hashes and protocol bindings in the authorized campaign, packet, and
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
run checks.

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
protocol files or validator dependencies that differ from snapshot Git blobs,
wrong-source attestations, and Ubuntu/Windows fragments from different workflow runs.
Every rejection is required before an output commitment can exist. Disposable test
repositories exercise the exact signed authorization path and Git-normalized LF blobs
under both `core.autocrlf=true` and `false`.

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
