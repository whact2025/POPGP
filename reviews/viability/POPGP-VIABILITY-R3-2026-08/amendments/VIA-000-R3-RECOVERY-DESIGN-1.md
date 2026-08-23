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
`refs/tags/popgp-via000-r3-protocol-<protocol-snapshot-commit>`. The tag is
content-addressed by naming convention: its 40-hex suffix must equal its resolved
commit. Before any scientific command, the frozen dispatch guard requires:

1. event name `workflow_dispatch`;
2. exact content-addressed tag ref;
3. explicit snapshot input equal to the tag suffix;
4. resolved tag commit equal to that input;
5. `github.sha` equal to that input; and
6. checkout HEAD equal to that input.

The runner and mutation runner retain the same commit/ref plus the shared GitHub run
ID and attempt in each platform summary. GitHub OIDC/Sigstore signs the summary and
evidence manifest with the workflow source digest. The assembler independently
resolves the content-addressed tag, compares every supplied protocol/schema artifact
with the named Git blobs, verifies frozen artifact hashes and both attestations,
requires identical platform run identities, and invokes the public semantic validator
before atomically emitting raw results and a commitment. The public validator repeats
the packet-commit, source, attestation, dispatch-ref, and cross-platform run checks.

Thus the following identity is single-valued:

```text
tag suffix = resolved tag = github.sha = checkout HEAD
           = runner source = mutation source = Sigstore source digest
           = assembler source argument = raw-results source = packet protocol_commit
```

## Negative controls

The registered R3 identity gate rejects push events, branch refs, wrong tag suffixes,
wrong ref resolution, wrong `github.sha`, later lifecycle HEADs, protocol files that
differ from the snapshot Git blobs, wrong-source attestations, and Ubuntu/Windows
fragments from different workflow runs. Every rejection is required before an output
commitment can exist. The exact snapshot-ref path is also exercised as a positive
control.

## Custody and authorization boundary

The builder accessed no custody directory, sealed manifest, secret seed, hidden label,
reveal material, or R2 invalid output package. R3 carries forward only the public R2
URI/SHA-256 commitment pairs. A fresh R3 custodian must privately verify all fourteen
sealed raw-byte hashes and packet IDs before preregistration or holdout. This draft
contains no R3 custody verification, attack receipt, raw result, output commitment,
reveal, audit, adjudication, or campaign decision.

Independent review with zero blockers is required before a maintainer creates the
final snapshot/tag and activates the campaign. This document is design evidence only.
