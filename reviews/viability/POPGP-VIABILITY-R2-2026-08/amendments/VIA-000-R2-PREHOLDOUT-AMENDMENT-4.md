# VIA-000 R2 pre-holdout amendment 4

This amendment addresses the two blocking scopes in independent re-review 2 of the
R2 protocol boundary. It is made while `VIA-000` remains preregistered,
`holdout_started: false`, unrevealed, and without raw results or an output commitment.
It changes no scientific candidate, hypothesis, decision threshold, mutation family,
hidden commitment, or R1 result.

## Authenticated producer provenance

An internally hash-closed package is no longer sufficient evidence of execution.
The exact-SHA GitHub Actions workflow now runs the clean platform protocol and the
frozen mutation-test runner, then uses the pinned official GitHub attestation action
to sign both `platform-summary.json` and `evidence-manifest.json`. The attestation is
bound to repository `whact2025/POPGP`, the frozen signer workflow, the exact protocol
source commit, and the SLSA provenance predicate. The retained Sigstore bundle is a
required input.

Before copying evidence or creating output, the assembler invokes GitHub CLI
attestation verification for both subjects with the frozen repository, signer,
source, predicate, and hosted-runner constraints. The public campaign validator
repeats this verification. A coherent replacement whose internal hashes are all
recomputed but which lacks the valid external attestation must fail before output.

## Execution-derived mutation receipts

Separately authored mutation files are removed from the assembly interface. The new
frozen mutation runner executes the exact registered pytest selectors under an
isolated, locked, non-editable environment. It retains one stdout stream, stderr
stream, and typed suite-result document, then derives every family receipt from the
observed passed node IDs and registered expected counts. Those bytes are included in
the signed manifest and summary. The semantic validator reconciles the command,
selectors, timestamps, hashes, node prefixes, counts, oracle IDs, and all eighteen
receipts.

This establishes authenticated attribution to the registered GitHub workflow and
mechanically binds the declared mutation outcomes to its retained public execution.
It does not claim protection against compromise or collusion of GitHub's OIDC/Sigstore
control plane or a repository administrator able to authorize the exact workflow.

## Portable platform evidence

The environment-manifest verifier now accepts the runner's canonical regular-file
records and canonical Linux symlink records, while rejecting mixed or extra shapes.
Its parser retains the default 100,000-node public-input ceiling everywhere else and
uses a 150,000-node ceiling only for the typed, attested environment manifest. The
limit admits the genuine hosted Ubuntu and Windows manifests measured at 133,309 and
127,917 expanded nodes while remaining bounded by the unchanged 16 MiB byte ceiling,
depth, cycle, numeric, and exact-entry-shape checks.
The potential-moment recomputation tolerance is calibrated to `2e-18`: it admits the
independently measured genuine Ubuntu-to-Windows recomputation difference of about
`1.24345e-18` while retaining rejection of the smallest registered attack at about
`4.46e-18`. Direct raster comparison remains bounded at four channel values.

## Required evidence before refreeze

The amendment is not approval to refreeze or start holdout. The exact response commit
must first produce successful ordinary CI and successful Ubuntu and Windows protocol
artifacts. The unmodified downloaded fragments must verify against their GitHub
attestations, assemble on a supported reviewer platform, pass authoritative semantic
and custody validation, and demonstrate fail-before-output behavior for missing or
forged attestations and the registered portability boundary controls. A fresh
independent re-review must report zero blockers. Only then may a maintainer refreeze
the protocol, after which a new independent falsifier must recommend SAFE before any
holdout transition.
