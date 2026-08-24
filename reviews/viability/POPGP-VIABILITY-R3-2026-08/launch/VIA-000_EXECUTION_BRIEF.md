# VIA-000 R3 draft execution brief

Status: design handoff for independent review only. Do not dispatch or start holdout.

## Fixed scientific identities

- Candidate: `5be3c38a0822d49953d0933f14ccab32ca12c896`
- Candidate tree: `6ad387f9f4e0bab7f97df1bb54a03177887f0707`
- Comparison baseline: `9a29e05f803666bf0e3a28417ea399e3e26769fc`
- R2 invalid attempt record:
  `../attempts/VIA-000-R2-INVALID-ATTEMPT-1.md` in the immutable R2 campaign
- R3 protocol snapshot: pending independent review and final freeze
- R3 content-addressed tag: pending; it must be
  `popgp-via000-r3-protocol-<final-protocol-snapshot-commit>`
- R3 signed authorization tag: pending; it must be an SSH-signed annotated tag named
  `popgp-via000-r3-authorization-<sha256-of-canonical-record>`, point to an immutable
  authorization commit, and bind campaign/packet/manifest bytes that authorize the
  exact protocol snapshot.
- Authorization signer: intentionally absent. Activation is blocked until a separate
  reviewed amendment freezes exactly one Ed25519 public key.

## Draft surfaces

- Campaign: `reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml`
- Packet: `reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml`
- Primary protocol: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json`
- Dispatch guard: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py`
- Runner: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1`
- Mutation runner: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py`
- Assembler: `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py`

RR18 preflight: the proof-only Windows cells must use the manifest-bound native
descriptor helper. The post-teardown workspace-relative root requires exact runner
ownership, protected non-defaulted one-ACE DACL, control mask 37892, and medium
NO_WRITE_UP. The closed envelope requires explicit runner ownership while retaining
its sole inherited ACE, control mask 33796, and inherited medium label. Any divergence
must clean the cell and leave no digest/cache/artifact.
- Raw-results schema:
  `protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json`
- Hosted workflow: `.github/workflows/via000-r3-protocol.yml`

## Required order after independent approval

1. Independent review accepts the remediation and a separate signer-key amendment
   freezes exactly one Ed25519 authorization public key.
2. A maintainer creates the final protocol snapshot commit and a lightweight tag whose
   exact name embeds that same commit.
3. A distinct authorization commit freezes campaign, packet, and protocol-manifest
   bytes that all bind that snapshot. The designated signer creates the canonical,
   content-addressed, signed annotated authorization tag pointing at that commit.
4. The campaign is activated only after both tag identities and the authoritative
   campaign validator pass.
5. A fresh custodian privately verifies all fourteen carried-forward sealed R2
   commitments and records only hash/byte-count/packet-ID outcomes.
6. A fresh falsifier executes the clean control and all eighteen mutation families.
7. Only after custody and falsifier gates pass may a maintainer record `attacked` and
   set `holdout_started: true`.
8. A separate reproduction runner manually dispatches the workflow from the exact
   content-addressed protocol tag and supplies the exact signed authorization ref.
   GitHub supplies that input to PowerShell only as step environment data. The shell
   validates the complete canonical ref before any use and passes all values through
   argument arrays; no workflow expression may occur in `run:` source.
   The workflow captures that ref as one tag object ID and uses only that object for
   parsing, peeling, signature verification, and retained identity. All such Git
   operations use `--no-replace-objects`, a scrubbed Git environment/config, and the
   absolute system Git/SSH verifier programs; a changed ref or substituted object
   aborts before platform execution.
   Windows uses `C:\Program Files\Git\cmd\git.exe` and
   `C:\Windows\System32\OpenSSH\ssh-keygen.exe`; Ubuntu uses `/usr/bin/git` and
   `/usr/bin/ssh-keygen`. Base Python is the pinned setup-action 3.11.15 output; uv
   0.11.11 is derived from that interpreter's `sysconfig`; TeX is the canonical
   `pdftex` beneath the pinned TeX Live 2026 root; and PowerShell is the fixed system
   installation. The workflow accepts only fixed hosted-runner labels and regular
   non-reparse executables at exact trusted roots, records/rechecks SHA-256 identities,
   clears PATH/PATHEXT and child injection state, and passes explicit paths to every
   child. Any missing/substituted/shadowed tool aborts before experiment workspace or
   output; failure removes the entire platform workspace and no artifact is uploaded.
   All trusted PowerShell run steps use GitHub's built-in `shell: pwsh`; custom
   absolute, `-File {0}`, and dot-source shell templates are forbidden. Each Windows
   script first requires the canonical `C:\Program Files\PowerShell\7\pwsh.exe`
   process and `$PSHOME`, the PowerShell 7 version policy, no applicable profile file,
   and the exact reviewed PATH. Ubuntu production steps require the equivalent exact
   `/opt/microsoft/powershell/7/pwsh`, PowerShell 7, no-profile-file, and
   `/opt/microsoft/powershell/7:/usr/bin:/bin` boundary. Every material side effect is checked before the step
   returns and rechecked at the next trusted consumption boundary. Earlier Windows
   hosted proof metadata produced through custom shells is invalid and must not be
   reused.
9. Six matrix jobs run: candidate, PDF, and mutation on each supported platform.
   Every entry is a fresh GitHub-hosted VM and all six must share `github.run_id` and
   `github.run_attempt`. No stage consumes another stage's environment, cache, temp
   directory, tool tree, configuration, or process state. Candidate stages upload
   generated evidence only. PDF stages create no candidate Python environment, scrub
   all TeX/kpathsea/font/native-loader variables, use `-no-shell-escape`, and retain
   identical before/after SHA-256 manifests of the complete TeX Live tree. Mutation
   stages independently clone and build the frozen test environment. Each stage signs
   its own summary and manifest; all uploaded bytes must be declared evidence, never
   executable/configuration state.
   Within each job, every candidate- or mutation-controlled command must run through
   the frozen containment protocol. Windows requires a restricted low-integrity token
   assigned to a kill-on-close Job Object before resume; Ubuntu requires a systemd
   fresh per-command unprivileged system account in a transient service with
   control-group kill, empty-cgroup/UID-process proof, and account removal.
   The Windows token uses only `DISABLE_MAX_PRIVILEGE` (never `LUA_TOKEN`), is queried
   before child creation for exact `S-1-16-4096`, and may retain no enabled privilege
   except optional `SeChangeNotifyPrivilege`. Retain and validate the exact flag, SID,
   canonical privilege list/count, protected-label policy, and teardown evidence.
   Only mutable staging is writable to the untrusted identity. Tool/configuration and
   trusted-evidence roots remain protected, and trusted evidence/attestation subjects
   are created only after whole-tree teardown and zero-descendant verification. The
   production hostile self-test must pass before candidate execution.
   Before activation, independently inspect or replay the separate non-scientific
   `.github/workflows/via000-r3-containment-proof.yml` branch-push gate. It must show
    one source/run/attempt and all six explicit platform/stage jobs using the exact
    frozen production helper. Each job must stage one canonical envelope at its fixed
    workspace-relative cell path only after teardown and expose only its lowercase
    SHA-256. Require the pinned cache action, an exact digest-bound key, a cache-miss
    preflight, no restore prefix, and cross-OS archive identity. The Ubuntu aggregate
    must require six exact cache hits with primary/matched-key equality, validate all
    six restored envelopes in memory, retain exactly six envelopes plus one aggregate
    manifest, upload that single tree, and then download and revalidate all seven files.
    Cache-save success alone is never evidence. This proof workflow has read-only
    permissions and cannot run or alter the campaign; its result is review evidence,
    not authorization.
    Require all twelve Windows proof run steps to resolve through built-in `pwsh` and
    reject any custom PowerShell shell string before accepting the six-cell result.
    Before any hosted replay, parse the exact eight frozen R3 protocol/receipt `.ps1`
    files with PowerShell 7 `Parser.ParseFile` and require zero errors; the exact token
    flag arrays must compare by count plus ordinal element equality, never by joining.
    Ubuntu's empty expected flag list must be a non-null object array. On Windows the
    post-teardown cache/export root and envelope must have the exact current-runner SID
    owner/DACL and medium `S-1-16-8192` `NO_WRITE_UP` label, rechecked before digest and
    immediately before cache save; inherited hosted-workspace ACL shape is not evidence.
    Require each Windows proof cell to resolve zstd uniquely as the ordinary,
    non-reparse `C:\tools\zstd\zstd.exe` version 1.5.7 under the sanitized PATH,
    bind its SHA-256, and repeat exact identity/hash checks immediately before and
    after cache save. Do not accept the failed RR18 run `32763190366` as evidence.
    Require proof.json and both containment-result.json write paths to use the one
    frozen canonical byte writer: compact strict UTF-8, no BOM/CR, exactly one final
    LF, exclusive create/replace, write-through flush, and byte/hash read-back. Do not
    accept artifact-free run `32767703776` or normalize inner bytes after hashing.
    After the one retained proof artifact uploads, accept only the upload action's bare
    lowercase 64-hex digest and exact positive ID/current repository-run-ID URL in a
    trusted built-in-pwsh step, prefix the digest exactly once as `sha256:<hex>`, and
    give only those canonical values to the prefixed-only redownload verifier. Do not
    accept run `32771982270` as lifecycle evidence because its verifier remained red.
    Launch that Ubuntu normalizer only through the exact literal
    `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive -File {0}`;
    do not accept built-in `pwsh`, `/usr/bin/pwsh`, alternate targets, or missing flags.
    Do not accept run `32777599858` as lifecycle evidence because its built-in-shell
    launch contradicted the exact process-identity gate and skipped the verifier.
    Require its initial live PATH to be exactly
    `/opt/microsoft/powershell/7:/usr/bin:/bin`, reset it immediately to
    `/usr/bin:/bin`, and reassert the sanitized value before artifact identity/stat or
    output access. Reject missing, doubled, reordered, alternate, or injected prefixes.
    The exact six-argument `-NoProfile` launch is the profile no-load invariant; do not
    require profile files to be absent. Run `32782295879` is failed, superseded, and
    cannot support lifecycle action.
10. The assembler receives a platform root containing `candidate/`, `pdf/`, and
    `mutation/` evidence roots for each platform. It requires and verifies all six
    stage attestations, rejects missing/cross-run/cross-platform/substituted stages and
    undeclared or executable/configuration artifacts, then combines only the candidate
    scientific results, PDF proof, and mutation proof into each final platform record.
    Invoke it with the same protocol and authorization
    refs, run ID, and attempt. The assembler derives the commit only from authorized
    immutable bytes, independently pins the authorization tag object, and runs a
    manifest-verified validator bundle extracted through no-replacement Git reads
    from the snapshot. Any mismatch must leave no output directory or commitment.
11. Only the custodian may authorize reveal after a valid immutable commitment.
12. Fresh statistical, claim, independent-review, and adjudication seats complete the
    frozen governance sequence.

Never dispatch the scientific workflow from a campaign branch, activation/handoff
commit, or mutable lifecycle HEAD. The only branch-push exception is the separate
synthetic containment proof workflow above; never substitute its non-scientific proof
for a campaign result. Never reuse the R2 invalid output package.
