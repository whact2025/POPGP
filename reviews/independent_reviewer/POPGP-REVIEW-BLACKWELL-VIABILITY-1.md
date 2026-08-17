# Independent review: POPGP-REVIEW-BLACKWELL-VIABILITY-1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-BLACKWELL-VIABILITY-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "claude-opus-5"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "d600ea7c-809e-41df-bba4-4ecbdb4d7734"
reviewer_orchestrator_id: "claude-code-standalone"
review_date: "2026-08-17"
commit_reviewed: "dfe222ce3a173f617471d1979f0021ae6ec23ebf"
baseline_commit: "f22694b7427e82533c844f897b9f098558226ea1"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "4abafab7c68f802d304007ca73f9b27f202800bb"
context_hash_method: "git rev-parse \"dfe222ce3a173f617471d1979f0021ae6ec23ebf^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - ".gitattributes"
  - ".gitignore"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "popgp/backend.py"
  - "popgp/engine.py"
  - "popgp_engine/CMakeLists.txt"
  - "popgp_engine/build.bat"
  - "popgp_engine/build.sh"
  - "popgp_engine/kernel/CMakeLists.txt"
  - "popgp_engine/kernel/README.md"
  - "popgp_engine/kernel/include/types.cuh"
  - "popgp_engine/kernel/src/area_law.cu"
  - "popgp_engine/kernel/src/clock.cu"
  - "popgp_engine/kernel/src/main.cpp"
  - "popgp_engine/kernel/src/phase_flow.cu"
  - "popgp_engine/kernel/tests/test_area_law.cu"
  - "popgp_engine/kernel/tests/test_main.cpp"
  - "popgp_engine/kernel/tests/test_phase_flow.cu"
  - "popgp_engine/vcpkg.json"
  - "protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-000-2026-08-13.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-2026-08-13.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-010.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-100.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-150.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-200.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-300.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-400.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/protocol.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_engine.py"
  - "git diff f22694b7427e82533c844f897b9f098558226ea1..dfe222ce3a173f617471d1979f0021ae6ec23ebf (complete, all 26 files)"
  - "git show 3428a24 8c3c4d5 99f13dc dfe222c (per-commit diffs)"
  - "git archive dfe222c (disposable byte copies used for all mutation execution)"
  - "git clone --no-checkout C:/src/POPGP (disposable clone for campaign validator mutation tests)"
access_level: local-repository-public-tree-plus-public-unauthenticated-github-read-only
independence_statement: |-
  This is process separation, not external scientific validation. The weaknesses first.

  1. MODEL. The reviewing seat is claude-opus-5 per this runtime's own identity report; the
     version/snapshot string is not exposed to me and is recorded as `unknown` rather than
     guessed. The builder model identity is NOT knowable from the repository: every seat in
     every campaign packet records `model_identity: unknown`, and the candidate commits are
     authored by a human Git identity. I therefore set
     `reviewer_model_differs_from_builder: false` — not because I believe the models are the
     same, but because I cannot establish separation and must not assert an unverified
     independence fact. `builder_session_id` is recorded `unknown` for the same reason;
     `builder_orchestrator_id` is recorded as the value the packets declare (`codex-desktop`),
     which I read but could not verify.
  2. OPERATOR. Shared. The same human (`fuocor`) operates the builder and this review, and has
     operated every seat in this campaign. There is no operator separation anywhere.
  3. SESSION. Fresh. This session began with the frozen target, the access restrictions and the
     reading list. It did not receive builder conclusions before forming its own assessment.
  4. ORDERING DISCIPLINE, and its limit. As instructed, I inspected the implementation and
     recorded 13 preliminary hypotheses (H1-H13) with expected failure modes in private working
     notes BEFORE opening either calibration memo. Those notes are outside the repository. Two
     of my pre-memo hypotheses were subsequently REFUTED by my own execution and are reported
     as refuted below rather than filed as findings; several others were confirmed. The memos,
     when read, were treated as builder assertions and every load-bearing claim was
     independently re-derived or explicitly classified as not confirmed.
  5. EXPOSURE BOUNDARY — DISCLOSED. `launch/VIA-000_EXECUTION_BRIEF.md` states that the
     calibration memos are builder context and must not be given to a clean reviewer before that
     seat commits its initial result. My task directed me to read both memos after recording
     hypotheses, and I did. I am NOT the `codex-via000-claims` seat and this artifact is not a
     VIA-000 packet receipt, so I do not believe this contaminates the packet's claim-auditor
     slot — but a future VIA-000 claim auditor must be a different session that has not read
     this artifact or those memos, and this exposure is recorded here so that constraint is
     visible. No conclusion below is copied from either memo.
  6. WHAT WAS NOT ACCESSED. I did not open `C:/src/POPGP-custody`, any sealed holdout or seed,
     private evaluator logic, credentials, or `POPGP_Codex_Handoff.md`. I did not start the
     holdout, reveal custody material, or advance any packet lifecycle state. I did read one
     untracked, gitignored build artifact (`C:/src/POPGP/popgp/_lib/phase_flow.dll`) to test the
     builder's architecture claim; it is not part of the frozen tree and I did not use it as
     proof of anything.
  7. WHAT IS ACTUALLY EVIDENCE. Executable counterexamples: a demonstrated `ctest` false
     success, four mutation tests against the committed engine regression, six mutation tests
     against the campaign validator, five adversarial `build.bat` invocations, a PE/fatbin
     analysis, a failed DLL load in a fresh process, and the full 12-command quality suite. That
     is real evidence about specific propositions. It is not independent experimental
     confirmation of any physical claim in this repository.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: false
  builder_model_identity: "unknown"
  builder_session_id: "unknown"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  The candidate is a genuine improvement over the baseline on every axis it touches, and the two
  calibration memos are unusually honest — they disclose the mean-field limitation, the clock
  stub, the CPU-only lock file, and their own non-decisive status, and my independent inspection
  confirmed each of those disclosures rather than contradicting them. The campaign refreeze is
  clean: all seven packets bind candidate 3428a24bdd05b5742888bd204482d50a2f613cab and tree
  42fcba754ed78145054f572bd0beedf8a157562a, all seven `protocol_rule_sha256` values are unchanged
  across the activation commits, `holdout_started` is false everywhere, the campaign decision is
  `pending`, the six dependent packets are `drafted`, and the public validator is clean and
  demonstrably non-vacuous — five of six tampering mutations I applied were rejected with precise
  messages. The full 12-command quality suite passes at the frozen tree with 180 tests, exactly
  the `required_test_count` the refrozen packet declares, and a clean working tree.

  Four blocking findings nevertheless stand, and three of them are about the gap between what was
  claimed and what can be reproduced.

  BUILD-001: `ctest` exits 0 when zero tests are found. I demonstrated this directly. The
  `--test` flag in both `build.bat` and `build.sh` therefore cannot distinguish "four CUDA tests
  passed" from "no test ran", and that flag is the exact gate whose output both the memo and
  REPRODUCIBILITY.md cite as evidence.

  BUILD-002: the native build is not reproducible. There is no CUDA Toolkit on this machine, no
  `nvcc`, no `cuobjdump`, and `CUDA_PATH` is empty; the toolchain that produced the cited receipt
  was a hand-extracted CUDA 13.3.1 archive payload created after a machine-wide installer failed
  an elevation boundary, and it no longer exists. Nothing in the repository pins or provisions a
  CUDA toolkit, and no configure-time check asserts a toolkit new enough to emit sm_120.

  BENCH-001: `popgp_sim` checks no CUDA error on any kernel launch and returns 0 even if every
  launch fails, performs no readback, no checksum and no warmup. The cited 6.01e8 cell-updates/s
  is therefore unverifiable in principle from the harness that produced it.

  CAMP-001: neither the review candidate nor the frozen scientific candidate exists on the
  declared public remote — the GitHub API returns HTTP 422 for both, while the superseded
  candidate 829b866 returns 200. VIA-000's own required order instructs the reproduction runner
  to use "fresh exact-SHA clones of scientific candidate 3428a24", which is currently impossible
  from `https://github.com/whact2025/POPGP`. No exact-SHA CI evidence can exist for the reviewed
  commit, and the API confirms zero workflow runs for it.

  On scope, the answer is unambiguous and is what the campaign itself already records: the native
  backend is a mean-field product-state path. Each cell is a single 2-spinor, so mutual
  information between cells is identically zero by construction and QCMI is not computable at
  all. The clock solver is a literal identity stub, area-law pruning is commented out, and the
  only exported evolution ABI is two phase-flow entry points. VIA-300 is `drafted`, has no
  receipts, requires E4-convergent-replication and an *entangling* backend, and names "the
  backend substitutes product proxies" as a competing null to be excluded. VIA-300 has not
  passed and Tier R is not established.

  Recommendation: changes requested, 4 blocking findings.

findings:
  - id: "BUILD-001"
    severity: high
    category: code
    location: "popgp_engine/build.bat:134-140 (`ctest -C !CONFIG! --output-on-failure`) and popgp_engine/build.sh:60"
    evidence: |-
      CTest exits 0 when it discovers no tests. Demonstrated directly, independent of POPGP:
      a minimal project containing only `enable_testing()` and no tests was configured with
      `cmake -S . -B build -G Ninja`, then:
        `ctest -C Release --output-on-failure`
          -> "No tests were found!!!"   CTEST_EXIT=0
        `ctest -C Release --output-on-failure --no-tests=error`
          -> "No tests were found!!!" / "Errors while running CTest"   CTEST_EXIT=8
      `build.bat` treats that 0 as success (`if errorlevel 1 exit /b 1` does not fire), prints
      "Build Complete!" and exits 0. `grep -rn "no-tests" popgp_engine/` returns nothing, so the
      guard is absent from both the Windows and the POSIX entry point. `build.sh:4` sets `set -e`,
      which propagates a genuine ctest failure but does not convert "zero tests" into a failure.
      There is no assertion anywhere on the number of tests executed, so the harness has no lower
      bound: `gtest_discover_tests` (popgp_engine/kernel/CMakeLists.txt:68) writes the test list
      at build time, and any condition that yields an empty list — a stale or missing ctest
      include file, a build that produced no test target, a discovery run that enumerated nothing
      — is reported as a pass.
    finding: |-
      The `--test` flag of the documented native build entry point is a false-success gate. It
      reports success both when all four CUDA tests pass and when no test runs at all, because
      CTest's exit status does not distinguish those cases and the scripts do not pass
      `--no-tests=error` or assert an expected test count.
    failure_scenario: |-
      Run `build.bat --test` (or `build.sh --test`) against a build tree in which test discovery
      produced an empty list. CTest prints "No tests were found!!!", returns 0, `build.bat`
      prints "Build Complete!" and returns 0, and a receipt recorded from that invocation states
      that the native tests passed. Reproduced in isolation above: CTEST_EXIT=0 with zero tests.
    consequence: |-
      Every native-test claim produced through this entry point is unfalsifiable from its own exit
      code. This matters concretely: `reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-2026-08-13.md:50`
      states "all four CUDA tests passed" and `docs/scientific_hardening/REPRODUCIBILITY.md`
      states "Its four CUDA tests passed on the Blackwell device", and the gate that produced
      those statements cannot tell four from zero. No committed scientific number depends on it,
      so the blast radius is the native build receipt only.
    required_action: |-
      Add `--no-tests=error` to the ctest invocation in both `popgp_engine/build.bat:137` and
      `popgp_engine/build.sh:60`, and additionally assert the expected test count (currently four,
      from the four `TEST(...)` macros in `kernel/tests/`) so that a partial discovery is also a
      failure. Re-record any native-test receipt produced before the guard existed. See
      TST-BUILD-001.
    verification: confirmed-by-execution
    blocking: true

  - id: "BUILD-002"
    severity: high
    category: hardware
    location: "popgp_engine/build.bat:103-131 (no toolkit provisioning or version assertion); popgp_engine/CMakeLists.txt:5-9; docs/scientific_hardening/REPRODUCIBILITY.md:13-20 and the added native paragraph"
    evidence: |-
      The machine has the hardware but not the toolchain, verified by execution:
        `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv`
          -> "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, 595.79, 12227 MiB, 12.0"
        `where nvcc` / `nvcc --version` -> not found
        `echo $CUDA_PATH` -> empty
        `ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"` -> no such directory
        no `cuobjdump.exe`, no `nvdisasm.exe`, no `cudart`/`cusolver`/`cusparse` anywhere under
        Program Files; only the driver `C:/Windows/System32/nvcuda.dll` is present.
      The locked environment is CPU-only: `torch 2.10.0+cpu`, `torch.cuda.is_available() False`,
      `torch.version.cuda None`. So neither a native build nor a real GPU evolution can be
      executed from the repository as locked.
      The build script provisions no toolkit and asserts no version. `popgp_engine/CMakeLists.txt`
      requires only `cmake_minimum_required(VERSION 3.25)` and `find_package(CUDAToolkit REQUIRED)`;
      nothing requires a toolkit capable of emitting sm_120 (CUDA >= 12.8), so on an older toolkit
      the default `CMAKE_CUDA_ARCHITECTURES "native"` (CMakeLists.txt:5-7) resolves to something
      other than sm_120 or fails, and `--cuda-arch 120` fails late with an nvcc error rather than
      a clear precondition failure.
      The builder's own record describes the toolchain as non-standard: the memo states the
      machine-wide installer "could not cross the elevation boundary, so its documented archive
      payload was extracted into an isolated local toolchain", and REPRODUCIBILITY.md:16-17 now
      lists "CUDA 13.3.73 development tools in an isolated local extraction" as the environment.
      That extraction is not in the repository, not scripted, not pinned, and not present now.
      Consequence, confirmed by execution: the artifact that toolchain produced cannot be loaded.
      Copying the untracked `C:/src/POPGP/popgp/_lib/phase_flow.dll` into a scratch directory and
      pointing `popgp.engine._LIB_DIR` at it in a fresh interpreter gives
      `is_engine_available() -> False` and
      `OSError: Could not find module '...\phase_flow.dll' (or one of its dependencies)`,
      because its imports `cusolver64_12.dll`, `cusparse64_12.dll` and `nvcudart_hybrid64.dll`
      are unresolvable. Its build tree (`popgp_engine/build`) has also been deleted, so no
      provenance chain ties it to the frozen source.
    finding: |-
      The native Blackwell build is not reproducible from the repository. The CUDA toolkit is an
      unpinned, hand-provisioned external dependency that no repository artifact installs,
      records, or version-asserts, and `docs/scientific_hardening/REPRODUCIBILITY.md` — a document
      whose purpose is reproducibility — records a native build receipt that cannot be reproduced
      from the tree it describes.
    failure_scenario: |-
      A reproduction seat clones the frozen candidate on a clean Windows host with a Blackwell GPU
      and runs the documented `cd popgp_engine && build.bat --clean --test --cuda-arch 120`. With
      no CUDA toolkit the run fails at `find_package(CUDAToolkit REQUIRED)`; with a toolkit older
      than CUDA 12.8 it fails inside nvcc on an unknown architecture, or silently produces
      non-Blackwell code under the `native` default. Nothing in the repository tells that seat
      which toolkit version to install, and nothing verifies the one it has.
    consequence: |-
      Question 1 of this review — is the Blackwell native build reproducible and merge-ready —
      is answered "no" on the reproducibility half. The native receipt in REPRODUCIBILITY.md and
      the VIA-300 calibration memo are single-machine observations that no third party can
      currently repeat. Bounded, and I state the bound: VIA-000 declares
      `native_or_scalable_claim_in_scope: false` with `accelerator_seconds: 0`, so no frozen
      campaign packet depends on this, and no committed scientific number changes.
    required_action: |-
      Pin the CUDA toolkit the way vcpkg is already pinned: record the required minimum toolkit
      version in `popgp_engine/CMakeLists.txt` (for example
      `if(CUDAToolkit_VERSION VERSION_LESS 12.8) message(FATAL_ERROR ...)` guarded on an sm_120
      target) so an insufficient toolkit fails at configure with a clear message; document the
      exact toolkit version, installer URL and SHA-256 in `popgp_engine/kernel/README.md` rather
      than only in a calibration memo; and either mark the native paragraph in
      REPRODUCIBILITY.md as a non-reproducible single-machine calibration or make it reproducible.
      See TST-BUILD-002.
    verification: confirmed-by-execution
    blocking: true

  - id: "BENCH-001"
    severity: medium
    category: code
    location: "popgp_engine/kernel/src/main.cpp:113-131; popgp_engine/kernel/src/phase_flow.cu:150-168"
    evidence: |-
      Read verbatim at the frozen tree. `launch_phase_flow_float` and `launch_phase_flow_double`
      (phase_flow.cu:150-168) launch the kernel and return `void`; neither calls
      `cudaGetLastError()` nor checks any status. In `main.cpp` the `CUDA_CHECK` macro is applied
      to every `cudaMalloc`/`cudaMemcpy` but to none of the 200 kernel launches in the timing loop
      (main.cpp:115-121). After the loop the program calls `cudaEventSynchronize(stop)` and
      `cudaEventElapsedTime(...)` and discards both return codes, prints the throughput, frees
      device memory ignoring errors, and `return 0`.
      Therefore a run in which every launch failed — the exact symptom of a binary containing no
      device code for the present architecture, `cudaErrorNoKernelImageForDevice` — completes,
      prints a very large "Updates/sec", and exits 0. The harness has no readback of `d_alphas`/
      `d_betas`, no finiteness check, no norm check and no checksum, so nothing distinguishes
      "the kernel ran" from "the kernel never ran". There is also no warmup: `cudaEventRecord(start)`
      at main.cpp:113 immediately precedes the first launch, so CUDA context creation and module
      load are inside the measured window.
      Timing itself is done correctly — CUDA events around the loop with an explicit
      `cudaEventSynchronize(stop)` — so the measurement boundary is sound; the defect is the
      absence of any success criterion. Note in fairness that the reported 166.45 ms for 100 steps
      is inconsistent with total launch failure, which would be near-zero, so I have no evidence
      the cited run was hollow; the point is that the harness cannot establish that it was not.
    finding: |-
      The native benchmark cannot detect kernel failure and returns success unconditionally. It
      performs no error checking on any launch, no synchronization-error propagation, no result
      readback, no checksum and no warmup, so its throughput figure is not self-validating and an
      architecture mismatch would present as a very fast, successful run.
    failure_scenario: |-
      Build `popgp_sim` with `--cuda-arch 86` on the Blackwell host in a configuration where PTX
      JIT is unavailable or fails. Every `phase_flow_kernel_soa` launch returns
      `cudaErrorNoKernelImageForDevice`, no cell is ever updated, and `popgp_sim` prints a large
      "Updates/sec" and exits 0. Nothing in the program or in `build.bat` reports a problem.
    consequence: |-
      The throughput claim "100 red/black phase-flow steps in 166.45 ms, approximately 6.01e8
      cell-updates/s" recorded in the VIA-300 calibration memo and in REPRODUCIBILITY.md rests on
      a harness that cannot fail. Combined with BUILD-002, that claim is classified in this review
      as not confirmed. No scientific gate depends on it.
    required_action: |-
      Wrap each launch site in `CUDA_CHECK(cudaGetLastError())` and add a
      `CUDA_CHECK(cudaDeviceSynchronize())` after the timing loop; add a warmup iteration outside
      the measured window; read back the final `d_alphas`/`d_betas`, assert per-cell norms are
      finite and within tolerance of 1, and print a deterministic checksum (the seed is already
      fixed at `std::mt19937 gen(42)`) so a run can be compared across machines and across
      architectures. Exit nonzero on any of those failures. See TST-BENCH-001.
    verification: confirmed-by-execution
    blocking: true

  - id: "CAMP-001"
    severity: high
    category: governance
    location: "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml:6-9 (`repository`, `candidate_commit`); launch/VIA-000_EXECUTION_BRIEF.md reproduction-runner prompt"
    evidence: |-
      Queried the public GitHub API unauthenticated, no credentials used:
        GET /repos/whact2025/POPGP -> 200, `"private": false`
        GET /repos/whact2025/POPGP/commits/dfe222ce3a173f617471d1979f0021ae6ec23ebf -> HTTP 422
        GET /repos/whact2025/POPGP/commits/3428a24bdd05b5742888bd204482d50a2f613cab -> HTTP 422
        GET /repos/whact2025/POPGP/commits/829b866d731865060927c365b7848ce0d710736c -> HTTP 200
        GET /repos/whact2025/POPGP/actions/runs?head_sha=dfe222ce... -> total_count 0
        GET /repos/whact2025/POPGP/actions/runs?head_sha=3428a24b... -> total_count 0
      So the superseded candidate is published but the refrozen scientific candidate and the
      review candidate are not: both exist only locally. The CI run the VIA-000 memo cites,
      31520334982, is real and green but its `head_sha` is 829b866d731865060927c365b7848ce0d710736c
      — the superseded candidate — which the memo itself correctly flags.
      The campaign declares `repository: https://github.com/whact2025/POPGP` and the execution
      brief instructs the reproduction runner to "Use fresh exact-SHA clones of scientific
      candidate 3428a24bdd05b5742888bd204482d50a2f613cab". That instruction cannot be carried out
      against the declared remote today.
    finding: |-
      The frozen scientific candidate is not present on the campaign's declared public repository,
      so the campaign's own required execution order is not currently executable: no external seat
      can obtain the exact-SHA tree it is required to reproduce, and no exact-SHA CI evidence can
      exist for it.
    failure_scenario: |-
      The reproduction runner follows step 4 of `VIA-000_EXECUTION_BRIEF.md` and attempts
      `git clone https://github.com/whact2025/POPGP && git checkout 3428a24bdd05b5742888bd204482d50a2f613cab`.
      The checkout fails because the object is not published. The runner either stalls or silently
      substitutes a local copy supplied by the operator, which destroys the independence the
      campaign is built to provide and would make the resulting receipt unverifiable by anyone
      else.
    consequence: |-
      VIA-000 is not safe to begin. The packet is otherwise correctly frozen and validator-clean —
      `holdout_started: false`, decision `pending`, dependents `drafted` — but the first
      irreversible step of the protocol, exposing the sealed holdout, would be taken on a
      candidate that no independent party can fetch. This is recoverable by a push and is not a
      defect in the science or the machinery.
    required_action: |-
      Push `3428a24bdd05b5742888bd204482d50a2f613cab` (and the activation/handoff commits the
      brief names, `8c3c4d53cddf28b0c51dc7b6c21eab49d28d0f9d` and
      `99f13dcc04d556892495e8e2361d257b1b2c113b`) to the declared public repository before any
      seat begins VIA-000, confirm a green exact-SHA CI run for the refrozen candidate, and record
      that run id in the campaign. Do not start the holdout until an external clone of the exact
      SHA succeeds. See TST-CAMP-001.
    verification: confirmed-by-execution
    blocking: true

  - id: "ENG-001"
    severity: medium
    category: code
    location: "popgp/engine.py:81-84 (`load_kwargs = {\"winmode\": 0} if os.name == \"nt\" else {}`); tests/unit/test_engine.py:1-44"
    evidence: |-
      The committed regression is a genuine control for the two changes it was written for, which
      I verified by mutation on disposable `git archive` copies (baseline: `1 passed`):
        revert handle retention to `os.add_dll_directory(str(d))` (discard return) -> 1 failed
        delete the `Path(cuda_path) / "bin" / "x64"` entry                          -> 1 failed
        delete the whole `CUDA_PATH` discovery block                                -> 1 failed
      So the answer to "would the committed tests fail if handle retention or toolkit discovery
      were removed" is yes for both. But a fourth mutation is not caught:
        replace `load_kwargs = {"winmode": 0} if os.name == "nt" else {}` with `load_kwargs = {}`
                                                                                   -> 1 passed
      `winmode=0` is the other half of the same contract: on CPython 3.8+ for Windows, directories
      registered with `os.add_dll_directory()` are only consulted when the library is loaded with
      the `LOAD_LIBRARY_SEARCH_*` semantics that `winmode=0` selects, as the comment at
      engine.py:81-83 itself states. The test fully mocks `os.add_dll_directory` and never
      exercises `ctypes.CDLL`, so it verifies bookkeeping only.
    finding: |-
      The Windows DLL-resolution contract is only half guarded. Handle retention and toolkit
      discovery are pinned by the new test; `winmode=0`, without which those registered
      directories are not searched at all, can be deleted with the suite still green.
    failure_scenario: |-
      A future refactor drops `winmode=0` from `_load_library` as redundant. `uv run pytest -q`
      reports 180 passed. On a real Windows host with a CUDA toolkit, `ctypes.CDLL` then resolves
      `phase_flow.dll` without searching the registered CUDA directories and fails on its
      transitive `cusolver64_12.dll` / `cusparse64_12.dll` dependencies, reintroducing exactly the
      loader defect this candidate was written to fix.
    consequence: |-
      A regression in the fix's own subject matter would not be caught by CI. Bounded: no current
      behavior is wrong, and the failure would be a loud load error rather than a silent numerical
      one. It cannot be exercised on the locked CPU-only environment, which is why the gap exists.
    required_action: |-
      Extend `tests/unit/test_engine.py` to assert the load path, not only the registration path:
      monkeypatch `ctypes.CDLL` and assert `_load_library()` invokes it with `winmode=0` on
      `os.name == "nt"` and without it elsewhere. See TST-ENG-001.
    verification: confirmed-by-execution
    blocking: false

  - id: "BUILD-003"
    severity: medium
    category: code
    location: "popgp_engine/build.bat:23-40 (`--cuda-arch` parsing) and :125 (configure); popgp_engine/build.sh:17-23"
    evidence: |-
      Executed against a copy of the script, exit codes captured through a PowerShell wrapper:
        `build.bat --bogus`      -> "Unknown parameter: --bogus"                            EXIT=1
        `build.bat --cuda-arch`  -> "Error: --cuda-arch requires a CMake CUDA architecture value." EXIT=1
      Both correct. But the value itself is never validated:
        `build.bat --cuda-arch --clean --test`
          -> "--- POPGP Engine Build (Release, CUDA architecture --clean) ---"
          The `--clean` flag was consumed as the architecture value AND `--test` was silently
          dropped, so a user who typo'd the invocation gets an unclean, untested build.
        `build.bat --cuda-arch not-an-arch`
          -> "--- POPGP Engine Build (Release, CUDA architecture not-an-arch) ---", then a CMake
          configure error and EXIT=1. Fail-closed, but only downstream.
        `build.bat --clean --test --cuda-arch 86`
          -> "--- POPGP Engine Build (Release, CUDA architecture 86) ---". CMake accepts 86, nvcc
          emits `compute_86`/`sm_86` including PTX, and the result JITs on the sm_120 device. The
          script reports success for a non-Blackwell binary.
      Nothing in the build verifies the produced artifact. There is no `cuobjdump`/`nvdisasm`
      assertion step, so the claim that `phase_flow.dll` contains native sm_120 cubins is not
      established by the build itself. My own inspection of the untracked prebuilt DLL is
      consistent with that claim but does not confirm it: the PE has a 92,392-byte `.nv_fatb`
      section, the only architecture token present anywhere in the file is `sm_120` (no other
      `sm_*`, no `compute_*`), it imports `cusolver64_12.dll`/`cusparse64_12.dll`/`nvcudart_hybrid64.dll`,
      and its PE timestamp is 2026-08-13T16:44:06Z. That is a string scan, not a fatbin decode; no
      `cuobjdump` exists on this machine to decode it.
    finding: |-
      `--cuda-arch` accepts any string without validation, silently swallows a following flag, and
      the build performs no verification that the produced binary actually contains device code
      for the requested architecture. A wrong-architecture build is reported as a successful one.
    failure_scenario: |-
      An operator intending a frozen Blackwell receipt runs `build.bat --cuda-arch --clean --test`.
      The architecture becomes the literal string `--clean`, `--test` is dropped, and either the
      configure fails with a confusing CMake message or — with `--cuda-arch 86` instead — the build
      succeeds, produces Ampere code, prints "Build Complete!", and is recorded as a Blackwell
      build receipt.
    consequence: |-
      Frozen build receipts cannot be trusted to describe the architecture they were produced for
      without an out-of-band `cuobjdump` step performed by hand, which is exactly how the current
      sm_120 claim was made. No committed number is affected.
    required_action: |-
      Validate the `--cuda-arch` value against an allow-list (`native`, `all`, `all-major`, or a
      numeric/`nn-real`/`nn-virtual` form) and reject a value beginning with `--`; and add a
      post-build verification step that runs `cuobjdump --list-elf` on the produced
      `phase_flow.dll` and fails unless every listed ELF matches the requested architecture. See
      TST-BUILD-003.
    verification: confirmed-by-execution
    blocking: false

  - id: "SCOPE-001"
    severity: medium
    category: science
    location: "popgp_engine/kernel/src/clock.cu:24-46; popgp_engine/kernel/src/area_law.cu:26-36 and :52-55; popgp_engine/kernel/src/phase_flow.cu:29-50; popgp/engine.py:111-120"
    evidence: |-
      Determined by reading the frozen source, independently of the memos, before opening them.
      SCOPE. Each cell is a single 2-spinor `(alpha, beta)` (phase_flow.cu:114-115), evolved by
      `apply_heisenberg_step`, whose own comment at :29 names it "Full Heisenberg mean-field
      interaction": cell 1 is rotated about cell 2's Bloch vector and vice versa. This is a product
      state. Its bipartite entanglement is identically zero, so mutual information between cells is
      identically zero and QCMI is not defined by any quantity the kernel holds. There is no
      density matrix, no reduced state and no entropy anywhere in `kernel/src/`. The entire
      exported evolution ABI is `launch_phase_flow_float` and `launch_phase_flow_double`
      (engine.py:111-120). I verified the unitary is nonetheless correct: with `s = sin(theta)/|h|`,
      `|U00|^2 + |U10|^2 = cos^2(theta) + s^2(hz^2 + 4|p|^2) = 1` exactly, which is consistent with
      the memo's reported per-cell norm error of 5.55e-16.
      STUBS. `clock.cu:39-41` reads "Placeholder: Identity Map (No Gravity) / Phi = Rho" and the
      function `solve_clock_potential` creates cuSOLVER and cuSPARSE handles, copies `rho` to `phi`
      with `cudaMemcpyDeviceToDevice`, and destroys the handles without ever using them. The
      emergent-clock solver is not implemented natively.
      DISABLED PATH. In `area_law.cu:52-55` the freeze branch body is commented out
      (`// node_active_mask[idx] = 0;`), so `prune_bulk_kernel` can only ever re-activate a node,
      never prune one; the "area law pruner" is a partial no-op.
      DEVICE PRINTF. `area_law.cu:26-29` and :35 execute `printf` inside a device kernel for
      `idx < 5`, which serializes and writes to stdout on every call.
      This finding is filed as accurate scope documentation, not as an undisclosed overclaim: the
      VIA-300 packet's `known_failure_to_retain` already states "the current mean-field CUDA
      backend cannot compute MI/QCMI and is Heisenberg-only", README.md:78-95 states the mean-field
      backend "cannot compute mutual information and the locality stage fails explicitly instead of
      substituting a false MI proxy", and both memos disclose the mean-field limitation and the
      clock stub. My independent reading confirms every one of those disclosures.
    finding: |-
      The native backend computes exactly one thing: mean-field product-state phase flow on a
      weighted graph. It cannot compute mutual information, conditional mutual information, the
      locality/geometry/clock pipeline, or any VIA-300 observable. The native clock solver is an
      identity stub and the area-law pruner's freeze path is commented out.
    failure_scenario: |-
      A reader treats the Blackwell build receipt as progress toward scalable scientific evidence
      and calls `solve_clock_potential` expecting an emergent-time solution. It returns `rho`
      unchanged, having created and destroyed a cuSOLVER handle it never used, with no error and
      no warning that the result is an identity map.
    consequence: |-
      Nothing in this candidate advances Tier R. Throughput, finite output, per-cell norm
      preservation and native unit tests are consistent with a correct mean-field kernel and are
      not evidence of mechanism viability. The disclosures are already correct, so the residual
      exposure is that `solve_clock_potential` is a silently-wrong callable rather than an
      explicit `NotImplementedError`.
    required_action: |-
      Make the stub loud rather than silent: have `solve_clock_potential` return a nonzero status
      or fail explicitly until the CSR construction is implemented, and either restore or delete
      the commented-out prune branch in `area_law.cu:52-55` rather than shipping a kernel whose
      documented purpose is disabled. Remove the device-side `printf` calls from `area_law.cu` or
      guard them behind a debug macro so they cannot contaminate a timed path. See TST-SCOPE-001.
    verification: read-only
    blocking: false

  - id: "DOC-001"
    severity: low
    category: claim
    location: "popgp_engine/kernel/README.md:11-15"
    evidence: |-
      The "Implementation Steps (Phase 2)" checklist marks all four items complete, including
      "[x] Implement `phase_flow_kernel` with graph coloring for parallel safety". There is no
      graph coloring in `phase_flow.cu`: the kernel is flat over edges and performs an unguarded
      read-modify-write on `cell_alphas[s]`, `cell_betas[s]`, `cell_alphas[d]`, `cell_betas[d]`.
      I initially hypothesised this was a live data race and recorded it as such before reading
      the memos; my own execution REFUTED that for every supported path, and I record the
      refutation rather than the hypothesis. The coloring exists, correctly, in the callers:
      `popgp/backend.py:540-557` `_edge_color_batches` greedily partitions edges into node-disjoint
      batches, `GPUBackend.evolve` (backend.py:524-538) launches one batch at a time, and its
      docstring states the reason verbatim — "Edges in a batch never share a node. Launching all
      lattice edges in a single kernel would race on cell amplitudes and make evolution
      nondeterministic." The native benchmark independently applies a red/black 1-D coloring at
      `kernel/src/main.cpp:51-64`. So the safety property holds end to end at both call sites; only
      its stated location is wrong.
      Separately, the same checklist marks the phase-2 items complete while `clock.cu` is an
      identity stub and the `area_law.cu` prune branch is commented out (see SCOPE-001).
    finding: |-
      `kernel/README.md` attributes parallel safety to the kernel when it is actually provided by
      every caller, and presents the phase-2 implementation checklist as complete when two of the
      components it covers are a stub and a disabled branch.
    failure_scenario: |-
      A contributor reads "graph coloring for parallel safety" as a property of
      `phase_flow_kernel_soa`, writes a new caller that passes a full uncolored edge list to the
      public `Engine.step`, and gets silently nondeterministic evolution on any graph with a
      node of degree greater than one. `Engine.step`'s own docstring (popgp/engine.py:149) states
      no such precondition.
    consequence: |-
      Documentation-only today, because both shipped callers colour correctly and the risk is
      explicitly documented at the one place a maintainer is most likely to read it
      (`GPUBackend.evolve`). It matters because the public `Engine.step` boundary carries no
      precondition and the README points the reader at the wrong layer.
    required_action: |-
      Correct `kernel/README.md:14` to say that node-disjoint batching is a caller contract
      supplied by `GPUBackend._edge_color_batches` and by the benchmark's red/black partition, not
      a kernel property; mark the clock and prune items as incomplete; and document the
      node-disjoint precondition in `Engine.step`'s docstring. See TST-DOC-001.
    verification: confirmed-by-execution
    blocking: false

  - id: "CAMP-002"
    severity: low
    category: governance
    location: "scripts/check_viability_campaign.py (lifecycle ordering); reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md:14-17"
    evidence: |-
      The campaign README states the six dependent packets "stay in `drafted` lifecycle state
      until their machine-declared prerequisites pass". Executed against a disposable clone at the
      candidate, promoting a dependent packet without any prerequisite being satisfied is accepted:
        set `VIA-300.lifecycle_phase: drafted -> preregistered`
          -> "Viability campaign contract is valid."   exit 0
      Five other tampering mutations were correctly rejected, which is what makes this one worth
      recording rather than dismissing:
        `holdout_started: false -> true`      -> "packet VIA-000: holdout cannot start before attack phase"      exit 1
        `lifecycle_phase -> adjudicated`      -> "adjudication requires decisive receipts"                        exit 1
        `required_test_count: 180 -> 999`     -> "packet rules differ from protocol snapshot"                     exit 1
        campaign `outcome: pending -> passed` -> "declared outcome passed != computed pending"                    exit 1
        packet `candidate_commit -> dfe222c`  -> "protocol_rule_sha256 differs from packet rules"                 exit 1
      The dependency DAG is machine-declared in `schemas/viability/requirements-v2.json`
      (VIA-300 wave 1, `dependencies: ["VIA-000"]`), and the validator checks that graph for
      cycles and wave ordering, but it does not gate a packet's lifecycle promotion on its
      dependencies' outcomes.
    finding: |-
      A dependent packet can be advanced from `drafted` to `preregistered` while its declared
      prerequisite is still `pending`, so the README's stated lifecycle invariant is not
      mechanically enforced at that transition.
    failure_scenario: |-
      A maintainer sets `VIA-300.lifecycle_phase: preregistered` while VIA-000 is unadjudicated.
      `scripts/check_viability_campaign.py` reports "Viability campaign contract is valid." and a
      reader of the campaign directory sees a Tier-R packet apparently cleared to proceed.
    consequence: |-
      Presentational only at this severity: `preregistered` unlocks nothing on its own, the
      dangerous transitions are gated (holdout start requires the attack phase, adjudication
      requires decisive receipts), and the campaign decision remains computed rather than
      declared. It is a gap between a documented invariant and its enforcement, not a route to a
      false terminal outcome.
    required_action: |-
      Either enforce the invariant — reject a lifecycle phase above `drafted` for any packet whose
      `requirements-v2.json` dependencies have not reached a passing adjudicated outcome — or
      soften `reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md:14-17` to state that the
      binding gate is adjudication rather than lifecycle phase. See TST-CAMP-002.
    verification: confirmed-by-execution
    blocking: false

requested_tests:
  - id: "TST-BUILD-001"
    description: |-
      Add `--no-tests=error` to the ctest invocation in `popgp_engine/build.bat:137` and
      `popgp_engine/build.sh:60`, and assert the expected native test count (four, from the four
      `TEST(...)` macros under `popgp_engine/kernel/tests/`). Demonstrate the guard failing on the
      executed counterexample that currently passes: configure a build tree with `enable_testing()`
      and no registered tests and run the `--test` path; today `ctest` prints "No tests were
      found!!!" and returns 0 while `build.bat` prints "Build Complete!" and returns 0, and with
      `--no-tests=error` ctest returns 8.
    rationale: |-
      BUILD-001. The documented native verification gate cannot distinguish four passing tests
      from zero executed tests, and it is the gate that produced the "four CUDA tests passed"
      statements in the calibration memo and REPRODUCIBILITY.md.
    blocking: true
  - id: "TST-BUILD-002"
    description: |-
      Add a configure-time precondition in `popgp_engine/CMakeLists.txt` that fails with a clear
      message when the detected `CUDAToolkit_VERSION` cannot emit the requested architecture — at
      minimum, require CUDA >= 12.8 whenever `CMAKE_CUDA_ARCHITECTURES` resolves to 120 or when it
      is `native` on a device of compute capability >= 12.0. Record the exact toolkit version,
      installer URL and SHA-256 in `popgp_engine/kernel/README.md`. Demonstrate the guard by
      configuring with an older toolkit (or a stubbed `CUDAToolkit_VERSION`) and showing a
      configure-time failure instead of a late nvcc error or a silent non-Blackwell build.
    rationale: |-
      BUILD-002. No repository artifact pins, provisions or version-asserts the CUDA toolkit; the
      toolchain that produced the cited receipt was hand-extracted and is absent, and no CUDA
      toolkit, `nvcc` or `cuobjdump` exists on this machine.
    blocking: true
  - id: "TST-BENCH-001"
    description: |-
      Harden `popgp_engine/kernel/src/main.cpp`: check `cudaGetLastError()` after every launch,
      `cudaDeviceSynchronize()` after the timing loop, add one warmup iteration outside the
      measured window, read back the final amplitudes, assert every value is finite and every
      per-cell norm is within 1e-12 of 1, and print a deterministic checksum from the fixed seed
      42. Exit nonzero on any failure. Demonstrate that the hardened harness fails when the binary
      contains no device code for the present architecture — for example by building with a
      deliberately mismatched `--cuda-arch` and PTX JIT disabled — where the current harness
      prints a large throughput and returns 0.
    rationale: |-
      BENCH-001. The benchmark ignores all CUDA error status, never reads back results, and
      returns 0 even if every kernel launch failed, so the cited 6.01e8 cell-updates/s is not
      self-validating.
    blocking: true
  - id: "TST-CAMP-001"
    description: |-
      Before any seat begins VIA-000, publish `3428a24bdd05b5742888bd204482d50a2f613cab`,
      `8c3c4d53cddf28b0c51dc7b6c21eab49d28d0f9d` and `99f13dcc04d556892495e8e2361d257b1b2c113b`
      to the declared repository and add a preflight check that fails unless
      `git clone <repository> && git checkout <candidate_commit>` succeeds from a clean network
      location and a green exact-SHA CI run exists. Demonstrate it failing at the current state,
      where `GET /repos/whact2025/POPGP/commits/3428a24b...` returns HTTP 422 and
      `actions/runs?head_sha=3428a24b...` returns `total_count: 0`.
    rationale: |-
      CAMP-001. VIA-000's required order instructs the reproduction runner to use fresh exact-SHA
      clones of the scientific candidate, which is impossible from the declared public remote.
    blocking: true
  - id: "TST-ENG-001"
    description: |-
      Extend `tests/unit/test_engine.py` to cover the load path as well as the registration path:
      monkeypatch `ctypes.CDLL`, invoke `engine._load_library()` with `os.name` forced to `"nt"`,
      and assert it was called with `winmode=0`; assert the keyword is absent on POSIX. The test
      must FAIL on the executed mutation that currently passes — replacing
      `load_kwargs = {"winmode": 0} if os.name == "nt" else {}` with `load_kwargs = {}` leaves the
      suite at 180 passed.
    rationale: |-
      ENG-001. Handle retention and toolkit discovery are pinned by the new regression, but
      `winmode=0` — without which the registered directories are never searched — is not.
    blocking: false
  - id: "TST-BUILD-003"
    description: |-
      Validate the `--cuda-arch` value in `build.bat` and `build.sh` against an allow-list
      (`native`, `all`, `all-major`, or `nn`/`nn-real`/`nn-virtual`) and reject any value beginning
      with `--`. Add a post-build step running `cuobjdump --list-elf` on the produced
      `phase_flow.dll` that fails unless every listed ELF matches the requested architecture.
      Demonstrate all three currently-passing counterexamples failing:
      `--cuda-arch --clean --test` (which today yields "CUDA architecture --clean" and silently
      drops `--test`), `--cuda-arch not-an-arch`, and `--cuda-arch 86` on an sm_120 device.
    rationale: |-
      BUILD-003. The architecture argument is unvalidated, silently swallows a following flag, and
      no step verifies that the produced binary contains code for the requested architecture.
    blocking: false
  - id: "TST-SCOPE-001"
    description: |-
      Make the unimplemented native paths fail loudly: change `solve_clock_potential` to return a
      nonzero status (or refuse to run) while it is an identity map, and add a native test
      asserting that it does not silently return `rho`. Restore or delete the commented-out prune
      branch at `area_law.cu:52-55` and add a test that exercises whichever behaviour is retained.
      Remove or debug-guard the device-side `printf` calls at `area_law.cu:26-29` and :35 and
      assert no kernel in a timed path emits output.
    rationale: |-
      SCOPE-001. `solve_clock_potential` is a silently-wrong callable, the area-law prune branch is
      disabled, and device printf contaminates any timed path.
    blocking: false
  - id: "TST-DOC-001"
    description: |-
      Correct `popgp_engine/kernel/README.md:14` to attribute node-disjoint batching to the caller
      contract (`GPUBackend._edge_color_batches`, and the red/black partition in
      `kernel/src/main.cpp:51-64`) rather than to the kernel, and mark the clock and prune
      checklist items incomplete. Document the node-disjoint precondition in `Engine.step`'s
      docstring and add a unit test asserting that documented precondition is present, so the
      public API boundary cannot lose it silently.
    rationale: |-
      DOC-001. The README attributes parallel safety to the kernel, which has none; the property is
      supplied by both callers, and the public `Engine.step` boundary states no precondition.
    blocking: false
  - id: "TST-CAMP-002"
    description: |-
      Either enforce or soften the documented lifecycle invariant. If enforcing: extend
      `scripts/check_viability_campaign.py` to reject any packet whose `lifecycle_phase` exceeds
      `drafted` while a declared `requirements-v2.json` dependency has not reached a passing
      adjudicated outcome, and demonstrate it failing on the executed mutation that currently
      passes — setting `VIA-300.lifecycle_phase: preregistered` while VIA-000 is `pending` yields
      "Viability campaign contract is valid." today.
    rationale: |-
      CAMP-002. The campaign README states dependents remain `drafted` until prerequisites pass;
      the validator does not gate that transition.
    blocking: false

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "BUILD-001-ctest-zero-test-false-success"
  predicted_outcome: |-
    If TST-BUILD-001 is implemented as specified and the `--test` path is exercised against a
    build tree in which `gtest_discover_tests` registered no tests, `ctest -C Release
    --output-on-failure` will print "No tests were found!!!" and return 0 at
    dfe222ce3a173f617471d1979f0021ae6ec23ebf, causing `build.bat` to print "Build Complete!" and
    return 0; the same invocation with `--no-tests=error` will return 8 and `build.bat` will
    return 1. After the guard and the expected-count assertion are added, a build in which all
    four native tests are discovered and pass will still return 0, and any build discovering fewer
    than four will return nonzero.
  predicted_failure_mode: |-
    The prediction is falsified if `ctest` returns nonzero for an empty test set on the reference
    CMake version. The most likely benign cause of an apparent refutation is a newer CMake that
    has changed the default of `--no-tests`: CMake 4.x still defaults to `ignore`, and this
    machine's CMake 4.3.2 reproduced exit 0, but a future default change to `error` would close
    the hole without any repository change. A replication must therefore print `cmake --version`
    and `ctest --version` alongside the exit code. A genuine refutation would require
    `gtest_discover_tests` to be shown incapable of ever yielding an empty registration set, which
    would remove the reachability rather than the exit-code behaviour.
  confidence_statement: |-
    High confidence in BUILD-001, BUILD-002, CAMP-001 and the campaign-refreeze verification: each
    rests on a command I ran with its exact output recorded, and the refreeze conclusions were
    derived from raw Git blob bytes rather than from the working tree, which matters because
    `.gitattributes` covers only `reviews/viability/**` and `protocols/**` while `core.autocrlf`
    is true, so worktree hashes of `schemas/` and `scripts/` files differ from their blob hashes.
    The validator hashes blob bytes and is correct on this point. High confidence in the scope
    conclusions, which follow from reading the kernel source directly and are independently
    corroborated by the frozen VIA-300 packet's own `known_failure_to_retain`. Moderate confidence
    in BENCH-001's severity: the mechanism is certain from the source, but I could not execute the
    benchmark and the cited 166.45 ms is inconsistent with a wholly failed run. Low confidence —
    explicitly not confirmed — on the central hardware claim: I could not verify that
    `phase_flow.dll` contains executable sm_120 SASS, because no `cuobjdump` or `nvdisasm` exists
    on this machine; my PE analysis found a real 92,392-byte `.nv_fatb` section whose only
    architecture token is `sm_120`, which is consistent with the claim but is a string scan, not a
    fatbin decode. No conclusion in this artifact depends on that DLL.

recommendation:
  approve: false
  blocking_findings: 4
  rationale: |-
    blocking_findings = BUILD-001 + BUILD-002 + BENCH-001 + CAMP-001 = 4.

    None of the four is a defect in the science, and none invalidates the campaign machinery,
    which I attacked and found sound. They are all the same class of problem: a claim whose
    supporting evidence cannot be reproduced or cannot fail. BUILD-001 is a verification gate that
    returns success when nothing was verified. BUILD-002 is a build receipt whose toolchain is not
    pinned, not provisioned and no longer present. BENCH-001 is a throughput number produced by a
    harness with no success criterion. CAMP-001 is a frozen candidate that the campaign's own
    required execution order cannot fetch.

    What holds up is substantial and should be recorded plainly. The candidate replaces a
    hard-coded `sm_86` default with a configure-time architecture selection and an explicit
    `--cuda-arch` path; it fixes a real Windows loader defect by retaining the
    `os.add_dll_directory()` handles, and I confirmed by mutation that the new regression fails
    when either the retention or the toolkit discovery is removed; it adds a `bin\x64` toolkit
    layout; it pins the vcpkg baseline in the manifest as well as the script, and the pinned
    commit `e5a1490e14` checks out and bootstraps a signature-validated tool; and it replaces
    `GTest::gtest_main` with an explicit entry point that is present and correct. The campaign
    refreeze is clean from raw Git bytes: all seven packets bind the scientific candidate and its
    tree, all seven `protocol_rule_sha256` values survive the activation commits unchanged, the
    only post-snapshot edits are pointer fields, `holdout_started` is false everywhere, the
    decision is `pending`, the dependents are `drafted`, and the validator rejected five of six
    tampering mutations with precise messages. The full 12-command suite passes with 180 tests —
    exactly the refrozen `required_test_count` — and a clean tree. Both calibration memos are
    candid, correctly labelled non-decisive, and every scope limitation they assert was confirmed
    by my own independent reading rather than taken on trust.

    Approval is withheld only on the four items above, each of which has a bounded, concrete
    corrective action. This recommendation concerns merge-readiness of software and campaign
    machinery under the declared contract. It is not a statement about POPGP's physical theory:
    it does not establish VIA-300, it does not establish Tier R mechanism viability, and it is not
    external scientific validation.
```

## 1. Method

**Frozen worktree.** `C:\src\POPGP-review-blackwell-viability-1`, created with
`git worktree add -b review/blackwell-viability-1 ... dfe222ce3a173f617471d1979f0021ae6ec23ebf`.

**Verified before review and again at the end.** `git rev-parse HEAD` →
`dfe222ce3a173f617471d1979f0021ae6ec23ebf`; `git rev-parse "HEAD^{tree}"` →
`4abafab7c68f802d304007ca73f9b27f202800bb`, matching the declared candidate tree exactly;
`git rev-parse "3428a24bdd05b5742888bd204482d50a2f613cab^{tree}"` →
`42fcba754ed78145054f572bd0beedf8a157562a`, matching the declared scientific-candidate tree;
`git merge-base --is-ancestor` confirmed both `f22694b7…` and `3428a24b…` are ancestors of the
candidate; `git status --porcelain` was empty before any work and empty at the end; branch
`review/blackwell-viability-1`.

**The frozen tree was never mutated.** Every mutation, counterexample and A/B ran in a disposable
copy — `git archive dfe222c | tar -x` for the engine mutations, a `git clone --no-checkout` for the
campaign-validator mutations, and standalone scratch directories for the CMake/ctest and
`build.bat` probes. No `git add`, `git commit` or `git checkout` was executed in the reviewer
worktree except the final artifact commit.

**Ordering discipline.** Thirteen hypotheses (H1–H13) with expected failure modes were written to
private notes outside the repository *before* either calibration memo was opened. Two were
subsequently refuted by my own execution and are reported as refuted, not as findings.

## 2. Required quality commands — exact results

All run at the frozen tree in the reviewer worktree.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | locked environment resolved |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines, balanced structure |
| `uv run pytest -q` | 0 | **180 passed in 875.59s** |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | — |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | — |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | — |
| `uv run python -m examples.physics_qg.source_law` | 0 | — |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | — |
| `uv run python -m examples.physics_qg.ca_model` | 0 | — |
| `uv run python scripts/check_validation_artifacts.py` | 0 | `Validation artifact contracts and required visual outputs are valid.` |
| `uv run python scripts/check_viability_campaign.py …/CAMPAIGN.yaml` | 0 | `Viability campaign contract is valid.` |
| `git status --porcelain` (after all of the above) | 0 | **empty** |

180 collected/passed equals the refrozen packet's `required_test_count: 180` exactly (raised from
179 by the single new `tests/unit/test_engine.py` test).

## 3. Independent hardware and toolchain findings

| Item | Result | How |
|---|---|---|
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU | `nvidia-smi --query-gpu=...` |
| Compute capability | **12.0** (= `sm_120`) — genuinely Blackwell | same |
| Driver / CUDA API | 595.79 / 13.2 | same |
| Memory | 12,227 MiB | same |
| `nvcc` | **absent** | `where nvcc`, `nvcc --version` |
| CUDA Toolkit | **absent** — no `…/NVIDIA GPU Computing Toolkit/CUDA` directory | `ls` |
| `cuobjdump` / `nvdisasm` | **absent** | `ls` |
| `CUDA_PATH` | **empty** | `env` |
| CUDA runtime libs | **absent** (only driver `nvcuda.dll` present) | bounded `find` |
| PyTorch in lock file | `2.10.0+cpu`, `cuda_available False`, `version.cuda None` | venv python |
| CMake / Ninja / vswhere | 4.3.2 / present / present | `where`, `--version` |

**Consequence:** the native build cannot be reproduced on this machine, and the untracked
`phase_flow.dll` cannot be loaded — `is_engine_available()` returns `False` and a direct
`ctypes.CDLL(..., winmode=0)` raises `OSError: Could not find module … (or one of its
dependencies)` because `cusolver64_12.dll`, `cusparse64_12.dll` and `nvcudart_hybrid64.dll` are
unresolvable.

## 4. Claim classification

| Claim | Classification |
|---|---|
| GPU is Blackwell, compute capability 12.0, driver 595.79, 12,227 MiB | **confirmed by independent execution** |
| Pinned vcpkg commit `e5a1490e14…` exists, checks out, bootstraps a signature-validated tool | **confirmed by independent execution** |
| Committed engine regression fails if handle retention is removed | **confirmed by independent execution** (mutation) |
| Committed engine regression fails if toolkit discovery is removed | **confirmed by independent execution** (mutation) |
| Campaign refreeze bindings, rule hashes, `holdout_started: false`, decision `pending` | **confirmed by independent execution** (raw Git bytes + validator + 6 mutations) |
| 180 tests, six examples, artifact contracts, clean tree | **confirmed by independent execution** |
| Cited CI run `31520334982` is green — but at superseded SHA `829b866` | **confirmed by independent execution** (public API) |
| `phase_flow.dll` contains native `sm_120` cubins | **not confirmed** — no `cuobjdump`; PE scan shows a real `.nv_fatb` section whose only architecture token is `sm_120`, which is *consistent with* but does not establish the claim |
| `nvcc 13.3.73` / MSVC 19.50 / `build.bat --clean --test --cuda-arch 120` succeeded | **not confirmed** — toolchain absent, build tree deleted, no provenance chain |
| "All four CUDA tests passed" | **not confirmed** — the gate that produced it returns 0 for zero tests (BUILD-001) |
| 166.45 ms / 6.01e8 cell-updates/s | **not confirmed** — harness has no success criterion (BENCH-001) |
| Native backend is mean-field product-state; cannot compute MI/QCMI; clock solver is a stub | **confirmed by independent inspection**, matching the builder's own disclosure |
| Exact-SHA CI exists for the reviewed candidate | **refuted** — `total_count: 0`; commit returns HTTP 422 |
| Kernel implements graph coloring for parallel safety (`kernel/README.md:14`) | **refuted** — coloring is in the callers, not the kernel; the safety property nonetheless holds end to end |

## 5. Hypotheses I refuted against myself

Recording these because an artifact that reported them as findings would have been wrong.

- **Data race in the phase-flow kernel.** I predicted, before reading the memos, that the flat
  edge-parallel kernel would race on shared nodes and produce nondeterministic evolution. It would
  — but no supported caller reaches that state. `GPUBackend.evolve` (`popgp/backend.py:524-538`)
  batches through `_edge_color_batches`, which greedily partitions edges into node-disjoint sets,
  and its docstring names the exact hazard. `kernel/src/main.cpp:51-64` applies a red/black 1-D
  colouring. Only the documentation attribution is wrong (DOC-001).
- **Contract-file hash mismatch.** Worktree bytes of `schemas/viability/requirements-v2.json` hash
  to `fdf65a6f…` against a declared `632528e8…`, which looked like a broken binding. It is not:
  `.gitattributes` covers only `reviews/viability/**` and `protocols/**`, `core.autocrlf` is true,
  and the declared value is the **blob** hash. The validator reads blob bytes via `_git_blob(...)`
  and normalises line endings when comparing working-tree content, so it is correct and
  platform-independent.

## 6. Required conclusions

**1. Is the Blackwell native build reproducible and merge-ready?**
**No.** The source changes are a real improvement and are internally sound, but the build is not
reproducible: no repository artifact pins, provisions or version-asserts the CUDA toolkit
(BUILD-002); the toolchain that produced the cited receipt was a hand-extracted archive payload
that no longer exists; the `--test` gate returns success when nothing was tested (BUILD-001); and
nothing verifies that the produced binary matches the requested architecture (BUILD-003).

**2. Is executable CUDA/native hardware feasibility demonstrated?**
**Partly, and not to this reviewer.** The Blackwell hardware and driver are independently
confirmed present and correct (compute capability 12.0). Executable native/CUDA feasibility is
**not confirmed**: there is no CUDA toolkit on this machine, the locked environment is CPU-only,
the prebuilt DLL does not load, and no `cuobjdump` exists to decode its fatbinary. The builder's
evidence is a single-machine observation I could not repeat.

**3. Is the campaign correctly frozen and safe to begin VIA-000?**
**Correctly frozen: yes. Safe to begin: no, not yet.** The freeze is sound in every respect I
tested — consistent candidate/tree bindings across all seven packets, unchanged rule hashes across
the activation commits, correct manifest and requirements bindings against Git bytes,
`holdout_started: false`, decision `pending`, dependents `drafted`, and a validator that rejected
five of six tampering mutations. The blocker is external: the frozen candidate is not published,
so the runner cannot obtain the exact-SHA clone the protocol requires (CAMP-001).

**4. Has VIA-300 passed?**
**No.** VIA-300 is `lifecycle_phase: drafted`, `packet_outcome: pending`, `round_status: not-run`,
`achieved_evidence: E0-proposal`, with no receipts. It requires `E4-convergent-replication` and
four capabilities — entangling-observables, exact-overlap, decreasing-error, four-level-ladder —
none of which the native mean-field kernel can produce. The packet's own
`known_failure_to_retain` states that the current mean-field CUDA backend cannot compute MI/QCMI,
and its `null_or_competitors` names "the backend substitutes product proxies" as a competitor to
be excluded.

**5. Has Tier R mechanism viability been established?**
**No.** Every Tier-R packet is unadjudicated and the campaign decision is `pending`. Compilation,
throughput, finite output, per-cell norm preservation and native unit-test success are not
evidence of mechanism viability, and this candidate produces no other kind.

**6. What evidence remains necessary?**
Publish the frozen candidate and obtain a green exact-SHA CI run (CAMP-001); close the four
blocking findings; then execute VIA-000 under its frozen protocol with independent custodian,
falsifier, runner, statistical-auditor, claim-auditor and adjudicator seats, with output committed
before reveal. Beyond that, VIA-300 needs a genuinely **entangling** scalable backend — one that
computes reduced density matrices, mutual information and QCMI — validated against exact
observables with decreasing error across four refinement levels and at least two independent
implementations. None of that exists in this candidate.

## 7. Limitations and exposure boundary

- **No exposure-boundary violation of the access restrictions.** `C:\src\POPGP-custody`, sealed
  holdouts, secret seeds, private evaluator logic, credentials and
  `C:\src\POPGP\POPGP_Codex_Handoff.md` were not opened. The holdout was not started, no custody
  material was revealed, and no packet lifecycle state was advanced.
- **One disclosed exposure.** I read both calibration memos, which
  `launch/VIA-000_EXECUTION_BRIEF.md` reserves from a clean reviewer until it commits its initial
  result. My task directed this, and I recorded hypotheses first; I am not the `codex-via000-claims`
  seat and this artifact is not a VIA-000 receipt. A future VIA-000 claim auditor must be a
  different session that has not read this artifact or those memos.
- **One untracked artifact inspected, not used as proof.** `C:\src\POPGP\popgp\_lib\phase_flow.dll`
  (gitignored, not in the frozen tree) was parsed and load-tested to probe the builder's
  architecture claim. No conclusion depends on it.
- **Native build not executed.** No CUDA toolkit exists here, so `build.bat --clean --test
  --cuda-arch 120` could not be run end to end. BUILD-001 was demonstrated on an equivalent
  minimal CMake/CTest project rather than on the CUDA build; BUILD-003's architecture-verification
  gap and BENCH-001's mechanism were established by source inspection plus executed argument
  probes.
- **Windows only.** Every command ran on Windows 11 / CPython 3.11 / torch 2.10.0+cpu. The Linux
  half of VIA-000's cross-platform requirement is untested by me.
- **Builder model identity unknowable.** All packet seats record `model_identity: unknown`, so
  `reviewer_model_differs_from_builder` is recorded `false` — meaning "not established", not
  "same model".
