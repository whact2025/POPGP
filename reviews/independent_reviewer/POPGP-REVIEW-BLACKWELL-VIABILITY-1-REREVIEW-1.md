# Independent re-review: POPGP-REVIEW-BLACKWELL-VIABILITY-1-REREVIEW-1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-BLACKWELL-VIABILITY-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "claude-opus-5"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "d600ea7c-809e-41df-bba4-4ecbdb4d7734"
reviewer_orchestrator_id: "claude-code-standalone"
review_date: "2026-08-18"
commit_reviewed: "e04ac0522614d5bedca70aa4f58e3efae46d828c"
baseline_commit: "f22694b7427e82533c844f897b9f098558226ea1"
prior_review_ref: "d14c95db2a5500e27c0ff2fa80bb81f5c4c65004:reviews/independent_reviewer/POPGP-REVIEW-BLACKWELL-VIABILITY-1.md"
builder_response_ref: "e04ac0522614d5bedca70aa4f58e3efae46d828c:reviews/codex/POPGP-REVIEW-BLACKWELL-VIABILITY-1-RESPONSE-1.md"
context_hash: "521b3efe2a87358a64b11ec9d3723213d0b0846b"
context_hash_method: "git rev-parse \"e04ac0522614d5bedca70aa4f58e3efae46d828c^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/templates/INDEPENDENT_REREVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "popgp/backend.py"
  - "popgp/engine.py"
  - "popgp_engine/CMakeLists.txt"
  - "popgp_engine/build.bat"
  - "popgp_engine/build.sh"
  - "popgp_engine/cmake/ValidateCudaArchitecture.cmake"
  - "popgp_engine/cmake/VerifyCTestCount.cmake"
  - "popgp_engine/cmake/VerifyCudaBinary.cmake"
  - "popgp_engine/kernel/CMakeLists.txt"
  - "popgp_engine/kernel/README.md"
  - "popgp_engine/kernel/include/types.cuh"
  - "popgp_engine/kernel/src/area_law.cu"
  - "popgp_engine/kernel/src/clock.cu"
  - "popgp_engine/kernel/src/main.cpp"
  - "popgp_engine/kernel/src/phase_flow.cu"
  - "popgp_engine/kernel/tests/test_area_law.cu"
  - "popgp_engine/kernel/tests/test_clock.cu"
  - "popgp_engine/kernel/tests/test_phase_flow.cu"
  - "popgp_engine/vcpkg.json"
  - "protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-BLACKWELL-VIABILITY-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-BLACKWELL-VIABILITY-1.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-010.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-100.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-150.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-200.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-300.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-400.yaml"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_engine.py"
  - "tests/unit/test_native_build_contract.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "git diff dfe222ce3a173f617471d1979f0021ae6ec23ebf..e04ac0522614d5bedca70aa4f58e3efae46d828c (complete, all 36 files)"
  - "git archive e04ac05 (disposable copies used for all eight mutation probes)"
  - "git clone https://github.com/whact2025/POPGP.git (fresh public network clone, seven exact-SHA checkouts)"
access_level: local-repository-public-tree-plus-public-unauthenticated-github-read-only-plus-local-cuda-toolkit
independence_statement: |-
  Process separation only, and this round is WEAKER on one axis than the initial review. The
  weaknesses first, because they qualify everything below.

  1. SAME REVIEWER SEAT AND SESSION AS THE INITIAL REVIEW. `reviewer_session_id`
     d600ea7c-809e-41df-bba4-4ecbdb4d7734 is the identical session that produced
     d14c95db…:reviews/independent_reviewer/POPGP-REVIEW-BLACKWELL-VIABILITY-1.md. I am verifying
     my own findings. A systematic error in the initial review — a wrong threat model, a missed
     failure mode, an overstated blocker — is invisible to this process. That is a real limit and
     it is why the outcomes below rest on freshly executed counterexamples rather than on
     agreement with what I wrote before.
  2. I ALREADY CORRECTED MYSELF ONCE. The initial review's BUILD-002 asserted that the extracted
     CUDA toolchain "no longer exists". That was wrong: I had searched Program Files, PATH and
     CUDA_PATH but not C:\src\POPGP-cuda-toolkit-13.3\local. The builder disputes that clause and
     the dispute is correct. I re-derived the toolkit state from the machine this round rather
     than from either the initial review or the response.
  3. MODEL. The reviewing seat is claude-opus-5 per this runtime. The builder model identity is
     recorded `unknown` in the response and in every packet seat, so I set
     `reviewer_model_differs_from_builder: false` — meaning "not established", not "same model".
     I will not assert an independence fact I cannot verify.
  4. OPERATOR. Shared. `fuocor` operates the builder and this review, and has operated every seat
     in this campaign. There is no operator separation anywhere.
  5. ORCHESTRATOR. No orchestrator relays between the roles; I ran standalone. The builder
     declares `codex-desktop`, which I read but could not verify.
  6. ORDERING DISCIPLINE. I inspected the complete dfe222c..e04ac05 diff and recorded sixteen
     hypotheses (H1-H16) with expected failure modes in private working notes outside the
     repository BEFORE opening the builder response or the remediation calibration memo, then
     executed the counterexamples, and only then read both. Every builder claim below was
     re-derived independently; several of my hypotheses probed cases the builder's own tests do
     not cover, and those are reported explicitly.
  7. WHAT IS ACTUALLY EVIDENCE. A clean native Blackwell build from the frozen source; authoritative
     cuobjdump verification; a rejected wrong-architecture build; two deterministic benchmark runs;
     eight mutation probes that each turn a green test red; seven exact-SHA checkouts from a fresh
     public network clone; three verified CI runs; and the full twelve-command quality suite. That
     is evidence about specific software propositions. It is not evidence about physics.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: false
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r1-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  All nine prior findings are verified-resolved and all nine prior requested tests are
  verified-satisfied. No new blocking finding is supported. Recommendation: approve.

  The four blockers are closed by mechanism, not by assertion, and I broke each one before
  accepting it. BUILD-001: `ctest --no-tests=error` is now in both entry points and a
  configure-time expected-count contract is compared against `ctest -N`; I drove that verifier
  with counts of 1, 2, 3, 0, non-numeric and a missing file, and only the exact match passes —
  so both partial AND excess discovery are rejected, the excess case being one the builder's own
  test does not cover. The real build enumerated exactly seven cases and all seven passed.
  BUILD-002: CMake now requires CUDA 12.8 twice over — a `find_package(CUDAToolkit 12.8 REQUIRED)`
  constraint and an explicit `VERSION_LESS` fatal check — placed before `enable_language(CUDA)`;
  a stubbed version probe rejects 12.0 and 12.7 and accepts 12.8 and 13.3.73. The exact installer
  URL and SHA-256 are now in the kernel README with CUDA_PATH instructions.
  BENCH-001: the benchmark performs an untimed warmup with a state reset, checks
  `cudaGetLastError()` after every launch, synchronizes, validates the timing, reads both spinor
  components back, rejects nonfinite values and per-cell norm error above 1e-12, prints an
  FNV-1a-64 checksum, and is registered as a CTest case. Two independent runs produced identical
  checksum 48e6ef8f40cb137c and identical norm error 1.1435297153639112e-14 — matching the
  builder's reported values exactly.
  CAMP-001: a fresh public network clone checked out all seven declared SHAs, and the three
  claimed Actions runs verify with the correct head_sha and success conclusion — none of them the
  superseded 829b866.

  The strongest single result is the negative control. Requesting `--cuda-arch 86` on this
  sm_120-only device fails at `VerifyCudaBinary.cmake:97` with "Requested sm_86 does not match a
  visible GPU architecture (120)", the link step fails, and because the verifier runs POST_BUILD
  *before* the deployment copy, no wrong-architecture binary ever reaches `popgp/_lib`. I
  confirmed the positive control authoritatively: a clean build from the frozen source produced
  three `sm_120` cubins under `cuobjdump --list-elf`, with SASS for all four kernels — not a
  string scan, which is how I had to classify this last round.

  Scope is unchanged and honestly represented. `phase_flow.cu` and `popgp/backend.py` are
  byte-unchanged by the remediation, so the backend remains a mean-field product state that cannot
  compute MI or QCMI. The clock solver now returns `POPGP_STATUS_NOT_IMPLEMENTED` and provably
  does not touch its output; pruning is active and tested on both transitions; every device
  `printf` is gone. The only added lines mentioning MI/QCMI or entanglement are limitation
  statements and an unchecked TODO. VIA-300 remains `drafted` with `packet_outcome: pending` and
  `achieved_evidence: E0-proposal`, the campaign decision is `pending`, and `holdout_started` is
  false in all seven packets.

  Approval covers merge-readiness of software and campaign machinery under the declared contract.
  It does not establish VIA-300, Tier R, an entangling backend, or external scientific validation.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "BUILD-001"
    outcome: verified-resolved
    evidence: |-
      Both entry points now carry the guard: `popgp_engine/build.bat:141` and
      `popgp_engine/build.sh:65` invoke
      `ctest --test-dir build -C <cfg> --output-on-failure --no-tests=error`, preceded by
      `cmake -DPOPGP_BUILD_DIR=... -DPOPGP_CONFIG=... -P cmake/VerifyCTestCount.cmake`.
      I drove VerifyCTestCount.cmake directly against a scratch project registering exactly 2
      tests, rewriting the expected-count file each time:
        expected=1 (partial)     -> rc=1 "Expected 1 native tests, but CTest discovered 2."
        expected=2 (exact)       -> rc=0 "Verified 2 native tests are registered"
        expected=3 (EXCESS)      -> rc=1 "Expected 3 native tests, but CTest discovered 2."
        expected=0               -> rc=1 "Invalid expected native-test count '0'"
        expected=abc             -> rc=1 "Invalid expected native-test count 'abc'"
        file removed             -> rc=1 "Expected native-test count file is missing"
      The excess case is not covered by the builder's own test; it fails closed because the
      comparison is exact string equality at VerifyCTestCount.cmake:30.
      Original zero-test counterexample reproduced and now fails: `ctest --test-dir <empty-build>
      --output-on-failure --no-tests=error` returns nonzero with "No tests were found" (the
      initial review measured exit 0 without the flag).
      The real build requires exactly seven: `popgp_engine/kernel/CMakeLists.txt:85` sets
      `POPGP_EXPECTED_NATIVE_TESTS 7` via `file(GENERATE)`; I counted six `TEST(` macros across
      the three .cu files plus `add_test(NAME benchmark_validation COMMAND popgp_sim)` at :61.
      `build.bat --clean --test --cuda-arch 120` printed "Verified 7 native tests are registered"
      then "100% tests passed, 0 tests failed out of 7", EXIT=0.
      Bypass is persisted-test-guarded: removing ` --no-tests=error` from build.bat, or renaming
      `VerifyCTestCount.cmake` in build.sh, each turns
      tests/unit/test_native_build_contract.py::test_cuda_architecture_argument_contract red
      ("1 failed, 2 passed" in both cases).
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Residual, non-blocking and not carried as a finding: the expected-count file lives in the
      build directory, so a hand-edited stale build tree could satisfy the check. `file(GENERATE)`
      rewrites it on every configure and build.bat always reconfigures, so no realistic workflow
      reaches it.
  - finding_id: "BUILD-002"
    outcome: verified-resolved
    evidence: |-
      FIRST, THE CORRECTION I OWE. The initial review's clause "the toolchain … no longer exists"
      was FALSE and the builder's `partially-accepted` disposition is right. Re-derived from the
      machine this round: `C:\src\POPGP-cuda-toolkit-13.3\local\bin` contains `nvcc.exe`,
      `cuobjdump.exe` and `nvdisasm.exe`, with `include/` and `lib/`; `nvcc --version` reports
      "Cuda compilation tools, release 13.3, V13.3.73". My initial search covered Program Files,
      PATH and CUDA_PATH but not that path, and I asserted a negative I had not established. That
      clause is withdrawn.
      THE SURVIVING HALF IS REMEDIATED. `popgp_engine/CMakeLists.txt:15-20` now enforces the
      minimum twice and before `enable_language(CUDA)` at :26:
        find_package(CUDAToolkit 12.8 REQUIRED)
        if(CUDAToolkit_VERSION VERSION_LESS 12.8) message(FATAL_ERROR "POPGP requires CUDA Toolkit
        12.8 or newer; found ${CUDAToolkit_VERSION}.")
      Stubbed configure-time probe of that exact guard (installing a 12.x toolkit is unreasonable;
      the request permitted a stub): 12.0 -> rc=1 "requires CUDA Toolkit 12.8 or newer; found
      12.0."; 12.7 -> rc=1; 12.8 -> rc=0; 13.3.73 -> rc=0. The `find_package` version constraint
      is an independent second mechanism enforced by CMake itself.
      PROVENANCE IS NOW DOCUMENTED IN THE REPOSITORY, not only in a memo:
      `popgp_engine/kernel/README.md:40-69` states the 12.8 minimum and its rationale, the exact
      installer
      https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/cuda_13.3.1_windows.exe,
      SHA-256 d68839fcce644576f0a1c6b066e0c5bc146a62db2fcac9fc6b4e6418e7ec533f, a download script,
      and CUDA_PATH/PATH setup including the `bin\x64` layout.
      END-TO-END POSITIVE CONTROL: with CUDA_PATH set to that extraction,
      `build.bat --clean --test --cuda-arch 120` completed EXIT=0.
      The CPU-only lock file is a separate matter and remains correct: `torch 2.10.0+cpu`,
      `torch.cuda.is_available() False`. Toolkit provenance is a build-time property; the lock
      file governs the Python runtime and is deliberately unchanged.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      LIMITATION, recorded as a limitation and not as a passed check: I did not download the
      multi-gigabyte installer, so the published SHA-256 is documented and internally consistent
      with the installed nvcc 13.3.73 but is NOT verified against NVIDIA by me. Residual risk the
      builder states and I agree with: the repository pins provenance but cannot redistribute the
      toolkit, so provisioning still depends on the external official installer.
  - finding_id: "BENCH-001"
    outcome: verified-resolved
    evidence: |-
      Rebuilt from the frozen source; no pre-existing DLL was used as evidence. Read verbatim at
      `popgp_engine/kernel/src/main.cpp`: untimed warmup of both coloured paths with
      `CUDA_CHECK(cudaGetLastError())` and `cudaDeviceSynchronize()` (:121-129), a state reset so
      the checksum represents exactly `steps` steps (:133-140), `CUDA_CHECK(cudaGetLastError())`
      after every launch inside the timed loop (:150,:154), `cudaDeviceSynchronize()` after the
      loop (:158), a finite/positive timing check (:162-165), readback of both arrays (:167-174),
      per-element finiteness and per-cell norm accumulation (:179-190), rejection at
      `max_norm_error > 1.0e-12` with `return EXIT_FAILURE` (:196-204), and an FNV-1a-64 checksum.
      Registered as a CTest case: `add_test(NAME benchmark_validation COMMAND popgp_sim)` at
      kernel/CMakeLists.txt:61; it ran as case 7/7 in my build.
      Two independent executions of the freshly built binary:
        run 1: EXIT=0, 167.25 ms, 5.98e+08 edge updates/s,
               checksum 48e6ef8f40cb137c, max norm error 1.1435297153639112e-14
        run 2: EXIT=0, 169.59 ms, 5.90e+08 edge updates/s,
               checksum 48e6ef8f40cb137c, max norm error 1.1435297153639112e-14
      Checksum and norm error are bit-identical across runs and match the builder's recorded
      values exactly; only wall time varies, which is the correct design. Norm error is 1.14e-14
      against the 1e-12 rejection threshold.
      WRONG-ARCHITECTURE REJECTION, executed: `build.bat --clean --test --cuda-arch 86` on this
      sm_120-only device gives EXIT=1, "FAILED: [code=1] kernel/phase_flow.dll" and
      "CMake Error at .../VerifyCudaBinary.cmake:97 … Requested sm_86 does not match a visible GPU
      architecture (120)". The link fails, so no no-kernel-image binary is ever produced for the
      benchmark to run; and had one been produced, the warmup's `cudaGetLastError()` would abort
      via CUDA_CHECK before any timing is printed. I could not make an invalid binary print
      successful throughput.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Cosmetic inaccuracy in the response, not a defect in the candidate: it records the binary as
      `popgp_engine\build\kernel\Release\popgp_sim.exe --verbose`. Under the Ninja single-config
      generator the path is `build/kernel/popgp_sim.exe`, and `main()` takes no arguments, so
      `--verbose` is inert. The substantive values reproduce exactly.
  - finding_id: "CAMP-001"
    outcome: verified-resolved
    evidence: |-
      Fresh public network clone (`git clone https://github.com/whact2025/POPGP.git`, exit 0),
      then `git checkout --detach <full SHA>` for each declared commit — all seven succeeded:
        9a29e05f803666bf0e3a28417ea399e3e26769fc -> tree 358fb1af6ca587b6c71ff2ef0fb87e335163eeaf
        e04ac0522614d5bedca70aa4f58e3efae46d828c -> tree 521b3efe2a87358a64b11ec9d3723213d0b0846b
        3428a24bdd05b5742888bd204482d50a2f613cab -> tree 42fcba754ed78145054f572bd0beedf8a157562a
        8c3c4d53cddf28b0c51dc7b6c21eab49d28d0f9d -> tree ea51ff8312474ceda743cf7a6a576fb6e2253b91
        99f13dcc04d556892495e8e2361d257b1b2c113b -> tree 6f32893a1a3c17d3ec1fab56c7fcd715b05ed5c3
        792d2d2737dcf48cf14cbb3af78a64ba2daad1f6 -> tree cdb848d219c84ebe0e76e75d52b39d309951dee2
        197219bbb006b4cd1f8b9f992cadedae8d09a341 -> tree 96b5d54277bf2ffa82cc297a012870f40c9eb932
      The three commits the initial review specifically demanded (3428a24, 8c3c4d5, 99f13dc) are
      among them. The final scientific-candidate tree matches the declared
      358fb1af6ca587b6c71ff2ef0fb87e335163eeaf exactly. Unauthenticated
      `GET /repos/whact2025/POPGP/commits/<sha>` returned HTTP 200 for all seven plus dfe222c.
      Exact-SHA Actions, read unauthenticated:
        run 32088634733 head_sha 9a29e05f… status completed conclusion success event push
        run 32089229292 head_sha 197219bb… status completed conclusion success event push
        run 32089515529 head_sha e04ac052… status completed conclusion success event push
      Each head_sha matches its claimed commit; none is the superseded 829b866.
      Campaign state re-derived from the handoff tree: CAMPAIGN.yaml candidate_commit
      9a29e05f…, tree_hash 358fb1af…, protocol_commit 792d2d27…, decision `outcome: pending`
      with `authorized_by: null`; all seven packets bind the same candidate/tree/protocol; VIA-000
      `preregistered`, the other six `drafted`; `holdout_started: false` in all seven.
      `uv run python scripts/check_viability_campaign.py …/CAMPAIGN.yaml` -> exit 0,
      "Viability campaign contract is valid." Custody was never opened and no lifecycle advanced.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The initial review's CAMP-001 was the one blocker with a purely external remedy, and it is
      fully discharged: what was previously HTTP 422 with zero workflow runs is now HTTP 200 with
      three verified green exact-SHA runs.
  - finding_id: "ENG-001"
    outcome: verified-resolved
    evidence: |-
      `popgp/engine.py:34-36` adds `_is_windows()` and :89 uses it for
      `load_kwargs = {"winmode": 0} if _is_windows() else {}`. The new parametrized test
      tests/unit/test_engine.py::test_load_library_uses_platform_specific_dll_search_semantics
      monkeypatches `engine.ctypes.CDLL`, calls the real `engine._load_library()`, and asserts the
      recorded call is `(str(library), {"winmode": 0})` on nt and `(str(library), {})` on posix —
      so it exercises the actual CDLL invocation, not a mock of the registration bookkeeping.
      Process-global `os.name` is no longer mutated: the old
      `monkeypatch.setattr(engine.os, "name", "nt")` is replaced by
      `monkeypatch.setattr(engine, "_is_windows", lambda: True)`, and a repository-wide grep for
      `setattr(engine.os, "name"` / `setattr(os, "name"` over tests/ returns nothing.
      Mutation controls executed on a disposable `git archive` copy (baseline: 4 passed):
        replace `load_kwargs = {...} if _is_windows() else {}` with `load_kwargs = {}`
          -> 1 failed, 3 passed
        revert handle retention to `os.add_dll_directory(str(d))` (discard return)
          -> 1 failed, 3 passed
      So the `winmode=0` gap the initial review reported — the one mutation that previously
      survived — is now closed, and the earlier retention guard still holds.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      LIMITATION: the locked environment is CPU-only (`torch 2.10.0+cpu`), so I could not
      independently reproduce the builder's 64-cell `GPUBackend` evolution through the DLL
      (63 edges, max norm error 5.551115123125783e-16). That specific claim is NOT independently
      reproduced by me. It is not load-bearing for any outcome here: the same kernel through the
      same DLL is exercised by the native benchmark and by six native tests, all of which I ran.
  - finding_id: "BUILD-003"
    outcome: verified-resolved
    evidence: |-
      Argument validation now runs before any environment setup, at build.bat:43 and build.sh:30,
      via `cmake -DPOPGP_CUDA_ARCHITECTURE=<v> -P cmake/ValidateCudaArchitecture.cmake`. Driving
      that script directly:
        accepted (rc=0): native, all, all-major, 120, 120-real, 120-virtual
        rejected (rc=1): "" (missing), --clean, -DFOO, not-an-arch, "120;86", sm_120, 12.0,
                         120-REAL, " 120"
      So missing, option-like, malformed, numeric, nn-real and nn-virtual inputs all behave as
      required, including the `--clean` swallow the initial review demonstrated.
      `cuobjdump` is mandatory: kernel/CMakeLists.txt:5-10 does
      `find_program(POPGP_CUOBJDUMP_EXECUTABLE NAMES cuobjdump HINTS "${CUDAToolkit_BIN_DIR}" REQUIRED)`,
      and VerifyCudaBinary.cmake:15 repeats `find_program(... REQUIRED)`.
      Every emitted ELF must match the requested numeric architecture — VerifyCudaBinary.cmake:87-94
      loops over all tokens and fails on the first mismatch. I probed the cases the builder's own
      test does not cover, all against the real script:
        empty cuobjdump output, arch 120   -> rc=1 at :85 (no ELF code found)
        mixed sm_120 + sm_86, arch 120     -> rc=1 at :89 (contains sm_86, not requested sm_120)
        120-virtual with ELF-only binary   -> rc=1 at :85 (no PTX)
        garbage output with no sm_ token   -> rc=1 at :85
        native with no visible GPU         -> rc=1 at :104
        sm_86 requested, visible 120       -> rc=1 at :97
        POSITIVE control sm_120            -> rc=0 "Verified … ELF=sm_120"
      LIVE NEGATIVE CONTROL on hardware: `build.bat --clean --test --cuda-arch 86` -> EXIT=1,
      link FAILED, "Requested sm_86 does not match a visible GPU architecture (120)". Because
      VerifyCudaBinary runs POST_BUILD ahead of the deployment copy, `popgp/_lib/phase_flow.dll`
      retained its previous verified sm_120 build and no mismatched binary was deployed.
      LIVE POSITIVE CONTROL: `build.bat --clean --test --cuda-arch 120` -> EXIT=0 with
      "Verified CUDA binary …/phase_flow.dll: ELF=sm_120; PTX=sm_120".
      AUTHORITATIVE BINARY CHECK on the freshly built DLL, not a string scan:
        cuobjdump --list-elf -> phase_flow.1.sm_120.cubin, .2.sm_120.cubin, .3.sm_120.cubin
        cuobjdump --list-ptx -> three sm_120 PTX entries
        cuobjdump -sass      -> arch = sm_120 with SASS for phase_flow_kernel_soa<double>,
                                phase_flow_kernel_soa<float>, calculate_cut_kernel,
                                prune_bulk_kernel
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Observation, non-blocking: VerifyCudaBinary accepts injected `POPGP_CUOBJDUMP_ELF_OUTPUT`/
      `_PTX_OUTPUT`, which is how the CPU-only Python test exercises it. The production POST_BUILD
      path does not inject them and cuobjdump is REQUIRED there, so the seam is a testability
      affordance rather than a bypass; the real path is what I exercised on hardware above.
  - finding_id: "SCOPE-001"
    outcome: verified-resolved
    evidence: |-
      CLOCK FAILS LOUDLY AND DOES NOT COPY. `popgp_engine/kernel/include/types.cuh:35-37` declares
      `constexpr int POPGP_STATUS_NOT_IMPLEMENTED = -1;` and changes the export to return `int`.
      `clock.cu:8-26` voids all seven parameters, writes
      "POPGP native clock solver is not implemented; output was not modified." to stderr, and
      returns the status. The `cudaMemcpy(phi, rho, …)` identity map is gone.
      Proven on hardware rather than by reading: native case 6/7
      `ClockTest.UnimplementedSolverFailsWithoutWritingOutput` seeds `phi = 123.5`, calls the
      solver, and asserts both `status == POPGP_STATUS_NOT_IMPLEMENTED` and that `phi` is
      unchanged — it passed in my build.
      PRUNING IS ACTIVE AND TESTED ON BOTH TRANSITIONS. `area_law.cu:46` restores
      `node_active_mask[idx] = 0;`, and native case 5/7 `AreaLawTest.PruningTransitionsAreApplied`
      exercises a retained node and a zeroed node; it passed. `AreaLawTest.BoundaryCutCalculation`
      still passes alongside it.
      NO DEVICE-SIDE OUTPUT IN A TIMED PATH. `grep -rn "printf" popgp_engine/kernel/src/*.cu`
      returns exactly one hit: the host-side `std::fprintf(stderr, …)` in clock.cu. Both device
      `printf` calls and the `cudaDeviceSynchronize()` that existed only to flush them are removed
      from area_law.cu.
      NOT REPRESENTED AS A WORKING CLOCK. `popgp_engine/kernel/README.md:17` lists
      "- [ ] Implement the native graph-Laplacian clock solver" as UNCHECKED, and the remediation
      calibration memo states "The clock solver now fails explicitly as not implemented; it is not
      a working clock reconstruction."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The exported ABI changed from `void` to `int`. I checked for breakage: no Python caller
      references `solve_clock_potential` (the ctypes surface in popgp/engine.py binds only
      `launch_phase_flow_float`/`_double`), and the only native caller is the new test. Nothing
      breaks.
  - finding_id: "DOC-001"
    outcome: verified-resolved
    evidence: |-
      `popgp_engine/kernel/README.md:15` now reads
      "- [x] Implement caller-side node-disjoint batching in `GPUBackend` and the benchmark",
      correctly attributing the property to the callers; :21 states that a caller must supply a
      node-disjoint batch and names `GPUBackend._edge_color_batches` as the supplier. The prior
      false claim that the kernel performs graph coloring is gone, and the checklist no longer
      marks the clock item complete (:17 is unchecked).
      `Engine.step`'s docstring (popgp/engine.py:154-159) now states that the edge arrays must
      describe a node-disjoint batch, that no source or destination index may occur in more than
      one edge in the call, and that the kernel "does not perform graph coloring".
      Pinned by a regression and verified as a genuine control on a disposable copy
      (baseline 4 passed):
        replace "node-disjoint batch" with "contiguous batch"          -> 1 failed, 3 passed
        replace "does not perform graph coloring" with
                "handles coloring internally"                          -> 1 failed, 3 passed
      Mean-field and clock limitations remain described accurately: README.md:78-95 still states
      the mean-field backend cannot compute mutual information and that the locality stage fails
      explicitly rather than substituting a false MI proxy.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: ""
  - finding_id: "CAMP-002"
    outcome: verified-resolved
    evidence: |-
      Enforcement was chosen over softening. `scripts/check_viability_campaign.py:2523-2528` adds,
      inside the `LIFECYCLE_ORDER[phase] >= LIFECYCLE_ORDER["preregistered"]` branch, a loop over
      `requirement["dependencies"]` that appends an error unless
      `packet_outcomes.get(dependency) == "passed"`.
      The exact original mutation now fails. On a disposable clone at the handoff, promoting
      VIA-300 from `drafted` to `preregistered` while VIA-000 is pending:
        exit 1, "packet VIA-300: lifecycle preregistered started before dependency VIA-000 passed"
      Same for VIA-010 (dependency VIA-000):
        exit 1, "packet VIA-010: lifecycle preregistered started before dependency VIA-000 passed"
      Baseline before and after each mutation: exit 0, "Viability campaign contract is valid."
      The gate is dependency-outcome driven, not phase-name driven, so a missing outcome, a
      blocked outcome and a failed outcome are all rejected by the same `!= "passed"` predicate;
      only a passing adjudicated dependency admits promotion. VIA-000 itself is unaffected because
      its declared dependency list is empty.
      Pinned by a persisted regression and verified as a genuine control: deleting the six-line
      gate from the validator turns
      tests/unit/test_viability_campaign_contract.py::test_dependent_packet_cannot_be_preregistered_before_dependencies_pass
      from "1 passed" to "1 failed", and restoring it returns "1 passed".
      Documentation is consistent rather than contradictory: the campaign README still states the
      dependents "stay in `drafted` lifecycle state until their machine-declared prerequisites
      pass", which is now true of the implementation.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: ""

prior_requested_test_results:
  - requested_test_id: "TST-BUILD-001"
    outcome: verified-satisfied
    evidence: |-
      Delivered as `popgp_engine/cmake/VerifyCTestCount.cmake` plus
      tests/unit/test_native_build_contract.py::test_native_test_gate_rejects_zero_and_partial_discovery,
      with `--no-tests=error` in both entry points. The request's specific counterexample — a build
      tree with `enable_testing()` and no registered tests — is executed inside that test and
      returns nonzero with "No tests were found"; I reproduced it standalone as well. The expected
      native count is asserted at exactly seven and the real build reported
      "Verified 7 native tests are registered" followed by 7/7 passing.
      Beyond the request, I confirmed the excess-count case (expected 3, discovered 2) also fails,
      and that removing `--no-tests=error` or the count verifier from either script turns
      test_cuda_architecture_argument_contract red.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-BUILD-002"
    outcome: verified-satisfied
    evidence: |-
      The request asked for a configure-time precondition failing clearly on an insufficient
      toolkit, plus documented provenance. Both are delivered:
      `popgp_engine/CMakeLists.txt:15-20` carries `find_package(CUDAToolkit 12.8 REQUIRED)` and an
      explicit `VERSION_LESS` FATAL_ERROR before `enable_language(CUDA)`;
      `popgp_engine/kernel/README.md:40-69` records the 12.8 minimum with rationale, the exact
      13.3.1 installer URL, its SHA-256, a download script and CUDA_PATH setup.
      The builder demonstrated only the positive path, so I probed the negative path myself with
      the stub the request permits: the guard rejects 12.0 and 12.7 with
      "POPGP requires CUDA Toolkit 12.8 or newer; found <v>." and accepts 12.8 and 13.3.73.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      LIMITATION: the installer SHA-256 is documented but not verified by me against NVIDIA; I did
      not download the installer. No older toolkit was installed; the version gate was exercised
      by stub, which the request expressly allowed.
  - requested_test_id: "TST-BENCH-001"
    outcome: verified-satisfied
    evidence: |-
      Every element the request enumerated is present and was exercised on a binary I built from
      the frozen source: launch error checks after each launch, `cudaDeviceSynchronize()`, an
      untimed warmup outside the measured window, readback of both arrays, finiteness rejection,
      per-cell norm rejection above 1e-12, a deterministic FNV-1a-64 checksum, and `EXIT_FAILURE`
      on any failure. `add_test(NAME benchmark_validation COMMAND popgp_sim)` registers it as
      CTest case 7/7, which passed.
      Determinism control: two runs gave identical checksum 48e6ef8f40cb137c and identical
      max norm error 1.1435297153639112e-14.
      Failure control: the wrong-architecture build is rejected at link time, so the "successful
      throughput from a no-kernel-image binary" scenario the request named cannot occur; the
      warmup's `cudaGetLastError()` is a second barrier ahead of any timing output.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-CAMP-001"
    outcome: verified-satisfied
    evidence: |-
      The request asked that the candidate be publishable and fetchable by exact SHA with a green
      exact-SHA CI run before VIA-000 begins. A fresh public network clone checked out all seven
      declared SHAs including the three the initial review named, the final candidate tree matched
      358fb1af6ca587b6c71ff2ef0fb87e335163eeaf, and three exact-SHA Actions runs
      (32088634733 / 32089229292 / 32089515529) verify as completed+success against
      9a29e05 / 197219b / e04ac05 respectively. VIA-000 has not started: `holdout_started: false`
      in all seven packets and campaign `outcome: pending`.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      The request also contemplated a preflight check enforcing fetchability. That was not added
      as executable code; the property is instead demonstrated by published commits and verified
      CI. I treat the request as satisfied on its stated purpose — the runner can now obtain the
      exact SHA — and note the absence of an automated preflight as a residual, not a shortfall.
  - requested_test_id: "TST-ENG-001"
    outcome: verified-satisfied
    evidence: |-
      Delivered exactly as requested:
      tests/unit/test_engine.py::test_load_library_uses_platform_specific_dll_search_semantics
      monkeypatches `engine.ctypes.CDLL`, invokes the real `_load_library()`, and asserts
      `winmode=0` on nt and no keyword on posix. The request's specific mutation — replacing
      `load_kwargs = {"winmode": 0} if _is_windows() else {}` with `load_kwargs = {}` — turns the
      suite red (1 failed, 3 passed), where it previously passed. The request's additional
      constraint that the test must not mutate process-global `os.name` is met via the
      `_is_windows()` seam; a grep over tests/ finds no `os.name` monkeypatch.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-BUILD-003"
    outcome: verified-satisfied
    evidence: |-
      Both halves delivered. Argument validation: `ValidateCudaArchitecture.cmake` accepts
      native/all/all-major/nn/nn-real/nn-virtual and rejects empty, `--`-prefixed, malformed,
      list-valued, `sm_`-prefixed, dotted and case-variant values — all nine rejections and six
      acceptances executed. Binary verification: `VerifyCudaBinary.cmake` requires cuobjdump
      (`find_program … REQUIRED`), checks every listed ELF against the requested numeric
      architecture, and additionally requires a visible-device match when
      `POPGP_REQUIRE_VISIBLE_CUDA_ARCH=ON`, which both entry points now pass.
      The request's demonstrations are all executed: `--cuda-arch 86` on the sm_120 device fails
      (EXIT=1, VerifyCudaBinary.cmake:97), and the sm_120 positive control passes with
      cuobjdump reporting three sm_120 cubins.
      Beyond the request I probed empty, malformed, mixed-architecture and virtual-without-PTX
      cuobjdump output; every one fails closed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-SCOPE-001"
    outcome: verified-satisfied
    evidence: |-
      All three requested behaviours are implemented and covered by native tests that I ran on
      hardware: `ClockTest.UnimplementedSolverFailsWithoutWritingOutput` asserts the explicit
      not-implemented status and that the output buffer is untouched;
      `AreaLawTest.PruningTransitionsAreApplied` exercises the restored prune branch on both a
      retained and a zeroed node; and the device `printf` calls plus their flush synchronization
      are removed, leaving zero device-side output in any timed path (verified by grep over all
      `.cu` sources — the sole remaining `fprintf` is host-side in clock.cu).
      Both new cases are inside the exact seven-case gate, so they cannot be silently dropped.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-DOC-001"
    outcome: verified-satisfied
    evidence: |-
      The kernel README now attributes node-disjoint batching to the callers and leaves the clock
      item unchecked; `Engine.step` documents the node-disjoint precondition and the absence of
      internal coloring; and
      tests/unit/test_engine.py::test_engine_step_documents_node_disjoint_edge_precondition pins
      both statements. The request's mutation demonstration is satisfied: altering either phrase in
      a disposable copy turns that test red (1 failed, 3 passed in each case).
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-CAMP-002"
    outcome: verified-satisfied
    evidence: |-
      The request offered enforcement or documented softening; enforcement was chosen and is
      executable. The validator rejects a dependent packet at `preregistered` or beyond unless
      every declared dependency has a `passed` adjudicated outcome, and
      tests/unit/test_viability_campaign_contract.py::test_dependent_packet_cannot_be_preregistered_before_dependencies_pass
      persists the exact mutation the request named. I reproduced the mutation against the live
      campaign (VIA-300 and VIA-010 both rejected with precise messages), confirmed the positive
      baseline still validates, and confirmed the test is a genuine control by deleting the gate
      from the validator and observing the test fail.
      Because the predicate is `packet_outcomes.get(dependency) != "passed"`, missing, blocked,
      failed and contradictory dependency outcomes are all rejected by construction; only a
      passing adjudicated dependency admits promotion.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""

predictions:
  experiment_id: "BUILD-003-wrong-architecture-negative-control"
  predicted_outcome: |-
    On any host whose only visible GPU reports compute capability 12.0, running
    `popgp_engine\build.bat --clean --test --cuda-arch 86` at
    e04ac0522614d5bedca70aa4f58e3efae46d828c will exit nonzero during the phase_flow link step
    with "Requested sm_86 does not match a visible GPU architecture (120)" raised from
    VerifyCudaBinary.cmake:97, and `popgp/_lib/phase_flow.dll` will not be replaced. The same
    command with `--cuda-arch 120` will exit 0, print
    "Verified CUDA binary …: ELF=sm_120; PTX=sm_120" and "Verified 7 native tests are registered",
    and pass 7/7 CTest cases, with `cuobjdump --list-elf` reporting exactly three sm_120 cubins
    and `popgp_sim` emitting checksum 48e6ef8f40cb137c with maximum norm error
    1.1435297153639112e-14.
  predicted_failure_mode: |-
    The prediction is falsified if the sm_86 build completes successfully on an sm_120-only
    device. The most likely benign cause of an apparent refutation is configuration rather than a
    defect: `POPGP_REQUIRE_VISIBLE_CUDA_ARCH` defaults ON but is a cache option, so a build
    directory previously configured with it OFF, or a direct `cmake` invocation bypassing
    build.bat, disables the visible-device equality check while leaving the ELF-token check
    intact. A replication must therefore assert the option's value in the CMake cache before
    concluding. A genuine refutation would require `cuobjdump --list-elf` to report an sm_86 cubin
    while VerifyCudaBinary reported success. The checksum prediction is falsifiable by any change
    to the kernel, the seed, the step count or the coloring, and is expected to be stable only at
    this exact tree.
  confidence_statement: |-
    High confidence in all nine finding outcomes and all nine test outcomes. Each rests on a
    command I ran at the frozen tree with its exact output recorded, and each claimed fix was
    attacked before being accepted: eight mutation probes each turn a green test red, and the
    architecture verifier was driven through seven edge cases including four the builder's own
    tests do not cover. High confidence in the sm_120 result specifically, because it comes from
    `cuobjdump` on a binary I built from the frozen source rather than from a string scan or a
    pre-existing artifact. High confidence in the campaign state, re-derived from the handoff tree
    and a fresh public clone rather than from the response.
    Moderate confidence on cross-platform behaviour: every native command ran on Windows 11 with
    MSVC/Ninja and CUDA 13.3.73. I did not execute `build.sh`, so its POSIX path is verified by
    text and by shared CMake modules rather than by execution; the exact-SHA Linux CI run covers
    the Python suite, not the native build.
    Explicitly not established, and recorded as limitations rather than passes: the installer
    SHA-256 was not verified against NVIDIA; no toolkit older than 12.8 was installed, so the
    version gate was exercised by stub; and the CPU-only lock file prevented me from reproducing
    the builder's 64-cell GPUBackend integration numbers.
    Low confidence that this round would detect an error shared with the initial review, because
    the same reviewer seat and session produced both.

recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    blocking_findings = 0 unresolved prior blockers + 0 supported new blockers = 0, which
    reconciles exactly to approve: true.

    All four prior blockers are independently verified resolved by mechanism. BUILD-001 is closed
    by `--no-tests=error` plus an exact expected-count contract that I drove through partial,
    exact, excess, zero, non-numeric and missing-file cases. BUILD-002 is closed by a doubled
    CUDA 12.8 precondition placed ahead of `enable_language(CUDA)`, whose negative path I
    exercised by stub, plus installer URL and SHA-256 now recorded in the repository rather than
    only in a memo — and I withdraw the initial review's factually wrong "toolchain no longer
    exists" clause, which the builder correctly disputed. BENCH-001 is closed by a benchmark that
    checks every launch, validates read-back state and norms, warms up outside the measured
    window, emits a deterministic checksum reproduced bit-identically across two runs, and is
    itself a CTest case. CAMP-001 is closed by publication: seven exact SHAs fetched from a fresh
    public clone and three green exact-SHA CI runs at the right head_sha.
    The five non-blocking findings are equally closed, each pinned by a persisted regression that
    I confirmed is a genuine control by mutating the implementation and observing the test fail.

    No new blocking finding is supported. I probed several residuals — the expected-count file
    living in the build directory, the cuobjdump output-injection seam used by the CPU-only test,
    and a cosmetic wrong path in the response's recorded benchmark command — and none is a defect
    in the candidate: each is either unreachable in a realistic workflow, a testability affordance
    whose production path is enforced by `find_program(... REQUIRED)`, or an inaccuracy in a
    response artifact rather than in the code. Filing them as findings would be noise, so they are
    recorded as notes against the relevant outcomes instead. Because approval is granted, no
    builder action is requested and `requested_tests` is empty.

    The scientific boundary is unchanged and is stated so it cannot be misread. `phase_flow.cu`
    and `popgp/backend.py` are byte-unchanged by this remediation: the native backend remains a
    mean-field product-state path that cannot compute mutual information or QCMI. The clock solver
    now fails explicitly instead of silently returning its input, which is honesty about an absent
    capability, not the arrival of one. VIA-300 remains `drafted` with `packet_outcome: pending`
    and `achieved_evidence: E0-proposal`; the campaign decision is `pending`; `holdout_started` is
    false. Approval here means the reviewed change is merge-ready under the declared contract and
    that VIA-000 is now safe to begin. It does not establish VIA-300, Tier R, an entangling
    backend, independent implementation, or external scientific validation, and it must not be
    reported as approval of POPGP's physical theory.
```

## 1. Method and governance verification

| Check | Result |
|---|---|
| Branch | `review/blackwell-viability-rereview-1` |
| HEAD | `e04ac0522614d5bedca70aa4f58e3efae46d828c` |
| Tree | `521b3efe2a87358a64b11ec9d3723213d0b0846b` — matches declared |
| Final scientific candidate tree | `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf` — matches declared |
| Initial review commit is ancestor | yes (`d14c95db…`) |
| Original candidate is ancestor | yes (`dfe222ce…`) |
| Builder response committed | yes, at the handoff |
| Initial review byte-unchanged | yes — blob `1b0f7a60d3170d61033089f797dfd8cfabb4c2f8` identical at `d14c95d` and `e04ac05` |
| Worktree clean before and after | yes |

Ordering was observed: sixteen hypotheses were recorded in private notes outside the repository
from the diff alone, the counterexamples were executed, and only then were the builder response and
the remediation calibration memo opened.

## 2. Exact command results

| Command | Exit | Result |
|---|---:|---|
| `uv sync --frozen` | 0 | locked environment resolved |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | — |
| `uv run pytest -q` | 0 | **187 passed in 887.81s** |
| six `python -m examples.physics_qg.*` | 0 ×6 | chain_1d, grid_2d, gravity_well, source_law, source_law_many_body, ca_model |
| `uv run python scripts/check_validation_artifacts.py` | 0 | contracts and required visuals valid |
| `uv run python scripts/check_viability_campaign.py …/CAMPAIGN.yaml` | 0 | `Viability campaign contract is valid.` |
| `git diff --check` | 0 | — |
| `git status --porcelain` | 0 | empty |
| `build.bat --clean --test --cuda-arch 120` | **0** | `Verified CUDA binary …: ELF=sm_120; PTX=sm_120`; `Verified 7 native tests are registered`; **7/7 passed** |
| `build.bat --clean --test --cuda-arch 86` | **1** | link FAILED — `Requested sm_86 does not match a visible GPU architecture (120)` |
| `cuobjdump --list-elf phase_flow.dll` | 0 | three `sm_120` cubins |
| `popgp_sim.exe` ×2 | 0, 0 | checksum `48e6ef8f40cb137c` both runs; norm error `1.1435297153639112e-14` both runs |

Mutation controls, each on a disposable `git archive` copy — all eight turn a green test red:
`winmode=0` removal, DLL-handle-retention revert, both `Engine.step` docstring phrases,
`--no-tests=error` removal, `VerifyCTestCount.cmake` removal, `ValidateCudaArchitecture.cmake`
removal, and deletion of the validator dependency gate.

## 3. Scientific boundary — the six questions

1. **Is the Blackwell build reproducible and merge-ready?** Yes, under the declared contract. A
clean build from the frozen source succeeded with a version-gated toolkit, validated architecture
argument, post-link binary verification and a non-vacuous seven-case test gate. Provisioning still
depends on NVIDIA's external installer, which is now pinned by URL and SHA-256.
2. **Is executable native `sm_120` feasibility independently reproduced?** Yes. I built the DLL
myself from the frozen tree and `cuobjdump --list-elf` reports three `sm_120` cubins with SASS for
all four kernels. This supersedes the initial review's *not confirmed* classification, which was
made without the tool.
3. **Is VIA-000 safe to begin?** Yes, on the criteria in scope here. The candidate is publicly
fetchable by exact SHA, three exact-SHA CI runs are green, the campaign validates, custody is
untouched and `holdout_started` is false. Starting it remains a maintainer decision requiring the
independent seats named in the execution brief.
4. **Has VIA-300 passed?** No. `lifecycle_phase: drafted`, `packet_outcome: pending`,
`round_status: not-run`, `achieved_evidence: E0-proposal`, no decisive receipt. It requires
E4-convergent-replication and an entangling backend.
5. **Has Tier R been established?** No. Every Tier-R packet is unadjudicated and the campaign
decision is `pending`.
6. **What scientific evidence remains absent?** Everything that distinguishes viability from
feasibility: an entangling state representation capable of MI and QCMI, exact-observable agreement
with decreasing error across four refinement levels, at least two independent implementations,
unaffiliated reproduction, and any external empirical validation. The backend is still a
mean-field product state, and the native clock solver is explicitly not implemented.

## 4. Limitations and exposure disclosures

- **Same reviewer seat and session as the initial review.** I am verifying my own findings. This is
  the weakest axis of this round and is why every outcome rests on freshly executed counterexamples.
- **One prior error corrected.** The initial review's "toolchain no longer exists" clause was wrong;
  the builder's dispute is upheld and the clause is withdrawn.
- **Installer hash unverified.** Documented and consistent with the installed `nvcc 13.3.73`, but I
  did not download the installer to check its SHA-256 against NVIDIA.
- **Version gate exercised by stub.** No toolkit older than 12.8 was installed; the request
  expressly permitted a stubbed configure-time test.
- **`build.sh` not executed.** The POSIX entry point is verified by text and by shared CMake modules;
  the Linux CI run covers the Python suite, not the native build.
- **GPUBackend integration not reproduced.** The locked environment is CPU-only (`torch 2.10.0+cpu`),
  so the builder's 64-cell CUDA-tensor evolution numbers are not independently reproduced. No
  outcome depends on them.
- **No exposure-boundary violation.** `C:\src\POPGP-custody`, sealed holdouts, secret seeds, private
  evaluator logic and `POPGP_Codex_Handoff.md` were never opened; no untracked builder binary was
  used as proof — the DLL I inspected I built myself; no other reviewer worktree's notes were read;
  VIA-000 was not started, `holdout_started` was not changed, no custody data was revealed and no
  packet lifecycle was advanced.
