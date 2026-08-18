# Builder response: POPGP-REVIEW-BLACKWELL-VIABILITY-1-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-BLACKWELL-VIABILITY-1-RESPONSE-1"
response_round: 1
response_date: "2026-08-17"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r1-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-BLACKWELL-VIABILITY-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-BLACKWELL-VIABILITY-1.md"
review_commit: "d14c95db2a5500e27c0ff2fa80bb81f5c4c65004"
candidate_commit_reviewed: "dfe222ce3a173f617471d1979f0021ae6ec23ebf"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No sealed holdout, secret seed, private evaluator, or custody content was accessed. The campaign remains pre-holdout."

summary: |-
  All nine findings and all nine requested tests are addressed in the implementation
  and verification record below. The four reported blockers were accepted, except
  that BUILD-002 is partially accepted because the review's factual claim that the
  local CUDA toolkit no longer existed was refuted by direct execution. Its underlying
  reproducibility concern was accepted and remediated.

  The native gate now fails on zero or partial discovery and requires exactly seven
  CTest cases. CUDA 12.8 is the explicit minimum, the exact 13.3.1 installer and hash
  are documented, architecture arguments are validated, and the linked binary is
  checked with cuobjdump against the requested and visible device architecture. The
  benchmark checks launch/synchronization errors, validates read-back state and norms,
  performs an untimed warmup, emits a deterministic checksum, and is itself a CTest.
  Clock reconstruction fails explicitly as not implemented, pruning is active and
  tested, and device printf output was removed.

  The final scientific candidate is
  9a29e05f803666bf0e3a28417ea399e3e26769fc (tree
  358fb1af6ca587b6c71ff2ef0fb87e335163eeaf). It is published, a clean remote clone
  checked it out successfully, and exact-SHA GitHub Actions run 32088634733 succeeded.
  Protocol snapshot 792d2d2737dcf48cf14cbb3af78a64ba2daad1f6 and activated
  handoff 197219bbb006b4cd1f8b9f992cadedae8d09a341 are also published. The
  campaign validates, remains pending, and holdout_started is false. These changes
  establish a reproducible native implementation calibration; they do not establish
  VIA-300, Tier R, entangling observables, or external scientific validation.

finding_responses:
  - finding_id: "BUILD-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Both build entry points pass --no-tests=error. CMake records the exact expected
      native count and a post-CTest verifier rejects zero or partial discovery. The
      native suite now contains six scientific/control cases plus the validating
      benchmark, for an expected count of seven.
    changed_files:
      - "popgp_engine/build.bat"
      - "popgp_engine/build.sh"
      - "popgp_engine/cmake/VerifyCTestCount.cmake"
      - "popgp_engine/kernel/CMakeLists.txt"
      - "tests/unit/test_native_build_contract.py"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "popgp_engine\\build.bat --clean --test --cuda-arch 120"
        result: "exit 0; exactly 7 CTest cases discovered and 7 passed on the Blackwell device"
      - command: "uv run pytest -q tests/unit/test_native_build_contract.py::test_native_test_gate_rejects_zero_and_partial_discovery"
        result: "exit 0; zero and partial discovery both rejected while the exact expected count passed"
    residual_risk: "Native test discovery still depends on CMake/GTest integration, but its observed count is now an explicit fail-closed contract."
    disagreement_ref: ""

  - finding_id: "BUILD-002"
    blocking_as_reported: true
    disposition: partially-accepted
    implementation_status: implemented
    rationale: |-
      The isolated CUDA 13.3 toolkit did exist at
      C:\\src\\POPGP-cuda-toolkit-13.3\\local; nvcc 13.3.73 and cuobjdump were
      executed there, so the review's machine-absence premise is disputed. The
      repository nevertheless lacked a durable minimum-version and provenance
      contract. CMake now requires CUDAToolkit 12.8 or newer and fails clearly; the
      kernel README pins the exact 13.3.1 installer URL and SHA-256 and documents
      CUDA_PATH setup. REPRODUCIBILITY identifies the calibration as single-machine.
    changed_files:
      - "popgp_engine/CMakeLists.txt"
      - "popgp_engine/kernel/README.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "C:\\src\\POPGP-cuda-toolkit-13.3\\local\\bin\\nvcc.exe --version"
        result: "exit 0; CUDA compilation tools release 13.3, V13.3.73"
      - command: "popgp_engine\\build.bat --clean --test --cuda-arch 120"
        result: "exit 0; configure accepted CUDA 13.3.73 and produced a verified sm_120 DLL"
    residual_risk: "The repository pins provenance and a minimum version but does not redistribute NVIDIA's toolkit; installation still depends on the external official installer."
    disagreement_ref: "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md#candidate-hardware-and-toolchain"

  - finding_id: "BENCH-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The benchmark now performs an untimed warmup, checks cudaGetLastError after
      launches, synchronizes before accepting timing, copies both spinor components
      back, rejects nonfinite values or excessive unit-norm error, and prints a
      deterministic FNV-1a-64 checksum. It exits nonzero on any failed check and is
      registered in CTest.
    changed_files:
      - "popgp_engine/kernel/src/main.cpp"
      - "popgp_engine/kernel/CMakeLists.txt"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "popgp_engine\\build\\kernel\\Release\\popgp_sim.exe --verbose"
        result: "exit 0; 448.57 ms, 2.23e8 edge updates/s, checksum 48e6ef8f40cb137c, maximum norm error 1.1435297153639112e-14"
      - command: "ctest --test-dir popgp_engine/build -C Release --output-on-failure --no-tests=error"
        result: "exit 0; benchmark_validation passed as one of exactly seven native cases"
    residual_risk: "The throughput remains a single-machine calibration and is not a scientific observable or cross-hardware performance claim."
    disagreement_ref: ""

  - finding_id: "CAMP-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The originally requested historical candidate, snapshot, and activation commits
      are fetchable from the declared public repository. The final remediated candidate
      is also published; a fresh network clone checked it out by full SHA. Exact-SHA
      Actions run 32088634733 succeeded. The final protocol snapshot and activation
      are published, and no holdout or dependent packet was started.
    changed_files:
      - "protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-010.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-100.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-150.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-200.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-300.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-400.yaml"
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/protocol.json"
    fix_commits:
      - "9a29e05f803666bf0e3a28417ea399e3e26769fc"
      - "792d2d2737dcf48cf14cbb3af78a64ba2daad1f6"
      - "197219bbb006b4cd1f8b9f992cadedae8d09a341"
    verification:
      - command: "git clone --no-checkout https://github.com/whact2025/POPGP.git C:\\src\\POPGP-exact-sha-preflight && git -C C:\\src\\POPGP-exact-sha-preflight checkout --detach 9a29e05f803666bf0e3a28417ea399e3e26769fc"
        result: "exit 0; fresh clone resolved exact full SHA"
      - command: "gh run view 32088634733 --json headSha,status,conclusion,url"
        result: "headSha 9a29e05f803666bf0e3a28417ea399e3e26769fc; completed; success"
      - command: "uv run python scripts/check_viability_campaign.py reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
        result: "exit 0; Viability campaign contract is valid"
    residual_risk: "Publication and CI establish accessibility and software reproducibility, not independent execution, custody integrity after holdout start, or scientific viability."
    disagreement_ref: ""

  - finding_id: "ENG-001"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Platform selection is isolated behind a module helper so tests exercise both
      branches without mutating process-wide os.name. The load-path test proves that
      Windows passes winmode=0 and POSIX passes no winmode keyword.
    changed_files:
      - "popgp/engine.py"
      - "tests/unit/test_engine.py"
    fix_commits:
      - "e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"
      - "7437c2fa878a9a21d7b7582f7d60b2a5ecf00445"
    verification:
      - command: "uv run pytest -q tests/unit/test_engine.py"
        result: "exit 0; 4 passed on Windows and exact-SHA Linux CI passed the same suite"
    residual_risk: "ctypes loader behavior still depends on the host's DLL search semantics and installed runtime dependencies."
    disagreement_ref: ""

  - finding_id: "BUILD-003"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Both scripts reject missing, option-like, and malformed architecture values.
      The post-link verifier requires cuobjdump, validates every listed ELF, and when
      visible-device enforcement is enabled requires the requested numeric target to
      match a visible architecture. Script-mode CMake now initializes its policy
      version for consistent IN_LIST behavior on local and CI CMake releases.
    changed_files:
      - "popgp_engine/build.bat"
      - "popgp_engine/build.sh"
      - "popgp_engine/cmake/ValidateCudaArchitecture.cmake"
      - "popgp_engine/cmake/VerifyCudaBinary.cmake"
      - "popgp_engine/kernel/CMakeLists.txt"
      - "tests/unit/test_native_build_contract.py"
    fix_commits:
      - "e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"
      - "9a29e05f803666bf0e3a28417ea399e3e26769fc"
    verification:
      - command: "uv run pytest -q tests/unit/test_native_build_contract.py"
        result: "exit 0; 3 tests covered argument, count, and architecture-verifier contracts"
      - command: "popgp_engine\\build.bat --clean --test --cuda-arch 86"
        result: "exit 1 after link; requested sm_86 rejected against visible sm_120; final sm_120 rebuild passed"
    residual_risk: "Special nonnumeric CMake architecture modes cannot be equated to one visible device and therefore receive binary-content validation without the numeric visible-device equality check."
    disagreement_ref: ""

  - finding_id: "SCOPE-001"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The identity clock stub now returns POPGP_STATUS_NOT_IMPLEMENTED and does not
      modify its output. The area-law pruning branch is active and tested for retained
      and zeroed edges. Device printf calls were removed from the timed kernel path.
    changed_files:
      - "popgp_engine/kernel/include/types.cuh"
      - "popgp_engine/kernel/src/clock.cu"
      - "popgp_engine/kernel/src/area_law.cu"
      - "popgp_engine/kernel/tests/test_clock.cu"
      - "popgp_engine/kernel/tests/test_area_law.cu"
      - "popgp_engine/kernel/CMakeLists.txt"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "ctest --test-dir popgp_engine/build -C Release --output-on-failure --no-tests=error"
        result: "exit 0; clock explicit-not-implemented and pruning-transition cases passed within the exact seven-case gate"
    residual_risk: "Clock reconstruction remains unavailable by design; callers must treat the explicit status as a retained capability limitation."
    disagreement_ref: ""

  - finding_id: "DOC-001"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The kernel README now attributes node-disjoint batching to the callers, states
      the clock limitation, and describes pruning accurately. Engine.step documents
      its node-disjoint precondition and lack of internal coloring, with a regression
      test on the public docstring.
    changed_files:
      - "popgp_engine/kernel/README.md"
      - "popgp/engine.py"
      - "tests/unit/test_engine.py"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "uv run pytest -q tests/unit/test_engine.py::test_engine_step_documents_node_disjoint_edge_precondition"
        result: "exit 0; public precondition and no-coloring statements retained"
    residual_risk: "The API cannot dynamically prove arbitrary external callers supplied node-disjoint batches."
    disagreement_ref: ""

  - finding_id: "CAMP-002"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The validator now rejects a packet lifecycle beyond drafted until every declared
      dependency has a passing adjudicated outcome. The exact preregistered-dependent
      mutation is persisted, while the current six dependent packets remain drafted.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
    fix_commits: ["e6f8dc5a55032f92ecfd5a18505705fd9387f1d4"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_dependent_packet_cannot_be_preregistered_before_dependencies_pass"
        result: "exit 0; preregistration while VIA-000 is pending was rejected and the passing dependency control was accepted"
    residual_risk: "Lifecycle correctness still depends on authentic, hash-reconciled adjudication artifacts, which the broader campaign contract checks separately."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-BUILD-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_native_build_contract.py::test_native_test_gate_rejects_zero_and_partial_discovery"
      - "popgp_engine/cmake/VerifyCTestCount.cmake"
    verification:
      - command: "uv run pytest -q tests/unit/test_native_build_contract.py::test_native_test_gate_rejects_zero_and_partial_discovery"
        result: "exit 0; exact count passed and zero/partial counts failed"
    rationale: "The requested empty-discovery counterexample and the additional partial-discovery case are fail-closed."
    disagreement_ref: ""

  - requested_test_id: "TST-BUILD-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "popgp_engine/CMakeLists.txt"
      - "popgp_engine/kernel/README.md"
    verification:
      - command: "popgp_engine\\build.bat --clean --test --cuda-arch 120"
        result: "exit 0 with CUDA 13.3.73; CMake contains a fatal CUDA 12.8 minimum-version precondition"
    rationale: "The final build demonstrates the positive version path; the declarative minimum fails during find_package/configure before compilation on older toolkits."
    disagreement_ref: ""

  - requested_test_id: "TST-BENCH-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "popgp_engine/kernel/src/main.cpp"
      - "popgp_engine/kernel/CMakeLists.txt"
    verification:
      - command: "ctest --test-dir popgp_engine/build -C Release -R benchmark_validation --output-on-failure --no-tests=error"
        result: "exit 0; launch/sync/readback/finite/norm/checksum validation completed"
    rationale: "A mismatched numeric architecture is now rejected by the mandatory post-link verifier before an invalid benchmark can be accepted; the benchmark independently rejects CUDA or output failures at runtime."
    disagreement_ref: ""

  - requested_test_id: "TST-CAMP-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md"
      - ".github/workflows/ci.yml"
    verification:
      - command: "gh api repos/whact2025/POPGP/commits/9a29e05f803666bf0e3a28417ea399e3e26769fc --jq .sha"
        result: "returned the exact candidate SHA"
      - command: "gh run view 32088634733 --json headSha,status,conclusion,url"
        result: "exact candidate; completed; success"
    rationale: "The old and final frozen commit chains are published, a clean clone checks out the final candidate, and exact-SHA CI is green before holdout start."
    disagreement_ref: ""

  - requested_test_id: "TST-ENG-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_engine.py::test_load_library_uses_platform_specific_dll_search_semantics"
    verification:
      - command: "uv run pytest -q tests/unit/test_engine.py"
        result: "exit 0; 4 passed, including Windows winmode=0 and POSIX no-keyword controls"
    rationale: "The test exercises the CDLL call itself and fails if the Windows keyword is removed."
    disagreement_ref: ""

  - requested_test_id: "TST-BUILD-003"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_native_build_contract.py::test_cuda_architecture_argument_contract"
      - "tests/unit/test_native_build_contract.py::test_cuda_binary_architecture_verifier_rejects_mismatch"
    verification:
      - command: "uv run pytest -q tests/unit/test_native_build_contract.py"
        result: "exit 0; 3 passed"
      - command: "build.bat --cuda-arch --clean --test; build.bat --cuda-arch not-an-arch; build.bat --clean --test --cuda-arch 86"
        result: "all three negative controls exited nonzero; the final 120 control rebuilt and passed"
    rationale: "Missing/malformed values fail early and a valid-but-wrong numeric target fails mandatory binary/device verification."
    disagreement_ref: ""

  - requested_test_id: "TST-SCOPE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "popgp_engine/kernel/tests/test_clock.cu::ClockTest.UnimplementedSolverFailsWithoutWritingOutput"
      - "popgp_engine/kernel/tests/test_area_law.cu::AreaLawTest.PruningTransitionsAreApplied"
    verification:
      - command: "ctest --test-dir popgp_engine/build -C Release --output-on-failure --no-tests=error"
        result: "exit 0; both cases passed within 7/7 native results"
    rationale: "The stub is loud, pruning behavior is executable, and timed kernels emit no device printf output."
    disagreement_ref: ""

  - requested_test_id: "TST-DOC-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_engine.py::test_engine_step_documents_node_disjoint_edge_precondition"
    verification:
      - command: "uv run pytest -q tests/unit/test_engine.py::test_engine_step_documents_node_disjoint_edge_precondition"
        result: "exit 0"
    rationale: "Both kernel documentation and the public Python API now assign node-disjoint batching to the caller."
    disagreement_ref: ""

  - requested_test_id: "TST-CAMP-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_dependent_packet_cannot_be_preregistered_before_dependencies_pass"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_dependent_packet_cannot_be_preregistered_before_dependencies_pass"
        result: "exit 0"
    rationale: "The exact lifecycle mutation now fails until the declared prerequisite has a passing adjudication."
    disagreement_ref: ""

new_or_changed_risks:
  - "The CUDA toolkit is version/provenance constrained but externally installed and not redistributed by this repository."
  - "The numeric architecture verifier intentionally fails a cross-compile target that does not match the visible device when visible-device enforcement is enabled."
  - "The native clock solver is explicitly unavailable; its nonzero status is a retained limitation, not an implemented clock mechanism."
  - "The native backend remains mean-field and cannot satisfy VIA-300's entangling MI/QCMI requirements."
  - "All hardware results are builder-owned single-machine calibration under a shared operator."

external_actions:
  - action: "Publish and run CI for the exact final scientific candidate."
    owner: "fuocor"
    status: complete
    evidence_ref: "https://github.com/whact2025/POPGP/actions/runs/32088634733"
  - action: "Verify a fresh public-network clone can check out the exact final scientific candidate."
    owner: "fuocor"
    status: complete
    evidence_ref: "reviews/viability/POPGP-VIABILITY-R1-2026-08/calibration/VIA-300-Blackwell-remediation-2026-08-17.md#publication-and-ci"
  - action: "Run a fresh independent adversarial re-review of this response handoff before merge or holdout start."
    owner: "independent reviewer"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "All 9 findings, all 9 requested tests, the full dfe222ce..handoff diff, exact candidate publication/CI, native Blackwell evidence, campaign refreeze invariants, regressions, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation claims are not independent resolution. Keep holdout_started false and do not treat native mean-field calibration as VIA-300 or Tier R evidence."
```

The builder does not assign final resolution status. That determination belongs to a
fresh independent re-review bound to the response-containing handoff commit.
