param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ubuntu-latest-x86_64", "windows-x86_64")]
    [string]$PlatformFamily,
    [Parameter(Mandatory = $true)]
    [ValidateSet("candidate", "pdf", "mutation")]
    [string]$StageId,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$ProtocolPath,
    [Parameter(Mandatory = $true)][string]$WorkspaceRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][string]$PowerShellPath,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$EventName,
    [Parameter(Mandatory = $true)][string]$SourceRef,
    [Parameter(Mandatory = $true)][string]$SourceSha,
    [Parameter(Mandatory = $true)][string]$WorkflowRef,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$RunAttempt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProofSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-RegularProofFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not [IO.Path]::IsPathFullyQualified($Path)) { throw "$Description path is not absolute" }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not ($item -is [IO.FileInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$Description is not one regular non-reparse file"
    }
    return $item.FullName
}

function Assert-FrozenProofBundle {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Parameters,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $receiptRoot = Join-Path $Root (
        "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000"
    )
    $bindings = @(
        @("containment_protocol_path", "containment_protocol_sha256", "containment-protocol.ps1"),
        @("containment_proof_runner_path", "containment_proof_runner_sha256", "containment-proof-runner.ps1"),
        @("containment_proof_fixture_path", "containment_proof_fixture_sha256", "containment-proof-hostile.ps1"),
        @("containment_proof_schema_path", "containment_proof_schema_sha256", "containment-proof.schema.json"),
        @("containment_proof_aggregator_path", "containment_proof_aggregator_sha256", "containment-proof-aggregator.py"),
        @("containment_proof_workflow_path", "containment_proof_workflow_sha256", "containment-proof-workflow.yml")
    )
    $resolved = @{}
    foreach ($binding in $bindings) {
        $pathField = [string]$binding[0]
        $hashField = [string]$binding[1]
        $receiptName = [string]$binding[2]
        if (-not $Parameters.ContainsKey($pathField) -or -not $Parameters.ContainsKey($hashField)) {
            throw "protocol omits frozen containment proof binding $pathField/$hashField"
        }
        $expected = [string]$Parameters[$hashField]
        if ($expected -cnotmatch '^[0-9a-f]{64}$') { throw "malformed proof hash $hashField" }
        $source = Assert-RegularProofFile -Path (Join-Path $Root ([string]$Parameters[$pathField])) `
            -Description $pathField
        $receipt = Assert-RegularProofFile -Path (Join-Path $receiptRoot $receiptName) `
            -Description "$pathField receipt"
        if ((Get-ProofSha256 -Path $source) -cne $expected -or
            (Get-ProofSha256 -Path $receipt) -cne $expected) {
            throw "proof source/receipt/hash differs for $pathField"
        }
        $resolved[$pathField] = $source
    }
    return $resolved
}

if ($Repository -cne "whact2025/POPGP" -or $EventName -cne "push") {
    throw "containment proof accepts only the repository push control plane"
}
if ($SourceSha -cnotmatch '^[0-9a-f]{40}$' -or
    $RunId -cnotmatch '^[1-9][0-9]*$' -or $RunAttempt -cnotmatch '^[1-9][0-9]*$') {
    throw "containment proof source/run identity is malformed"
}
if ($SourceRef -cnotmatch '^refs/heads/(campaign|review)/via000-r3-protocol-[A-Za-z0-9._/-]+$') {
    throw "containment proof ref is outside the safe feature/review branch scope"
}
$expectedWorkflowRef = "$Repository/.github/workflows/via000-r3-containment-proof.yml@$SourceRef"
if ($WorkflowRef -cne $expectedWorkflowRef) { throw "containment proof workflow ref differs" }
foreach ($path in @($RepoRoot, $ProtocolPath, $WorkspaceRoot, $OutputRoot, $PowerShellPath)) {
    if (-not [IO.Path]::IsPathFullyQualified($path)) { throw "proof path is not absolute: $path" }
}
if ((Test-Path -LiteralPath $WorkspaceRoot) -or (Test-Path -LiteralPath $OutputRoot)) {
    throw "containment proof workspace or output already exists"
}

$protocolFile = Assert-RegularProofFile -Path $ProtocolPath -Description "proof protocol"
$protocol = Get-Content -LiteralPath $protocolFile -Raw | ConvertFrom-Json -AsHashtable
$parameters = $protocol.parameters
$bundle = Assert-FrozenProofBundle -Parameters $parameters -Root $RepoRoot
$trustedPowerShell = Assert-RegularProofFile -Path $PowerShellPath -Description "trusted PowerShell"
$expectedPowerShell = if ($PlatformFamily -eq "windows-x86_64") {
    "C:\Program Files\PowerShell\7\pwsh.exe"
} else {
    "/opt/microsoft/powershell/7/pwsh"
}
if ($trustedPowerShell -cne $expectedPowerShell) { throw "trusted PowerShell path differs" }

$helper = [string]$bundle.containment_protocol_path
. $helper
$systemTools = if ($PlatformFamily -eq "windows-x86_64") {
    @{}
} else {
    @{
        sudo = "/usr/bin/sudo"
        systemd_run = "/usr/bin/systemd-run"
        systemctl = "/usr/bin/systemctl"
        useradd = "/usr/sbin/useradd"
        userdel = "/usr/sbin/userdel"
        id = "/usr/bin/id"
    }
}
Test-Via000ContainmentAvailability -PlatformFamily $PlatformFamily -SystemTools $systemTools

$success = $false
try {
    New-Item -ItemType Directory -Path $WorkspaceRoot -ErrorAction Stop | Out-Null
    $mutable = Join-Path $WorkspaceRoot "mutable"
    $evidence = Join-Path $WorkspaceRoot "trusted-evidence"
    $toolClosure = Join-Path $WorkspaceRoot "tool-closure"
    foreach ($path in @($mutable, $evidence, $toolClosure)) {
        New-Item -ItemType Directory -Path $path -ErrorAction Stop | Out-Null
    }
    $evidenceSubject = Join-Path $evidence "protected-evidence.txt"
    $toolSubject = Join-Path $toolClosure "protected-tool.txt"
    $mutableFixture = Join-Path $mutable "hostile.ps1"
    Copy-Item -LiteralPath ([string]$bundle.containment_proof_fixture_path) `
        -Destination $mutableFixture -ErrorAction Stop
    if ((Get-ProofSha256 -Path $mutableFixture) -cne
        [string]$parameters.containment_proof_fixture_sha256) {
        throw "mutable hostile fixture copy differs from the frozen Git bytes"
    }
    [IO.File]::WriteAllText(
        $evidenceSubject,
        "non-scientific-evidence/$SourceSha/$PlatformFamily/$StageId",
        [Text.UTF8Encoding]::new($false)
    )
    [IO.File]::WriteAllText(
        $toolSubject,
        "non-scientific-tool/$SourceSha/$PlatformFamily/$StageId",
        [Text.UTF8Encoding]::new($false)
    )
    $evidenceHash = Get-ProofSha256 -Path $evidenceSubject
    $toolHash = Get-ProofSha256 -Path $toolSubject
    Set-Via000RootIntegrity -Path $WorkspaceRoot -Kind traverse -SystemTools $systemTools
    Set-Via000RootIntegrity -Path $mutable -Kind mutable -SystemTools $systemTools
    Set-Via000RootIntegrity -Path $evidence -Kind protected -SystemTools $systemTools
    Set-Via000RootIntegrity -Path $toolClosure -Kind protected -SystemTools $systemTools
    Protect-Via000ReadOnlyClosure -Path $toolClosure -SystemTools $systemTools

    $stdout = Join-Path $evidence "stdout.txt"
    $stderr = Join-Path $evidence "stderr.txt"
    $containedResult = Join-Path $evidence "containment-result.json"
    $closure = @{
        $trustedPowerShell = Get-ProofSha256 -Path $trustedPowerShell
        $toolSubject = $toolHash
        $protocolFile = Get-ProofSha256 -Path $protocolFile
    }
    foreach ($path in $bundle.Values) { $closure[[string]$path] = Get-ProofSha256 -Path $path }
    foreach ($path in $systemTools.Values) { $closure[[string]$path] = Get-ProofSha256 -Path $path }
    Invoke-Via000ContainedCommand -Label "proof-$StageId" `
        -ContractId "rr7-hosted-containment-proof" -PlatformFamily $PlatformFamily `
        -FilePath $trustedPowerShell -Arguments @(
            "-NoLogo", "-NoProfile", "-NonInteractive", "-File", $mutableFixture,
            "-Mode", "attack", "-PowerShellPath", $trustedPowerShell,
            "-MutableRoot", $mutable, "-TrustedEvidencePath", $evidenceSubject,
            "-TrustedToolPath", $toolSubject, "-StageId", $StageId
        ) -WorkingDirectory $mutable -MutableRoot $mutable -TrustedRoot $evidence `
        -StdoutPath $stdout -StderrPath $stderr -ResultPath $containedResult `
        -Environment @{} -Closure $closure -SystemTools $systemTools -TimeoutSeconds 45

    $result = Get-Content -LiteralPath $containedResult -Raw | ConvertFrom-Json -AsHashtable
    $expectedPrimitive = if ($PlatformFamily -eq "windows-x86_64") {
        "windows-low-integrity-restricted-token-job-object"
    } else {
        "ubuntu-systemd-ephemeral-user-control-group"
    }
    $expectedPrivilege = if ($PlatformFamily -eq "windows-x86_64") {
        "low-integrity-restricted-token"
    } else {
        "systemd-ephemeral-user"
    }
    if ($result.primitive -cne $expectedPrimitive -or
        $result.privilege_separation -cne $expectedPrivilege -or
        $result.descendants_quiescent -ne $true -or
        $result.active_processes_after_teardown -ne 0 -or
        $result.exit_code -ne 0 -or $result.timed_out -ne $false) {
        throw "production containment result is incomplete or uses the wrong primitive"
    }
    if ($PlatformFamily -eq "ubuntu-latest-x86_64" -and (
        [string]$result.ephemeral_identity_uid -cnotmatch '^[1-9][0-9]*$' -or
        $result.ephemeral_identity_processes_empty -ne $true -or
        $result.ephemeral_identity_removed -ne $true
    )) {
        throw "production containment result did not retire the ephemeral identity"
    }
    foreach ($required in @("descendant-running", "child-of-child-ready")) {
        if (-not (Test-Path -LiteralPath (Join-Path $mutable $required) -PathType Leaf)) {
            throw "hostile fixture did not establish the live child-of-child precondition"
        }
    }
    Start-Sleep -Seconds 5
    $forbiddenMarkers = @(
        Get-ChildItem -LiteralPath $mutable -File -ErrorAction Stop |
            Where-Object {
                $_.Name -like "*-succeeded" -or
                $_.Name -eq "delayed-descendant-survived" -or
                $_.Name -like "*-control-plane-environment-leaked"
            }
    )
    if ($forbiddenMarkers.Count -ne 0) {
        throw "hostile fixture crossed a protected boundary: $($forbiddenMarkers.Name -join ',')"
    }
    if ((Get-ProofSha256 -Path $evidenceSubject) -cne $evidenceHash -or
        (Get-ProofSha256 -Path $toolSubject) -cne $toolHash) {
        throw "protected evidence or tool subject changed after containment"
    }
    Assert-Via000Closure -Closure $closure -MutableRoot $mutable -Moment "proof-final"

    $artifactName = "via000-r3-containment-proof-$PlatformFamily-$StageId"
    $proof = [ordered]@{
        schema_version = 1
        proof_kind = "via000-r3-hosted-containment-cell"
        non_scientific = $true
        repository = $Repository
        workflow = ".github/workflows/via000-r3-containment-proof.yml"
        workflow_ref = $WorkflowRef
        event_name = $EventName
        source_ref = $SourceRef
        source_sha = $SourceSha
        run_id = $RunId
        run_attempt = $RunAttempt
        platform_family = $PlatformFamily
        stage_id = $StageId
        cell = "$PlatformFamily/$StageId"
        artifact_name = $artifactName
        production_helper_sha256 = [string]$parameters.containment_protocol_sha256
        proof_runner_sha256 = [string]$parameters.containment_proof_runner_sha256
        hostile_fixture_sha256 = [string]$parameters.containment_proof_fixture_sha256
        proof_schema_sha256 = [string]$parameters.containment_proof_schema_sha256
        proof_aggregator_sha256 = [string]$parameters.containment_proof_aggregator_sha256
        proof_workflow_sha256 = [string]$parameters.containment_proof_workflow_sha256
        receipt_bindings_verified = $true
        primitive = $expectedPrimitive
        privilege_separation = $expectedPrivilege
        descendants_quiescent = $true
        active_processes_after_teardown = 0
        os_process_tree_empty = $true
        untrusted_identity_processes_empty = $true
        untrusted_identity_retired = $true
        child_of_child_observed_before_direct_exit = $true
        protected_evidence_read_denied = $true
        protected_evidence_write_denied = $true
        protected_tool_write_denied = $true
        replace_restore_denied = $true
        hardlink_substitution_denied = $true
        control_plane_environment_scrubbed = $true
        delayed_descendant_write_absent = $true
        closure_unchanged = $true
        protected_evidence_sha256 = $evidenceHash
        protected_tool_sha256 = $toolHash
        containment_result_sha256 = Get-ProofSha256 -Path $containedResult
        no_campaign_execution = $true
        no_candidate_checkout = $true
        no_lifecycle_mutation = $true
        no_custody_access = $true
        no_commitment_or_reveal = $true
    }
    $proofPath = Join-Path $evidence "proof.json"
    $proof | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $proofPath -Encoding utf8NoBOM
    New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
    foreach ($path in @($proofPath, $containedResult, $stdout, $stderr)) {
        $destination = Join-Path $OutputRoot ([IO.Path]::GetFileName($path))
        [IO.File]::WriteAllBytes($destination, [IO.File]::ReadAllBytes($path))
        [IO.File]::SetAttributes($destination, [IO.FileAttributes]::Normal)
    }
    Protect-Via000ReadOnlyClosure -Path $OutputRoot -SystemTools $systemTools
    $success = $true
} finally {
    if (Test-Path -LiteralPath $WorkspaceRoot) {
        Remove-Item -LiteralPath $WorkspaceRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not $success -and (Test-Path -LiteralPath $OutputRoot)) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
