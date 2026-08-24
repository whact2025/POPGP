param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "exact-proof-baseline",
        "minimal-pwsh-baseline",
        "minimal-cmd-baseline",
        "token-environment-pwsh",
        "no-lua-low-il-pwsh"
    )]
    [string]$Factor,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$ControlRoot,
    [Parameter(Mandatory = $true)][string]$GithubOutput,
    [Parameter(Mandatory = $true)][string]$SourceSha,
    [Parameter(Mandatory = $true)][string]$WorkflowSha,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$RunAttempt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedPowerShell = "C:\Program Files\PowerShell\7\pwsh.exe"
$expectedPath = "C:\Program Files\PowerShell\7;C:\Windows\System32;C:\Windows"
$observedPowerShell = [IO.Path]::GetFullPath(
    [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
)
$psHomePowerShell = [IO.Path]::GetFullPath((Join-Path $PSHOME "pwsh.exe"))
$profileFiles = @(
    $PROFILE.AllUsersAllHosts,
    $PROFILE.AllUsersCurrentHost,
    $PROFILE.CurrentUserAllHosts,
    $PROFILE.CurrentUserCurrentHost
) | Select-Object -Unique
if (-not $observedPowerShell.Equals($expectedPowerShell, [StringComparison]::OrdinalIgnoreCase) -or
    -not $psHomePowerShell.Equals($expectedPowerShell, [StringComparison]::OrdinalIgnoreCase) -or
    $PSVersionTable.PSVersion.ToString() -cnotmatch '^7\.[0-9]+\.[0-9]+$' -or
    @($profileFiles | Where-Object { Test-Path -LiteralPath $_ }).Count -ne 0 -or
    [string]$env:PATH -cne $expectedPath) {
    throw "RR13 control plane is not the reviewed built-in PowerShell 7 boundary"
}
foreach ($value in @($RepoRoot, $ControlRoot, $GithubOutput)) {
    if (-not [IO.Path]::IsPathFullyQualified($value)) { throw "RR13 control path is not absolute" }
}
if ($SourceSha -cnotmatch '^[0-9a-f]{40}$' -or $WorkflowSha -cnotmatch '^[0-9a-f]{40}$' -or
    $RunId -cnotmatch '^[1-9][0-9]*$' -or $RunAttempt -cnotmatch '^[1-9][0-9]*$') {
    throw "RR13 control-plane identity is malformed"
}
if (Test-Path -LiteralPath $ControlRoot) { throw "RR13 control root is not fresh" }
[void](New-Item -ItemType Directory -Path $ControlRoot)

function Get-Rr13Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-Rr13OrdinaryFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description,
        [switch]$AllowMultipleLinks
    )
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $streams = @(Get-Item -LiteralPath $Path -Stream * -ErrorAction Stop)
    $links = @(& "C:\Windows\System32\fsutil.exe" hardlink list $Path)
    $linkExit = $LASTEXITCODE
    if (-not ($item -is [IO.FileInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        $streams.Count -ne 1 -or [string]$streams[0].Stream -cne ':$DATA' -or
        $linkExit -ne 0 -or (-not $AllowMultipleLinks -and $links.Count -ne 1)) {
        throw "$Description is not one ordinary single-link file"
    }
    return $item.FullName
}

$protocolRoot = Join-Path $RepoRoot "protocols/POPGP-VIABILITY-R3-2026-08"
$containmentSource = Join-Path $protocolRoot "VIA-000-CONTAINMENT.ps1"
$proofRunnerSource = Join-Path $protocolRoot "VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
$hostileSource = Join-Path $protocolRoot "VIA-000-CONTAINMENT-HOSTILE.ps1"
$diagnosticNativeSource = Join-Path $protocolRoot "VIA-000-RR13-NATIVE-DIAGNOSTIC.ps1"
foreach ($binding in @(
    @($containmentSource, "44908f2529876865a88ac0ae82c5da7d38089ccd0159def4e2dbf2f55f43500d"),
    @($proofRunnerSource, "520fd7cbf8197c9304deaf43d3b51d4493a5212d8e00e3790398ad0c968f6ad4"),
    @($hostileSource, "bf3bc6470171d8d93b073de4fd85a87d45694e349de8d56dd8b3a4a4af038ebf")
)) {
    $path = Assert-Rr13OrdinaryFile -Path ([string]$binding[0]) -Description "reviewed campaign source"
    if ((Get-Rr13Sha256 -Path $path) -cne [string]$binding[1]) {
        throw "reviewed campaign source hash differs"
    }
}
. $containmentSource
if ($Factor -in @("token-environment-pwsh", "no-lua-low-il-pwsh")) {
    . $diagnosticNativeSource
    Initialize-Via000Rr13NativeDiagnostic
}
Test-Via000ContainmentAvailability -PlatformFamily "windows-x86_64" -SystemTools @{}

$workspace = Join-Path $ControlRoot "workspace"
$mutable = Join-Path $workspace "mutable"
$evidence = Join-Path $workspace "protected-evidence"
$toolClosure = Join-Path $workspace "protected-tool"
foreach ($path in @($workspace, $mutable, $evidence, $toolClosure)) {
    [void](New-Item -ItemType Directory -Path $path)
}
$protectedEvidence = Join-Path $evidence "protected-evidence.txt"
$protectedTool = Join-Path $toolClosure "protected-tool.txt"
[IO.File]::WriteAllText($protectedEvidence, "rr13-protected-evidence", [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($protectedTool, "rr13-protected-tool", [Text.UTF8Encoding]::new($false))
$evidenceHash = Get-Rr13Sha256 -Path $protectedEvidence
$toolHash = Get-Rr13Sha256 -Path $protectedTool
Set-Via000RootIntegrity -Path $workspace -Kind traverse -SystemTools @{}
Set-Via000RootIntegrity -Path $mutable -Kind mutable -SystemTools @{}
Set-Via000RootIntegrity -Path $evidence -Kind protected -SystemTools @{}
Set-Via000RootIntegrity -Path $toolClosure -Kind protected -SystemTools @{}
Protect-Via000ReadOnlyClosure -Path $toolClosure -SystemTools @{}

$stdout = Join-Path $evidence "stdout.txt"
$stderr = Join-Path $evidence "stderr.txt"
$containedResult = Join-Path $evidence "containment-result.json"
$minimalSentinel = Join-Path $mutable "rr13-sentinel.bin"
$pwsh = Assert-Rr13OrdinaryFile -Path $expectedPowerShell `
    -Description "trusted PowerShell" -AllowMultipleLinks
$cmd = Assert-Rr13OrdinaryFile -Path "C:\Windows\System32\cmd.exe" `
    -Description "trusted cmd" -AllowMultipleLinks
$closure = @{
    $containmentSource = Get-Rr13Sha256 -Path $containmentSource
    $proofRunnerSource = Get-Rr13Sha256 -Path $proofRunnerSource
    $hostileSource = Get-Rr13Sha256 -Path $hostileSource
    $protectedTool = $toolHash
    $pwsh = Get-Rr13Sha256 -Path $pwsh
    $cmd = Get-Rr13Sha256 -Path $cmd
}
if ($Factor -in @("token-environment-pwsh", "no-lua-low-il-pwsh")) {
    $closure[$diagnosticNativeSource] = Get-Rr13Sha256 -Path $diagnosticNativeSource
}
Assert-Via000Closure -Closure $closure -MutableRoot $mutable -Moment "rr13-before"

$executable = $pwsh
$arguments = @()
$sentinel = $minimalSentinel
$tokenFlags = "DISABLE_MAX_PRIVILEGE|LUA_TOKEN"
$environmentConstruction = "manual-filtered-parent"
$nativePhase = "not-started"
$privilegesDisabled = $true
$integritySid = "S-1-16-4096"
$invokeMessage = ""
if ($Factor -eq "exact-proof-baseline") {
    $mutableFixture = Join-Path $mutable "hostile.ps1"
    Copy-Item -LiteralPath $hostileSource -Destination $mutableFixture
    if ((Get-Rr13Sha256 -Path $mutableFixture) -cne (Get-Rr13Sha256 -Path $hostileSource)) {
        throw "exact hostile fixture copy differs"
    }
    $sentinel = Join-Path $mutable "child-of-child-ready"
    $arguments = @(
        "-NoLogo", "-NoProfile", "-NonInteractive", "-File", $mutableFixture,
        "-Mode", "attack", "-PowerShellPath", $pwsh,
        "-MutableRoot", $mutable, "-TrustedEvidencePath", $protectedEvidence,
        "-TrustedToolPath", $protectedTool, "-StageId", "candidate"
    )
} elseif ($Factor -eq "minimal-cmd-baseline") {
    $executable = $cmd
    $arguments = @(
        "/d", "/s", "/c",
        '@echo off & >"%VIA000_SENTINEL%" <nul set /p "=rr13" & exit /b 0'
    )
} else {
    $arguments = @(
        "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
        '[IO.File]::WriteAllBytes([string]$env:VIA000_SENTINEL,[Text.Encoding]::ASCII.GetBytes("rr13")); exit 0'
    )
}

if ($Factor -in @("exact-proof-baseline", "minimal-pwsh-baseline", "minimal-cmd-baseline")) {
    $additionalEnvironment = if ($Factor -eq "exact-proof-baseline") {
        @{}
    } else {
        @{ VIA000_SENTINEL = $minimalSentinel }
    }
    try {
        Invoke-Via000ContainedCommand -Label "rr13-$Factor" `
            -ContractId "rr13-native-child-factor-diagnostic" `
            -PlatformFamily "windows-x86_64" -FilePath $executable `
            -Arguments $arguments -WorkingDirectory $mutable -MutableRoot $mutable `
            -TrustedRoot $evidence -StdoutPath $stdout -StderrPath $stderr `
            -ResultPath $containedResult -Environment $additionalEnvironment `
            -Closure $closure -SystemTools @{} -TimeoutSeconds 45
    } catch {
        $invokeMessage = [string]$_.Exception.Message
    }
    if (-not (Test-Path -LiteralPath $containedResult -PathType Leaf)) {
        throw "exact campaign primitive did not persist its post-teardown result"
    }
    $result = Get-Content -LiteralPath $containedResult -Raw | ConvertFrom-Json -AsHashtable
    $nativePhase = "post-teardown-quiescent"
} else {
    $nativeStaging = Join-Path $mutable "native-staging"
    [void](New-Item -ItemType Directory -Path $nativeStaging)
    Set-Via000RootIntegrity -Path $nativeStaging -Kind mutable -SystemTools @{}
    $untrustedStdout = Join-Path $nativeStaging "stdout.txt"
    $untrustedStderr = Join-Path $nativeStaging "stderr.txt"
    $cleanEnvironment = ConvertTo-Via000CleanEnvironment `
        -Additional @{ VIA000_SENTINEL = $minimalSentinel } -MutableTemp $nativeStaging
    $dictionary = [Collections.Generic.Dictionary[string,string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($entry in $cleanEnvironment.GetEnumerator()) {
        $dictionary[[string]$entry.Key] = [string]$entry.Value
    }
    $native = [Via000Rr13.NativeDiagnostic]::Run(
        $executable, $arguments, $mutable, $dictionary,
        $untrustedStdout, $untrustedStderr, 45, $Factor,
        $nativeStaging, $minimalSentinel, $expectedPath
    )
    foreach ($pair in @(@($untrustedStdout, $stdout), @($untrustedStderr, $stderr))) {
        if (-not (Test-Path -LiteralPath $pair[0] -PathType Leaf)) {
            [IO.File]::WriteAllText($pair[0], "", [Text.UTF8Encoding]::new($false))
        }
        Copy-Item -LiteralPath $pair[0] -Destination $pair[1]
    }
    $result = [ordered]@{
        schema_version = 1
        primitive = [string]$native.Primitive
        direct_exit_code = [int]$native.ExitCode
        exit_code = [int]$native.ExitCode
        timed_out = [bool]$native.TimedOut
        descendants_quiescent = ([uint32]$native.ActiveProcessesAfterTermination -eq 0)
        active_processes_after_teardown = [uint32]$native.ActiveProcessesAfterTermination
        stdout_sha256 = Get-Rr13Sha256 -Path $stdout
        stderr_sha256 = Get-Rr13Sha256 -Path $stderr
        phase = [string]$native.Phase
        token_flags = [string]$native.TokenFlags
        environment_construction = [string]$native.EnvironmentConstruction
        integrity_sid = [string]$native.IntegritySid
        privileges_disabled = [bool]$native.PrivilegesDisabled
    }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $containedResult -Encoding utf8NoBOM
    $nativePhase = [string]$native.Phase
    $tokenFlags = [string]$native.TokenFlags
    $environmentConstruction = [string]$native.EnvironmentConstruction
    $integritySid = [string]$native.IntegritySid
    $privilegesDisabled = [bool]$native.PrivilegesDisabled
    Remove-Item -LiteralPath $nativeStaging -Recurse -Force
}

if ([string]$result.primitive -cne "windows-low-integrity-restricted-token-job-object" -or
    $result.descendants_quiescent -ne $true -or
    [uint32]$result.active_processes_after_teardown -ne 0 -or
    $result.timed_out -ne $false -or $nativePhase -cne "post-teardown-quiescent") {
    throw "RR13 containment did not reach proven whole-job teardown"
}
Assert-Via000Closure -Closure $closure -MutableRoot $mutable -Moment "rr13-after"
$protectedUnchanged = (
    (Get-Rr13Sha256 -Path $protectedEvidence) -ceq $evidenceHash -and
    (Get-Rr13Sha256 -Path $protectedTool) -ceq $toolHash
)
if (-not $protectedUnchanged) { throw "RR13 protected closure changed" }
foreach ($path in @($stdout, $stderr, $containedResult)) {
    [void](Assert-Rr13OrdinaryFile -Path $path -Description "post-teardown diagnostic subject")
}

$sentinelPresent = Test-Path -LiteralPath $sentinel -PathType Leaf
if ($sentinelPresent -and $Factor -ne "exact-proof-baseline") {
    $sentinelBytes = [IO.File]::ReadAllBytes($sentinel)
    $expectedSentinel = [Text.Encoding]::ASCII.GetBytes("rr13")
    if ($sentinelBytes.Length -ne $expectedSentinel.Length -or
        [Convert]::ToHexString($sentinelBytes) -cne [Convert]::ToHexString($expectedSentinel)) {
        throw "RR13 minimal sentinel bytes differ"
    }
}
$exitSigned = [int]$result.exit_code
$exitUnsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes($exitSigned), 0)
$exitHex = "0x{0:X8}" -f $exitUnsigned
$stdoutLength = (Get-Item -LiteralPath $stdout).Length
$stderrLength = (Get-Item -LiteralPath $stderr).Length
$executablePath = [IO.Path]::GetFullPath($executable)
$summary = [ordered]@{
    schema_version = 1
    diagnostic_kind = "via000-r3-rr13-windows-native-child-factor"
    non_authoritative = $true
    factor = $Factor
    source_sha = $SourceSha
    workflow_sha = $WorkflowSha
    run_id = $RunId
    run_attempt = $RunAttempt
    platform = "windows-x86_64"
    native_phase = $nativePhase
    exit_signed = $exitSigned
    exit_unsigned = [uint64]$exitUnsigned
    exit_hex = $exitHex
    sentinel_present = [bool]$sentinelPresent
    stdout_nonempty = ($stdoutLength -gt 0)
    stderr_nonempty = ($stderrLength -gt 0)
    stdout_sha256 = Get-Rr13Sha256 -Path $stdout
    stderr_sha256 = Get-Rr13Sha256 -Path $stderr
    token_flags = $tokenFlags
    environment_construction = $environmentConstruction
    executable_path = $executablePath
    executable_sha256 = Get-Rr13Sha256 -Path $executablePath
    low_integrity_sid = $integritySid
    privileges_disabled = [bool]$privilegesDisabled
    suspended_creation = $true
    assigned_to_job_before_resume = $true
    kill_on_job_close = $true
    explicit_job_termination = $true
    active_processes_after_teardown = 0
    protected_root_unchanged = [bool]$protectedUnchanged
    control_environment_scrubbed = $true
    invoke_reported_nonzero = [bool]($invokeMessage.Length -gt 0)
}
$summaryPath = Join-Path $ControlRoot "diagnostic-summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8NoBOM
[void](Assert-Rr13OrdinaryFile -Path $summaryPath -Description "RR13 trusted summary")
$summaryHash = Get-Rr13Sha256 -Path $summaryPath
$beforeOutput = (Get-Item -LiteralPath $GithubOutput -Force -ErrorAction Stop).Length
@(
    "summary_sha256=$summaryHash",
    "exit_signed=$exitSigned",
    "exit_unsigned=$exitUnsigned",
    "exit_hex=$exitHex",
    "sentinel_present=$($sentinelPresent.ToString().ToLowerInvariant())",
    "stdout_nonempty=$(($stdoutLength -gt 0).ToString().ToLowerInvariant())",
    "stderr_nonempty=$(($stderrLength -gt 0).ToString().ToLowerInvariant())"
) | Add-Content -LiteralPath $GithubOutput -Encoding utf8NoBOM
if ((Get-Item -LiteralPath $GithubOutput -Force -ErrorAction Stop).Length -le $beforeOutput) {
    throw "RR13 output facts were not appended"
}
Write-Host (
    "RR13 factor={0} phase={1} exit_signed={2} exit_unsigned={3} exit_hex={4} " +
    "sentinel_present={5} stdout_nonempty={6} stderr_nonempty={7} " +
    "token_flags={8} environment={9} active_after=0"
) -f $Factor, $nativePhase, $exitSigned, $exitUnsigned, $exitHex,
    $sentinelPresent.ToString().ToLowerInvariant(),
    ($stdoutLength -gt 0).ToString().ToLowerInvariant(),
    ($stderrLength -gt 0).ToString().ToLowerInvariant(),
    $tokenFlags, $environmentConstruction
