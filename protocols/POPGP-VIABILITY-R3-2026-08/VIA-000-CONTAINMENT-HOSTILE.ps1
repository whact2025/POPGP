param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("attack", "relay", "writer")]
    [string]$Mode,
    [Parameter(Mandatory = $true)][string]$PowerShellPath,
    [Parameter(Mandatory = $true)][string]$MutableRoot,
    [Parameter(Mandatory = $true)][string]$TrustedEvidencePath,
    [Parameter(Mandatory = $true)][string]$TrustedToolPath,
    [Parameter(Mandatory = $true)]
    [ValidateSet("candidate", "pdf", "mutation")]
    [string]$StageId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Marker {
    param([Parameter(Mandatory = $true)][string]$Name)
    [IO.File]::WriteAllText(
        (Join-Path $MutableRoot $Name),
        "$Mode/$StageId",
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-ProtectedAttacks {
    param([Parameter(Mandatory = $true)][string]$Phase)
    foreach ($pair in @(
        @("evidence", $TrustedEvidencePath),
        @("tool", $TrustedToolPath)
    )) {
        $name = [string]$pair[0]
        $target = [string]$pair[1]
        try {
            [IO.File]::WriteAllText($target, "direct-$Phase", [Text.UTF8Encoding]::new($false))
            Write-Marker -Name "$Phase-$name-direct-write-succeeded"
        } catch {}
        try {
            $before = [IO.File]::ReadAllBytes($target)
            [IO.File]::WriteAllText($target, "replacement-$Phase", [Text.UTF8Encoding]::new($false))
            [IO.File]::WriteAllBytes($target, $before)
            Write-Marker -Name "$Phase-$name-replace-restore-succeeded"
        } catch {}
        try {
            $link = Join-Path $MutableRoot "$Phase-$name-hardlink"
            if (Test-Path -LiteralPath $link) { Remove-Item -LiteralPath $link -Force }
            New-Item -ItemType HardLink -Path $link -Target $target -ErrorAction Stop | Out-Null
            Move-Item -LiteralPath $link -Destination $target -Force -ErrorAction Stop
            Write-Marker -Name "$Phase-$name-hardlink-substitution-succeeded"
        } catch {}
    }
    try {
        [void][IO.File]::ReadAllBytes($TrustedEvidencePath)
        Write-Marker -Name "$Phase-evidence-read-succeeded"
    } catch {}
    if (@(Get-ChildItem Env: | Where-Object {
        $_.Name -like "GITHUB_*" -or $_.Name -like "ACTIONS_*" -or
        $_.Name -like "RUNNER_*"
    }).Count -ne 0) {
        Write-Marker -Name "$Phase-control-plane-environment-leaked"
    }
}

function Start-Descendant {
    param([Parameter(Mandatory = $true)][ValidateSet("relay", "writer")][string]$ChildMode)
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $PowerShellPath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    foreach ($argument in @(
        "-NoLogo", "-NoProfile", "-NonInteractive", "-File", $PSCommandPath,
        "-Mode", $ChildMode, "-PowerShellPath", $PowerShellPath,
        "-MutableRoot", $MutableRoot, "-TrustedEvidencePath", $TrustedEvidencePath,
        "-TrustedToolPath", $TrustedToolPath, "-StageId", $StageId
    )) {
        [void]$info.ArgumentList.Add($argument)
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) { throw "could not start $ChildMode descendant" }
    $process.Dispose()
}

if ($Mode -eq "writer") {
    Write-Marker -Name "descendant-running"
    Start-Sleep -Seconds 4
    Invoke-ProtectedAttacks -Phase "delayed"
    Write-Marker -Name "delayed-descendant-survived"
    Start-Sleep -Seconds 60
    exit 0
}

if ($Mode -eq "relay") {
    Start-Descendant -ChildMode "writer"
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    while (-not (Test-Path -LiteralPath (Join-Path $MutableRoot "descendant-running"))) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) { exit 76 }
        Start-Sleep -Milliseconds 25
    }
    Write-Marker -Name "child-of-child-ready"
    Start-Sleep -Seconds 60
    exit 0
}

Invoke-ProtectedAttacks -Phase "direct"
Start-Descendant -ChildMode "relay"
$readyDeadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
while (-not (Test-Path -LiteralPath (Join-Path $MutableRoot "child-of-child-ready"))) {
    if ([DateTimeOffset]::UtcNow -ge $readyDeadline) { exit 77 }
    Start-Sleep -Milliseconds 25
}
exit 0
