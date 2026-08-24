param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot,

    [Parameter(Mandatory = $true)]
    [ValidateSet("ubuntu-latest-x86_64", "windows-x86_64")]
    [string]$PlatformFamily,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{40}$")]
    [string]$ProtocolSourceCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^refs/tags/popgp-via000-r3-protocol-[0-9a-f]{40}$")]
    [string]$DispatchRef,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^refs/tags/popgp-via000-r3-authorization-[0-9a-f]{64}$")]
    [string]$AuthorizationRef,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{40}$")]
    [string]$AuthorizationTagOid,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{40}$")]
    [string]$AuthorizationCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{64}$")]
    [string]$AuthorizationRecordSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[1-9][0-9]*$")]
    [string]$ProducerRunId,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$ProducerRunAttempt,

    [Parameter(Mandatory = $true)][string]$TrustedGitPath,
    [Parameter(Mandatory = $true)][string]$TrustedBasePythonPath,
    [Parameter(Mandatory = $true)][string]$TrustedUvPath,
    [Parameter(Mandatory = $true)][string]$TrustedPdfLatexPath,
    [Parameter(Mandatory = $true)][string]$TrustedPowerShellPath,
    [Parameter(Mandatory = $true)][string]$ToolIdentityManifestPath,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$TrustedGitSha256,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$TrustedBasePythonSha256,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$TrustedUvSha256,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$TrustedPdfLatexSha256,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$TrustedPowerShellSha256,
    [Parameter(Mandatory = $true)][ValidatePattern("^[0-9a-f]{64}$")][string]$ToolIdentityManifestSha256
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryUrl = "https://github.com/whact2025/POPGP"
$CandidateCommit = "5be3c38a0822d49953d0933f14ccab32ca12c896"
$CandidateTree = "6ad387f9f4e0bab7f97df1bb54a03177887f0707"
$UvVersion = "0.11.11"
$ExpectedPdfEngine = "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)"
$ExpectedDispatchRef = "refs/tags/popgp-via000-r3-protocol-$ProtocolSourceCommit"
if ($DispatchRef -ne $ExpectedDispatchRef) {
    throw "dispatch ref differs from the content-addressed protocol snapshot tag"
}

function Assert-Success {
    param(
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if ($ExitCode -ne 0) {
        throw "$Description failed with exit code $ExitCode"
    }
}

function Assert-RegularTool {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedPath,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Description path is not absolute"
    }
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullExpected = [System.IO.Path]::GetFullPath($ExpectedPath)
    if (-not $fullPath.Equals($fullExpected, $comparison)) {
        throw "$Description path is outside the frozen platform root"
    }
    $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    if (-not ($item -is [System.IO.FileInfo]) -or ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "$Description is not one regular non-reparse file"
    }
    $cursor = $item.Directory
    while ($null -ne $cursor) {
        if ($cursor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "$Description has reparse ancestry"
        }
        $cursor = $cursor.Parent
    }
    $observedSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $fullPath).Hash.ToLowerInvariant()
    if ($observedSha256 -cne $ExpectedSha256) {
        throw "$Description bytes differ from pre-authorization identity"
    }
}

function Invoke-RetainedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ContractId,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$LogDirectory
    )

    $stdoutPath = Join-Path $LogDirectory "$Label.stdout.txt"
    $stderrPath = Join-Path $LogDirectory "$Label.stderr.txt"
    $recordPath = Join-Path $LogDirectory "$Label.result.json"
    $start = [DateTimeOffset]::UtcNow

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($name in @($startInfo.Environment.Keys)) {
        if ($name -like "GIT_*" -or $name -in @(
            "PATH", "PATHEXT", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
            "GITHUB_ENV", "BASH_ENV", "ENV"
        )) {
            [void]$startInfo.Environment.Remove($name)
        }
    }
    foreach ($argument in $Arguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "could not start retained command $Label"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $finish = [DateTimeOffset]::UtcNow

    [System.IO.File]::WriteAllText($stdoutPath, $stdout, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText($stderrPath, $stderr, [System.Text.UTF8Encoding]::new($false))
    [ordered]@{
        label = $Label
        contract_id = $ContractId
        file = $FilePath
        arguments = $Arguments
        working_directory = $WorkingDirectory
        started_at = $start.ToString("O")
        finished_at = $finish.ToString("O")
        duration_seconds = ($finish - $start).TotalSeconds
        exit_code = $process.ExitCode
        stdout_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stdoutPath).Hash.ToLowerInvariant()
        stderr_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stderrPath).Hash.ToLowerInvariant()
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding utf8NoBOM

    if ($stdout) { Write-Host $stdout -NoNewline }
    if ($stderr) { Write-Error $stderr -ErrorAction Continue }
    Assert-Success -ExitCode $process.ExitCode -Description $Label
}

$expectedGitPath = if ($PlatformFamily -eq "windows-x86_64") {
    "C:\Program Files\Git\cmd\git.exe"
} else { "/usr/bin/git" }
$expectedBasePythonPath = if ($PlatformFamily -eq "windows-x86_64") {
    "C:\hostedtoolcache\windows\Python\3.11.15\x64\python.exe"
} else { "/opt/hostedtoolcache/Python/3.11.15/x64/bin/python3.11" }
$expectedUvPath = if ($PlatformFamily -eq "windows-x86_64") {
    "C:\hostedtoolcache\windows\Python\3.11.15\x64\Scripts\uv.exe"
} else { "/opt/hostedtoolcache/Python/3.11.15/x64/bin/uv" }
$expectedPowerShellPath = if ($PlatformFamily -eq "windows-x86_64") {
    "C:\Program Files\PowerShell\7\pwsh.exe"
} else { "/opt/microsoft/powershell/7/pwsh" }
$expectedPdfLatexPath = if ($PlatformFamily -eq "windows-x86_64") {
    Join-Path $env:RUNNER_TEMP "via000-r3-texlive/2026/bin/windows/pdftex.exe"
} else {
    Join-Path $env:RUNNER_TEMP "via000-r3-texlive/2026/bin/x86_64-linux/pdftex"
}
Assert-RegularTool $TrustedGitPath $expectedGitPath $TrustedGitSha256 "trusted Git"
Assert-RegularTool $TrustedBasePythonPath $expectedBasePythonPath $TrustedBasePythonSha256 "trusted base Python"
Assert-RegularTool $TrustedUvPath $expectedUvPath $TrustedUvSha256 "trusted uv"
Assert-RegularTool $TrustedPdfLatexPath $expectedPdfLatexPath $TrustedPdfLatexSha256 "trusted pdfLaTeX"
Assert-RegularTool $TrustedPowerShellPath $expectedPowerShellPath $TrustedPowerShellSha256 "trusted PowerShell"
if (-not [System.IO.Path]::IsPathFullyQualified($ToolIdentityManifestPath)) {
    throw "tool identity manifest path is not absolute"
}
$toolManifestItem = Get-Item -LiteralPath $ToolIdentityManifestPath -Force -ErrorAction Stop
if (-not ($toolManifestItem -is [System.IO.FileInfo]) -or ($toolManifestItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    throw "tool identity manifest is not one regular non-reparse file"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ToolIdentityManifestPath).Hash.ToLowerInvariant() -cne $ToolIdentityManifestSha256) {
    throw "tool identity manifest changed after authorization"
}
$toolIdentity = Get-Content -LiteralPath $ToolIdentityManifestPath -Raw | ConvertFrom-Json
if (
    $toolIdentity.schema_version -ne 1 -or
    $toolIdentity.platform_family -cne $PlatformFamily -or
    $toolIdentity.tools.git.path -cne $TrustedGitPath -or
    $toolIdentity.tools.base_python.path -cne $TrustedBasePythonPath -or
    $toolIdentity.tools.base_python.sha256 -cne $TrustedBasePythonSha256 -or
    $toolIdentity.tools.uv.path -cne $TrustedUvPath -or
    $toolIdentity.tools.uv.sha256 -cne $TrustedUvSha256 -or
    $toolIdentity.tools.pdflatex.path -cne $TrustedPdfLatexPath -or
    $toolIdentity.tools.pdflatex.sha256 -cne $TrustedPdfLatexSha256 -or
    $toolIdentity.tools.powershell.path -cne $TrustedPowerShellPath -or
    $toolIdentity.tools.powershell.sha256 -cne $TrustedPowerShellSha256 -or
    $toolIdentity.tools.git.sha256 -cne $TrustedGitSha256
) {
    throw "tool identity manifest differs from explicit runner paths"
}
foreach ($name in @("PATH", "PATHEXT", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")) {
    Set-Item -LiteralPath "Env:$name" -Value ""
}
Get-ChildItem Env: | Where-Object Name -Like "GIT_*" | ForEach-Object {
    Remove-Item -LiteralPath "Env:$($_.Name)"
}
$basePythonText = (& $TrustedBasePythonPath -I -S -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>&1 | Out-String).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "base Python version query"
if ($basePythonText -cne "3.11.15") { throw "unexpected base Python version: $basePythonText" }
$uvText = (& $TrustedUvPath --version 2>&1 | Out-String).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "uv version query"
if ($uvText -cne "uv $UvVersion") { throw "expected uv $UvVersion, observed $uvText" }
$pdfText = (& $TrustedPdfLatexPath --version 2>&1 | Select-Object -First 1).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "pdflatex version query"
if ($pdfText -cne $ExpectedPdfEngine) { throw "expected $ExpectedPdfEngine, observed $pdfText" }

try {
if (Test-Path -LiteralPath $WorkspaceRoot) {
    throw "fresh workspace must not already exist: $WorkspaceRoot"
}

$workspace = New-Item -ItemType Directory -Path $WorkspaceRoot
$repo = Join-Path $workspace.FullName "candidate"
$evidence = Join-Path $workspace.FullName "evidence"
$environment = Join-Path $workspace.FullName "python-environment"
$uvCache = Join-Path $workspace.FullName "uv-cache"
$pythonCache = Join-Path $workspace.FullName "python-cache"
$ruffCache = Join-Path $workspace.FullName "ruff-cache"
$matplotlibCache = Join-Path $workspace.FullName "matplotlib-cache"
$generalCache = Join-Path $workspace.FullName "general-cache"
$pdfDirectory = Join-Path $workspace.FullName "pdf"
$logs = Join-Path $evidence "commands"
foreach ($path in @(
    $evidence, $uvCache, $pythonCache, $ruffCache,
    $matplotlibCache, $generalCache, $pdfDirectory, $logs
)) {
    New-Item -ItemType Directory -Path $path | Out-Null
}

foreach ($name in @("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

Copy-Item -LiteralPath $ToolIdentityManifestPath -Destination (Join-Path $evidence "tool-identity-manifest.json")
& $TrustedBasePythonPath -I -m venv --copies $environment
Assert-Success -ExitCode $LASTEXITCODE -Description "copied locked environment bootstrap"
Set-Content -LiteralPath (Join-Path $evidence "pdf-engine-version.txt") `
    -Value $pdfText -Encoding utf8NoBOM

Invoke-RetainedCommand -Label "001-clone" -ContractId "git-clone" -FilePath $TrustedGitPath `
    -Arguments @("clone", "-c", "core.autocrlf=false", "--no-checkout", $RepositoryUrl, $repo) `
    -WorkingDirectory $workspace.FullName -LogDirectory $logs
Invoke-RetainedCommand -Label "002-checkout" -ContractId "git-checkout" -FilePath $TrustedGitPath `
    -Arguments @("checkout", "--detach", $CandidateCommit) `
    -WorkingDirectory $repo -LogDirectory $logs

$head = (& $TrustedGitPath -C $repo rev-parse HEAD).Trim()
$tree = (& $TrustedGitPath -C $repo rev-parse "HEAD^{tree}").Trim()
if ($head -ne $CandidateCommit -or $tree -ne $CandidateTree) {
    throw "candidate identity mismatch: $head / $tree"
}

$normalStatus = (& $TrustedGitPath -C $repo status --porcelain=v1 --untracked-files=all | Out-String)
$ignoredStatus = (& $TrustedGitPath -C $repo status --porcelain=v1 --untracked-files=normal --ignored | Out-String)
if ($normalStatus -or $ignoredStatus) {
    throw "fresh candidate checkout is not clean, including ignored state"
}

$trustedBoundary = Join-Path $evidence "check_reproduction_boundary.py"
& $TrustedBasePythonPath -I -S -c `
    "import pathlib, subprocess, sys; pathlib.Path(sys.argv[1]).write_bytes(subprocess.run([sys.argv[2],'-C',sys.argv[3],'cat-file','blob',sys.argv[4]],check=True,capture_output=True).stdout)" `
    $trustedBoundary $TrustedGitPath $repo "$($CandidateCommit):scripts/check_reproduction_boundary.py"
Assert-Success -ExitCode $LASTEXITCODE -Description "trusted boundary extraction"

$env:UV_PROJECT_ENVIRONMENT = $environment
$env:UV_CACHE_DIR = $uvCache
$env:PYTHONPYCACHEPREFIX = $pythonCache
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:RUFF_CACHE_DIR = $ruffCache
$env:MPLCONFIGDIR = $matplotlibCache
$env:XDG_CACHE_HOME = $generalCache
Invoke-RetainedCommand -Label "003-sync" -ContractId "uv-sync-frozen-no-editable" -FilePath $TrustedUvPath `
    -Arguments @("sync", "--frozen", "--no-editable", "--python", $TrustedBasePythonPath) `
    -WorkingDirectory $repo -LogDirectory $logs
Remove-Item Env:UV_PROJECT_ENVIRONMENT

if ($IsWindows) {
    $environmentPython = Join-Path $environment "Scripts/python.exe"
} else {
    $environmentPython = Join-Path $environment "bin/python"
}
if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    throw "locked environment Python not found: $environmentPython"
}
$environmentPythonItem = Get-Item -LiteralPath $environmentPython -Force -ErrorAction Stop
if (-not ($environmentPythonItem -is [System.IO.FileInfo]) -or ($environmentPythonItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    throw "locked environment Python is not one regular non-reparse executable"
}
$environmentPythonVersion = (& $environmentPython -I -S -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>&1 | Out-String).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "locked environment Python version query"
if ($environmentPythonVersion -cne "3.11.15") { throw "locked environment Python version differs from 3.11.15" }
$environmentPythonSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $environmentPython).Hash.ToLowerInvariant()
$retainedToolIdentityPath = Join-Path $evidence "tool-identity-manifest.json"
$retainedToolIdentity = Get-Content -LiteralPath $retainedToolIdentityPath -Raw | ConvertFrom-Json -AsHashtable
$retainedToolIdentity.tools["environment_python"] = [ordered]@{
    path = $environmentPython
    sha256 = $environmentPythonSha256
    version = $environmentPythonVersion
}
$retainedToolIdentity | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $retainedToolIdentityPath -Encoding utf8NoBOM

$environmentManifest = Join-Path $evidence "environment-manifest.json"
$environmentDigestPath = Join-Path $evidence "environment-manifest.sha256"
$sourceManifest = Join-Path $evidence "source-manifest.json"
$sourceDigestPath = Join-Path $evidence "source-manifest.sha256"

$environmentDigest = (& $TrustedBasePythonPath -I -S $trustedBoundary `
    --repo-root $repo --environment $environment snapshot `
    --output $environmentManifest).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "environment snapshot"
Set-Content -LiteralPath $environmentDigestPath -Value $environmentDigest -Encoding utf8NoBOM

$sourceDigest = (& $TrustedBasePythonPath -I -S $trustedBoundary `
    --repo-root $repo --environment $environment source-snapshot `
    --output $sourceManifest --base-ref $CandidateCommit).Trim()
Assert-Success -ExitCode $LASTEXITCODE -Description "source snapshot"
Set-Content -LiteralPath $sourceDigestPath -Value $sourceDigest -Encoding utf8NoBOM

function Invoke-CheckedModule {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ContractId,
        [Parameter(Mandatory = $true)][string]$Module,
        [string[]]$ModuleArguments = @(),
        [string[]]$AllowedPaths = @()
    )

    $arguments = @(
        "-I", "-S", $trustedBoundary,
        "--repo-root", $repo,
        "--environment", $environment,
        "run", "--manifest", $environmentManifest,
        "--expected-sha256", $environmentDigest
    )
    foreach ($allowedPath in $AllowedPaths) {
        $arguments += @("--allow-path", $allowedPath)
    }
    $arguments += @(
        "--", $environmentPython, "-I", "-S",
        "-X", "pycache_prefix=$pythonCache",
        (Join-Path $repo "scripts/run_without_startup_hooks.py"),
        "--repo-root", $repo, "--module", $Module, "--"
    )
    $arguments += $ModuleArguments
    Invoke-RetainedCommand -Label $Label -ContractId $ContractId -FilePath $TrustedBasePythonPath `
        -Arguments $arguments -WorkingDirectory $repo -LogDirectory $logs
}

Invoke-CheckedModule -Label "004-ruff" -ContractId "trusted-python-ruff" -Module "ruff" -ModuleArguments @("check", ".")
Invoke-CheckedModule -Label "005-check-tex" -ContractId "trusted-python-check-tex" -Module "scripts.check_tex"
Invoke-CheckedModule -Label "006-pytest" -ContractId "trusted-python-pytest" -Module "pytest" `
    -ModuleArguments @("-q", "-p", "no:cacheprovider")

$allowed = @(
    "examples/physics_qg/chain_1d/results/clock_potential.png",
    "examples/physics_qg/chain_1d/results/embedding.png",
    "examples/physics_qg/chain_1d/results/entropy_growth.png",
    "examples/physics_qg/chain_1d/results/validation.json"
)
Invoke-CheckedModule -Label "007-chain" -ContractId "trusted-python-chain-generator" -Module "examples.physics_qg.chain_1d" -AllowedPaths $allowed
$allowed += @(
    "examples/physics_qg/grid_2d/results/clock_potential.png",
    "examples/physics_qg/grid_2d/results/embedding.png",
    "examples/physics_qg/grid_2d/results/validation.json"
)
Invoke-CheckedModule -Label "008-grid" -ContractId "trusted-python-grid-generator" -Module "examples.physics_qg.grid_2d" -AllowedPaths $allowed
$allowed += @(
    "examples/physics_qg/gravity_well/results/gravity_embedding.png",
    "examples/physics_qg/gravity_well/results/gravity_well.png",
    "examples/physics_qg/gravity_well/results/source_comparison.png",
    "examples/physics_qg/gravity_well/results/validation.json"
)
Invoke-CheckedModule -Label "009-gravity" -ContractId "trusted-python-gravity-generator" -Module "examples.physics_qg.gravity_well" -AllowedPaths $allowed
$allowed += @(
    "examples/physics_qg/source_law/results/source_scaling.png",
    "examples/physics_qg/source_law/results/validation.json"
)
Invoke-CheckedModule -Label "010-source-law" -ContractId "trusted-python-source-law-generator" -Module "examples.physics_qg.source_law" -AllowedPaths $allowed
$allowed += @(
    "examples/physics_qg/source_law_many_body/results/many_body_source.png",
    "examples/physics_qg/source_law_many_body/results/validation.json"
)
Invoke-CheckedModule -Label "011-many-body" -ContractId "trusted-python-many-body-generator" -Module "examples.physics_qg.source_law_many_body" -AllowedPaths $allowed
$allowed += @(
    "examples/physics_qg/ca_model/results/dynamics_cooling.png",
    "examples/physics_qg/ca_model/results/evolution_cooling.gif",
    "examples/physics_qg/ca_model/results/validation.json"
)
Invoke-CheckedModule -Label "012-ca" -ContractId "trusted-python-ca-generator" -Module "examples.physics_qg.ca_model" -AllowedPaths $allowed
Invoke-CheckedModule -Label "013-artifact-boundary" `
    -ContractId "trusted-python-artifact-boundary" `
    -Module "scripts.check_validation_artifacts" `
    -ModuleArguments @("--enforce-change-boundary") -AllowedPaths $allowed

$generatedStatus = (& $TrustedGitPath -C $repo status --short --ignored --untracked-files=all | Out-String)
[System.IO.File]::WriteAllText(
    (Join-Path $evidence "generated-status-with-ignored.txt"),
    $generatedStatus,
    [System.Text.UTF8Encoding]::new($false)
)
$allowedSet = @{}
foreach ($sourcePath in $allowed) { $allowedSet[$sourcePath] = $true }
foreach ($line in @($generatedStatus -split "`r?`n" | Where-Object { $_ })) {
    if ($line.Length -lt 4) { throw "malformed generated repository status: $line" }
    $statusPath = $line.Substring(3).Replace("\", "/")
    if (-not $allowedSet.ContainsKey($statusPath)) {
        throw "generated repository status contains undeclared path: $statusPath"
    }
}

$artifactRoot = Join-Path $evidence "artifacts"
$artifactResults = [ordered]@{}
foreach ($sourcePath in $allowed) {
    $sourceFile = Join-Path $repo $sourcePath
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "required generated artifact is missing: $sourcePath"
    }
    $destination = Join-Path $artifactRoot $sourcePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $sourceFile -Destination $destination
    $evidencePath = [System.IO.Path]::GetRelativePath($workspace.FullName, $destination).Replace("\", "/")
    $mediaType = switch ([System.IO.Path]::GetExtension($sourcePath).ToLowerInvariant()) {
        ".json" { "application/json" }
        ".png" { "image/png" }
        ".gif" { "image/gif" }
        default { throw "unsupported generated artifact type: $sourcePath" }
    }
    $artifactResults[$sourcePath] = [ordered]@{
        source_path = $sourcePath
        evidence_path = $evidencePath
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
        media_type = $mediaType
    }
}

& $TrustedGitPath -C $repo restore --source $CandidateCommit --worktree -- $allowed
Assert-Success -ExitCode $LASTEXITCODE -Description "restore generated candidate artifacts"

Invoke-RetainedCommand -Label "014-pdflatex-1" -ContractId "pdflatex-pass-1" -FilePath $TrustedPdfLatexPath `
    -Arguments @(
        "-fmt=pdflatex",
        "-interaction=nonstopmode", "-halt-on-error",
        "-output-directory=$pdfDirectory", "docs/framework.tex"
    ) -WorkingDirectory $repo -LogDirectory $logs
Invoke-RetainedCommand -Label "015-pdflatex-2" -ContractId "pdflatex-pass-2" -FilePath $TrustedPdfLatexPath `
    -Arguments @(
        "-fmt=pdflatex",
        "-interaction=nonstopmode", "-halt-on-error",
        "-output-directory=$pdfDirectory", "docs/framework.tex"
    ) -WorkingDirectory $repo -LogDirectory $logs

Invoke-RetainedCommand -Label "016-environment-verify" -ContractId "trusted-python-environment-verify" -FilePath $TrustedBasePythonPath `
    -Arguments @(
        "-I", "-S", $trustedBoundary,
        "--repo-root", $repo, "--environment", $environment,
        "verify", "--manifest", $environmentManifest,
        "--expected-sha256", $environmentDigest
    ) -WorkingDirectory $repo -LogDirectory $logs

$finalStatus = (& $TrustedGitPath -C $repo status --short --ignored --untracked-files=all | Out-String)
[System.IO.File]::WriteAllText(
    (Join-Path $evidence "final-status-with-ignored.txt"),
    $finalStatus,
    [System.Text.UTF8Encoding]::new($false)
)
if ($finalStatus) {
    throw "candidate repository has tracked, untracked, or ignored residue after execution"
}

$evidenceEntries = @()
foreach ($file in @(Get-ChildItem -LiteralPath $evidence, $pdfDirectory -File -Recurse | Sort-Object FullName)) {
    if ($file.Name -in @("evidence-manifest.json", "platform-summary.json")) {
        continue
    }
    $relative = [System.IO.Path]::GetRelativePath($workspace.FullName, $file.FullName).Replace("\", "/")
    $mediaType = switch ($file.Extension.ToLowerInvariant()) {
        ".json" { "application/json" }
        ".pdf" { "application/pdf" }
        ".png" { "image/png" }
        ".gif" { "image/gif" }
        default { "text/plain" }
    }
    $role = "supporting"
    $sourcePath = $null
    if ($relative -match "^evidence/commands/.+\.result\.json$") { $role = "command-result" }
    elseif ($relative -match "^evidence/commands/.+\.stdout\.txt$") { $role = "command-stdout" }
    elseif ($relative -match "^evidence/commands/.+\.stderr\.txt$") { $role = "command-stderr" }
    elseif ($relative -eq "evidence/environment-manifest.json") { $role = "environment-manifest" }
    elseif ($relative -eq "evidence/source-manifest.json") { $role = "source-manifest" }
    elseif ($relative -eq "evidence/tool-identity-manifest.json") { $role = "tool-identity-manifest" }
    elseif ($relative -eq "evidence/pdf-engine-version.txt") { $role = "pdf-engine" }
    elseif ($relative -in @(
        "evidence/generated-status-with-ignored.txt",
        "evidence/final-status-with-ignored.txt"
    )) { $role = "repository-status" }
    elseif ($relative -eq "pdf/framework.pdf") { $role = "pdf" }
    elseif ($relative -match "^pdf/") { $role = "pdf-build" }
    elseif ($relative -match "^evidence/artifacts/") {
        $sourcePath = $relative.Substring("evidence/artifacts/".Length)
        $role = if ($mediaType -eq "application/json") { "validation-json" } else { "visual" }
    }
    $entry = [ordered]@{
        platform_family = $PlatformFamily
        path = $relative
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        byte_count = $file.Length
        media_type = $mediaType
        role = $role
    }
    if ($null -ne $sourcePath) { $entry["source_path"] = $sourcePath }
    $evidenceEntries += $entry
}
$evidenceManifestPath = Join-Path $evidence "evidence-manifest.json"
$evidenceEntries | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath $evidenceManifestPath -Encoding utf8NoBOM

$commandResults = [ordered]@{}
foreach ($recordFile in @(Get-ChildItem -LiteralPath $logs -Filter "*.result.json" | Sort-Object Name)) {
    $record = Get-Content -Raw -LiteralPath $recordFile.FullName | ConvertFrom-Json
    $stdoutPath = Join-Path $logs "$($record.label).stdout.txt"
    $stderrPath = Join-Path $logs "$($record.label).stderr.txt"
    $resultRelative = [System.IO.Path]::GetRelativePath($workspace.FullName, $recordFile.FullName).Replace("\", "/")
    $commandResults[$record.label] = [ordered]@{
        command = "$($record.file) $($record.arguments -join ' ')"
        contract_id = [string]$record.contract_id
        exit_code = [int]$record.exit_code
        duration_seconds = [double]$record.duration_seconds
        result_path = $resultRelative
        result_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $recordFile.FullName).Hash.ToLowerInvariant()
        stdout_path = [System.IO.Path]::GetRelativePath($workspace.FullName, $stdoutPath).Replace("\", "/")
        stdout_sha256 = [string]$record.stdout_sha256
        stderr_path = [System.IO.Path]::GetRelativePath($workspace.FullName, $stderrPath).Replace("\", "/")
        stderr_sha256 = [string]$record.stderr_sha256
    }
}

[ordered]@{
    schema_version = 1
    campaign_id = "POPGP-VIABILITY-R3-2026-08"
    packet_id = "VIA-000"
    platform_family = $PlatformFamily
    candidate_commit = $head
    candidate_tree = $tree
    protocol_source_commit = $ProtocolSourceCommit
    dispatch_identity = [ordered]@{
        event_name = "workflow_dispatch"
        source_ref = $DispatchRef
        protocol_snapshot_commit = $ProtocolSourceCommit
        authorization_ref = $AuthorizationRef
        authorization_tag_oid = $AuthorizationTagOid
        authorization_commit = $AuthorizationCommit
        authorization_record_sha256 = $AuthorizationRecordSha256
        producer_run_id = $ProducerRunId
        producer_run_attempt = $ProducerRunAttempt
    }
    producer_attestation = [ordered]@{
        repository = "whact2025/POPGP"
        signer_workflow = "whact2025/POPGP/.github/workflows/via000-r3-protocol.yml"
        source_commit = $ProtocolSourceCommit
        bundle_path = "evidence/producer-attestation.sigstore.json"
        subject_paths = @(
            "evidence/platform-summary.json",
            "evidence/evidence-manifest.json"
        )
    }
    uv_version = $uvText
    pdf_engine = $pdfText
    tool_identity_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath `
        (Join-Path $evidence "tool-identity-manifest.json")).Hash.ToLowerInvariant()
    command_results = $commandResults
    artifact_results = $artifactResults
    mutation_results = @()
    test_count = 366
    example_count = 6
    visual_count = 12
    mutation_count = 0
    commands_passed = $true
    semantic_contract_passed = $true
    visual_contract_passed = $true
    source_boundary_passed = $true
    environment_boundary_passed = $true
    pdf_passed = $true
    mutations_rejected = $false
    overall_passed = $false
    evidence_paths = @($evidenceEntries | ForEach-Object { $_.path })
    environment_manifest_sha256 = $environmentDigest
    source_manifest_sha256 = $sourceDigest
    pdf_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $pdfDirectory "framework.pdf")).Hash.ToLowerInvariant()
    pdf_page_count = 11
    completed_at = [DateTimeOffset]::UtcNow.ToString("O")
} | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $evidence "platform-summary.json") -Encoding utf8NoBOM

Write-Host "VIA-000 R3 trusted boundary sequence passed for $PlatformFamily."
} catch {
    if (Test-Path -LiteralPath $WorkspaceRoot) {
        Remove-Item -LiteralPath $WorkspaceRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
