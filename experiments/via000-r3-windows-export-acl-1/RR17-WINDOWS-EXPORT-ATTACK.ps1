param(
    [Parameter(Mandatory = $true)][string]$MutableRoot,
    [Parameter(Mandatory = $true)][string]$ExportRoot,
    [Parameter(Mandatory = $true)][string]$EnvelopePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-SuccessMarker {
    param([Parameter(Mandatory = $true)][string]$Name)
    [IO.File]::WriteAllText(
        (Join-Path $MutableRoot "$Name-succeeded"),
        $Name,
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-Attack {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Operation
    )
    try {
        & $Operation
        Write-SuccessMarker -Name $Name
    } catch {
        # Expected: the exact production low-integrity token cannot mutate this boundary.
    }
}

[IO.File]::WriteAllText(
    (Join-Path $MutableRoot "mutable-write-allowed"),
    "allowed",
    [Text.UTF8Encoding]::new($false)
)
$replacement = Join-Path $MutableRoot "replacement.json"
[IO.File]::WriteAllText($replacement, "replacement`n", [Text.UTF8Encoding]::new($false))

Invoke-Attack -Name "create" -Operation {
    [IO.File]::WriteAllText(
        (Join-Path $ExportRoot "created.txt"),
        "created",
        [Text.UTF8Encoding]::new($false)
    )
}
Invoke-Attack -Name "write" -Operation {
    [IO.File]::WriteAllText($EnvelopePath, "changed`n", [Text.UTF8Encoding]::new($false))
}
Invoke-Attack -Name "hardlink" -Operation {
    New-Item -ItemType HardLink -Path (Join-Path $ExportRoot "hardlink.json") `
        -Target $EnvelopePath -ErrorAction Stop | Out-Null
}
Invoke-Attack -Name "reparse" -Operation {
    New-Item -ItemType SymbolicLink -Path (Join-Path $ExportRoot "reparse") `
        -Target $MutableRoot -ErrorAction Stop | Out-Null
}
Invoke-Attack -Name "rename" -Operation {
    [IO.File]::Move($EnvelopePath, (Join-Path $ExportRoot "renamed.json"))
}
Invoke-Attack -Name "delete" -Operation {
    [IO.File]::Delete($EnvelopePath)
}
Invoke-Attack -Name "replace" -Operation {
    [IO.File]::Move($replacement, $EnvelopePath, $true)
}

exit 0
