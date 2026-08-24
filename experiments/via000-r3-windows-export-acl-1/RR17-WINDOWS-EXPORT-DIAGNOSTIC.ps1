param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("capture", "recheck")]
    [string]$Mode,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$RunnerTemp,
    [Parameter(Mandatory = $true)][string]$PowerShellPath,
    [Parameter(Mandatory = $true)][string]$SourceSha,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$RunAttempt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $IsWindows) { throw "RR17 export descriptor diagnostic requires Windows" }
foreach ($value in @($RepoRoot, $RunnerTemp, $PowerShellPath)) {
    if (-not [IO.Path]::IsPathFullyQualified($value)) { throw "RR17 received a relative path" }
}
if ($SourceSha -cnotmatch '^[0-9a-f]{40}$' -or $RunId -cnotmatch '^[1-9][0-9]*$' -or
    $RunAttempt -cnotmatch '^[1-9][0-9]*$') {
    throw "RR17 run identity is malformed"
}

$helperPath = Join-Path $RepoRoot "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
$attackPath = Join-Path $RepoRoot "experiments/via000-r3-windows-export-acl-1/RR17-WINDOWS-EXPORT-ATTACK.ps1"
foreach ($path in @($helperPath, $attackPath, $PowerShellPath)) {
    $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
    if (-not ($item -is [IO.FileInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "RR17 trusted source is not an ordinary file"
    }
}
. $helperPath
Initialize-Via000WindowsNative

if (-not ("Via000R3Rr17.NativeDescriptor" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.AccessControl;

namespace Via000R3Rr17 {
  public sealed class RawAceResult {
    public int Index { get; set; }
    public string Sid { get; set; } = "";
    public string AceType { get; set; } = "";
    public int AceTypeNumeric { get; set; }
    public string AceQualifier { get; set; } = "";
    public long AccessMask { get; set; }
    public string AceFlags { get; set; } = "";
    public int AceFlagsNumeric { get; set; }
    public string InheritanceFlags { get; set; } = "";
    public int InheritanceFlagsNumeric { get; set; }
    public string PropagationFlags { get; set; } = "";
    public int PropagationFlagsNumeric { get; set; }
    public bool IsInherited { get; set; }
  }

  public sealed class DescriptorResult {
    public string OwnerSid { get; set; } = "";
    public string ControlFlags { get; set; } = "";
    public int ControlFlagsNumeric { get; set; }
    public bool DaclProtected { get; set; }
    public int RawAceCount { get; set; }
    public RawAceResult[] Aces { get; set; } = new RawAceResult[0];
  }

  public static class NativeDescriptor {
    const uint OWNER_SECURITY_INFORMATION = 0x00000001;
    const uint DACL_SECURITY_INFORMATION = 0x00000004;

    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern uint GetNamedSecurityInfo(
      string name, int objectType, uint information, out IntPtr owner, out IntPtr group,
      out IntPtr dacl, out IntPtr sacl, out IntPtr descriptor);
    [DllImport("advapi32.dll", SetLastError=true)]
    static extern uint GetSecurityDescriptorLength(IntPtr descriptor);
    [DllImport("kernel32.dll")]
    static extern IntPtr LocalFree(IntPtr memory);

    public static DescriptorResult Query(string path) {
      IntPtr descriptor = IntPtr.Zero;
      try {
        uint status = GetNamedSecurityInfo(
          path, 1, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
          out _, out _, out _, out _, out descriptor);
        if (status != 0) throw new Win32Exception((int)status, "GetNamedSecurityInfo RR17");
        if (descriptor == IntPtr.Zero) throw new InvalidOperationException("native descriptor is absent");
        uint length = GetSecurityDescriptorLength(descriptor);
        if (length == 0 || length > 1048576) throw new InvalidOperationException("native descriptor length differs");
        byte[] bytes = new byte[length];
        Marshal.Copy(descriptor, bytes, 0, checked((int)length));
        var raw = new RawSecurityDescriptor(bytes, 0);
        var output = new DescriptorResult {
          OwnerSid = raw.Owner == null ? "" : raw.Owner.Value,
          ControlFlags = raw.ControlFlags.ToString(),
          ControlFlagsNumeric = (int)raw.ControlFlags,
          DaclProtected = (raw.ControlFlags & ControlFlags.DiscretionaryAclProtected) != 0,
          RawAceCount = raw.DiscretionaryAcl == null ? 0 : raw.DiscretionaryAcl.Count
        };
        var aces = new List<RawAceResult>();
        if (raw.DiscretionaryAcl != null) {
          for (int index = 0; index < raw.DiscretionaryAcl.Count; index++) {
            GenericAce generic = raw.DiscretionaryAcl[index];
            CommonAce common = generic as CommonAce;
            AceFlags inheritance = generic.AceFlags &
              (AceFlags.ObjectInherit | AceFlags.ContainerInherit);
            AceFlags propagation = generic.AceFlags &
              (AceFlags.NoPropagateInherit | AceFlags.InheritOnly);
            aces.Add(new RawAceResult {
              Index = index,
              Sid = common == null || common.SecurityIdentifier == null ? "" : common.SecurityIdentifier.Value,
              AceType = generic.AceType.ToString(),
              AceTypeNumeric = (int)generic.AceType,
              AceQualifier = common == null ? "" : common.AceQualifier.ToString(),
              AccessMask = common == null ? 0L : (long)(uint)common.AccessMask,
              AceFlags = generic.AceFlags.ToString(),
              AceFlagsNumeric = (int)generic.AceFlags,
              InheritanceFlags = inheritance.ToString(),
              InheritanceFlagsNumeric = (int)inheritance,
              PropagationFlags = propagation.ToString(),
              PropagationFlagsNumeric = (int)propagation,
              IsInherited = (generic.AceFlags & AceFlags.Inherited) != 0
            });
          }
        }
        output.Aces = aces.ToArray();
        return output;
      } finally {
        if (descriptor != IntPtr.Zero) LocalFree(descriptor);
      }
    }
  }
}
'@
}

function Get-Rr17Descriptor {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("root", "file")][string]$Kind
    )
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (($Kind -eq "root" -and -not ($item -is [IO.DirectoryInfo])) -or
        ($Kind -eq "file" -and -not ($item -is [IO.FileInfo])) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "RR17 descriptor subject type or reparse state differs"
    }
    $acl = Get-Acl -LiteralPath $item.FullName -ErrorAction Stop
    $managedOwner = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
    $managedRules = @($acl.GetAccessRules(
        $true, $true, [Security.Principal.SecurityIdentifier]
    ))
    $managedAces = @()
    for ($index = 0; $index -lt $managedRules.Count; $index++) {
        $rule = $managedRules[$index]
        $managedAces += [pscustomobject][ordered]@{
            index = $index
            sid = [string]$rule.IdentityReference.Value
            access_control_type = [string]$rule.AccessControlType
            access_control_type_numeric = [int]$rule.AccessControlType
            file_system_rights = [string]$rule.FileSystemRights
            access_mask = [long]$rule.FileSystemRights
            inheritance_flags = [string]$rule.InheritanceFlags
            inheritance_flags_numeric = [int]$rule.InheritanceFlags
            propagation_flags = [string]$rule.PropagationFlags
            propagation_flags_numeric = [int]$rule.PropagationFlags
            is_inherited = [bool]$rule.IsInherited
        }
    }
    $native = [Via000R3Rr17.NativeDescriptor]::Query($item.FullName)
    $rawAces = @()
    foreach ($ace in @($native.Aces)) {
        $rawAces += [pscustomobject][ordered]@{
            index = [int]$ace.Index
            sid = [string]$ace.Sid
            ace_type = [string]$ace.AceType
            ace_type_numeric = [int]$ace.AceTypeNumeric
            ace_qualifier = [string]$ace.AceQualifier
            access_mask = [long]$ace.AccessMask
            ace_flags = [string]$ace.AceFlags
            ace_flags_numeric = [int]$ace.AceFlagsNumeric
            inheritance_flags = [string]$ace.InheritanceFlags
            inheritance_flags_numeric = [int]$ace.InheritanceFlagsNumeric
            propagation_flags = [string]$ace.PropagationFlags
            propagation_flags_numeric = [int]$ace.PropagationFlagsNumeric
            is_inherited = [bool]$ace.IsInherited
        }
    }
    $label = [Via000R3.NativeContainment]::GetMandatoryLabel($item.FullName)
    return [pscustomobject][ordered]@{
        kind = $Kind
        managed_owner_sid = $managedOwner
        native_owner_sid = [string]$native.OwnerSid
        control_flags = [string]$native.ControlFlags
        control_flags_numeric = [int]$native.ControlFlagsNumeric
        dacl_protected = [bool]$native.DaclProtected
        managed_ace_count = $managedRules.Count
        raw_ace_count = [int]$native.RawAceCount
        managed_aces = @($managedAces)
        raw_aces = @($rawAces)
        mandatory_label = [pscustomobject][ordered]@{
            sid = [string]$label.Sid
            policy_mask = [uint32]$label.PolicyMask
            ace_flags = [byte]$label.AceFlags
            ace_count = [uint32]$label.AceCount
        }
    }
}

function Assert-Rr17Subject {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("root", "file")][string]$Kind,
        [string]$ExpectedSha256 = ""
    )
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (($Kind -eq "root" -and -not ($item -is [IO.DirectoryInfo])) -or
        ($Kind -eq "file" -and -not ($item -is [IO.FileInfo])) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "RR17 subject is not ordinary and non-reparse"
    }
    if ($Kind -eq "file") {
        $streams = @(Get-Item -LiteralPath $item.FullName -Stream * -ErrorAction Stop)
        $fsutil = "C:\Windows\System32\fsutil.exe"
        $links = @(& $fsutil hardlink list $item.FullName)
        if ($LASTEXITCODE -ne 0 -or $streams.Count -ne 1 -or
            [string]$streams[0].Stream -cne ':$DATA' -or $links.Count -ne 1 -or
            $ExpectedSha256 -cnotmatch '^[0-9a-f]{64}$' -or
            (Get-Via000Sha256 -Path $item.FullName) -cne $ExpectedSha256) {
            throw "RR17 envelope stream, link, or hash identity differs"
        }
    }
}

function Assert-Rr17MediumLabel {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Descriptor,
        [Parameter(Mandatory = $true)][ValidateSet("root", "file")][string]$Kind
    )
    $expectedFlags = if ($Kind -eq "root") { 3 } else { 16 }
    if ($Descriptor.mandatory_label.ace_count -ne 1 -or
        [string]$Descriptor.mandatory_label.sid -cne "S-1-16-8192" -or
        [uint32]$Descriptor.mandatory_label.policy_mask -ne 1 -or
        [byte]$Descriptor.mandatory_label.ace_flags -ne $expectedFlags) {
        throw "RR17 exact medium NO_WRITE_UP label differs"
    }
}

$diagnosticRoot = Join-Path $RunnerTemp "via000-r3-rr17-windows-export-acl-$RunId-$RunAttempt"
$mutableRoot = Join-Path $diagnosticRoot "mutable"
$trustedRoot = Join-Path $diagnosticRoot "trusted"
$exportRoot = Join-Path $diagnosticRoot "export"
$envelopePath = Join-Path $exportRoot "envelope.json"
$evidencePath = Join-Path $trustedRoot "descriptor-evidence.json"
$expectedEnvelopeBytes = [Text.UTF8Encoding]::new($false).GetBytes("{`"rr17`":`"descriptor-diagnostic`",`"safe`":true}`n")
$expectedEnvelopeSha = ([BitConverter]::ToString(
    [Security.Cryptography.SHA256]::HashData($expectedEnvelopeBytes)
)).Replace('-', '').ToLowerInvariant()
$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value

if ($Mode -eq "capture") {
    if (Test-Path -LiteralPath $diagnosticRoot) { throw "RR17 diagnostic root is not fresh" }
    foreach ($path in @($diagnosticRoot, $mutableRoot, $trustedRoot)) {
        New-Item -ItemType Directory -Path $path -ErrorAction Stop | Out-Null
    }
    $tools = @{}
    Test-Via000ContainmentAvailability -PlatformFamily windows-x86_64 -SystemTools $tools
    Set-Via000RootIntegrity -Path $mutableRoot -Kind mutable -SystemTools $tools
    Set-Via000RootIntegrity -Path $trustedRoot -Kind protected -SystemTools $tools
    $closure = @{
        $helperPath = Get-Via000Sha256 -Path $helperPath
        $attackPath = Get-Via000Sha256 -Path $attackPath
        $PowerShellPath = Get-Via000Sha256 -Path $PowerShellPath
    }

    $bootstrapResultPath = Join-Path $trustedRoot "bootstrap-result.json"
    Invoke-Via000ContainedCommand -Label "rr17-bootstrap" `
        -ContractId "rr17-windows-export-descriptor-diagnostic" `
        -PlatformFamily windows-x86_64 -FilePath $PowerShellPath `
        -Arguments @("-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "exit 0") `
        -WorkingDirectory $mutableRoot -MutableRoot $mutableRoot -TrustedRoot $trustedRoot `
        -StdoutPath (Join-Path $trustedRoot "bootstrap-stdout.txt") `
        -StderrPath (Join-Path $trustedRoot "bootstrap-stderr.txt") `
        -ResultPath $bootstrapResultPath -Environment @{} -Closure $closure `
        -SystemTools $tools -TimeoutSeconds 30
    $bootstrap = Get-Content -LiteralPath $bootstrapResultPath -Raw | ConvertFrom-Json
    if ($bootstrap.descendants_quiescent -ne $true -or
        [int]$bootstrap.active_processes_after_teardown -ne 0 -or
        [string]$bootstrap.token_integrity_sid -cne "S-1-16-4096" -or
        [int]$bootstrap.exit_code -ne 0) {
        throw "RR17 bootstrap production containment did not tear down exactly"
    }

    New-Item -ItemType Directory -Path $exportRoot -ErrorAction Stop | Out-Null
    $setterOutcome = "accepted"
    try {
        $null = Set-Via000WindowsExportSecurity -Path $exportRoot
    } catch {
        $setterOutcome = [string]$_.Exception.Message
        if ($setterOutcome -cne "Windows export owner or protected DACL cardinality differs") {
            throw
        }
    }
    [IO.File]::WriteAllBytes($envelopePath, $expectedEnvelopeBytes)
    Assert-Rr17Subject -Path $exportRoot -Kind root
    Assert-Rr17Subject -Path $envelopePath -Kind file -ExpectedSha256 $expectedEnvelopeSha
    if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($envelopePath)) -cne
        [Convert]::ToBase64String($expectedEnvelopeBytes)) {
        throw "RR17 trusted runner could not read exact envelope bytes"
    }
    $trustedProbe = Join-Path $exportRoot "trusted-probe.txt"
    [IO.File]::WriteAllText($trustedProbe, "trusted", [Text.UTF8Encoding]::new($false))
    if ([IO.File]::ReadAllText($trustedProbe) -cne "trusted") {
        throw "RR17 trusted runner write/read differs"
    }
    Remove-Item -LiteralPath $trustedProbe -Force -ErrorAction Stop

    $rootBefore = Get-Rr17Descriptor -Path $exportRoot -Kind root
    $fileBefore = Get-Rr17Descriptor -Path $envelopePath -Kind file
    Assert-Rr17MediumLabel -Descriptor $rootBefore -Kind root
    Assert-Rr17MediumLabel -Descriptor $fileBefore -Kind file

    $attackResultPath = Join-Path $trustedRoot "attack-result.json"
    Invoke-Via000ContainedCommand -Label "rr17-export-attack" `
        -ContractId "rr17-windows-export-descriptor-diagnostic" `
        -PlatformFamily windows-x86_64 -FilePath $PowerShellPath `
        -Arguments @(
            "-NoLogo", "-NoProfile", "-NonInteractive", "-File", $attackPath,
            "-MutableRoot", $mutableRoot, "-ExportRoot", $exportRoot,
            "-EnvelopePath", $envelopePath
        ) -WorkingDirectory $mutableRoot -MutableRoot $mutableRoot -TrustedRoot $trustedRoot `
        -StdoutPath (Join-Path $trustedRoot "attack-stdout.txt") `
        -StderrPath (Join-Path $trustedRoot "attack-stderr.txt") `
        -ResultPath $attackResultPath -Environment @{} -Closure ($closure + @{
            $envelopePath = $expectedEnvelopeSha
        }) -SystemTools $tools -TimeoutSeconds 30
    $attackResult = Get-Content -LiteralPath $attackResultPath -Raw | ConvertFrom-Json
    if ($attackResult.descendants_quiescent -ne $true -or
        [int]$attackResult.active_processes_after_teardown -ne 0 -or
        [string]$attackResult.token_integrity_sid -cne "S-1-16-4096" -or
        [int]$attackResult.exit_code -ne 0 -or
        -not (Test-Path -LiteralPath (Join-Path $mutableRoot "mutable-write-allowed") -PathType Leaf)) {
        throw "RR17 low-integrity production attack containment differs"
    }
    $attackNames = @("create", "write", "hardlink", "reparse", "rename", "delete", "replace")
    foreach ($name in $attackNames) {
        if (Test-Path -LiteralPath (Join-Path $mutableRoot "$name-succeeded")) {
            throw "RR17 low-integrity $name attack crossed the export boundary"
        }
    }
    $remainingExportItems = @(Get-ChildItem -LiteralPath $exportRoot -Force -ErrorAction Stop)
    if ($remainingExportItems.Count -ne 1 -or $remainingExportItems[0].Name -cne "envelope.json") {
        throw "RR17 export root retained an unexpected attack subject"
    }
    Assert-Rr17Subject -Path $envelopePath -Kind file -ExpectedSha256 $expectedEnvelopeSha
    $rootAfter = Get-Rr17Descriptor -Path $exportRoot -Kind root
    $fileAfter = Get-Rr17Descriptor -Path $envelopePath -Kind file
    if (($rootBefore | ConvertTo-Json -Depth 12 -Compress) -cne
            ($rootAfter | ConvertTo-Json -Depth 12 -Compress) -or
        ($fileBefore | ConvertTo-Json -Depth 12 -Compress) -cne
            ($fileAfter | ConvertTo-Json -Depth 12 -Compress)) {
        throw "RR17 descriptor changed during the low-integrity attack"
    }

    $evidence = [pscustomobject][ordered]@{
        schema_version = 1
        diagnostic_id = "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC"
        non_authoritative = $true
        source_sha = $SourceSha
        run_id = $RunId
        run_attempt = $RunAttempt
        current_user_sid = $currentSid
        setter_outcome = $setterOutcome
        helper_sha256 = Get-Via000Sha256 -Path $helperPath
        attack_sha256 = Get-Via000Sha256 -Path $attackPath
        envelope_sha256 = $expectedEnvelopeSha
        bootstrap_descendants_quiescent = $true
        bootstrap_active_processes_after_teardown = 0
        attack_descendants_quiescent = $true
        attack_active_processes_after_teardown = 0
        attack_token_integrity_sid = [string]$attackResult.token_integrity_sid
        attacks_denied = @($attackNames)
        mutable_write_allowed = $true
        root = $rootAfter
        file = $fileAfter
    }
    $evidence | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $evidencePath -Encoding utf8NoBOM
    Write-Host ("RR17_DESCRIPTOR_EVIDENCE=" + ($evidence | ConvertTo-Json -Depth 12 -Compress))
    return
}

$saved = Get-Content -LiteralPath $evidencePath -Raw -ErrorAction Stop | ConvertFrom-Json
if ([string]$saved.source_sha -cne $SourceSha -or [string]$saved.run_id -cne $RunId -or
    [string]$saved.run_attempt -cne $RunAttempt -or
    [string]$saved.current_user_sid -cne $currentSid -or
    [string]$saved.envelope_sha256 -cne $expectedEnvelopeSha) {
    throw "RR17 cross-step identity differs"
}
Assert-Rr17Subject -Path $exportRoot -Kind root
Assert-Rr17Subject -Path $envelopePath -Kind file -ExpectedSha256 $expectedEnvelopeSha
$rootRecheck = Get-Rr17Descriptor -Path $exportRoot -Kind root
$fileRecheck = Get-Rr17Descriptor -Path $envelopePath -Kind file
Assert-Rr17MediumLabel -Descriptor $rootRecheck -Kind root
Assert-Rr17MediumLabel -Descriptor $fileRecheck -Kind file
if (($saved.root | ConvertTo-Json -Depth 12 -Compress) -cne
        ($rootRecheck | ConvertTo-Json -Depth 12 -Compress) -or
    ($saved.file | ConvertTo-Json -Depth 12 -Compress) -cne
        ($fileRecheck | ConvertTo-Json -Depth 12 -Compress)) {
    throw "RR17 cross-step descriptor requery differs"
}
$expectedItems = @(Get-ChildItem -LiteralPath $exportRoot -Force -ErrorAction Stop)
if ($expectedItems.Count -ne 1 -or $expectedItems[0].Name -cne "envelope.json") {
    throw "RR17 cross-step export root contains an unexpected subject"
}
Write-Host ("RR17_DESCRIPTOR_RECHECK=" + ([pscustomobject][ordered]@{
    source_sha = $SourceSha
    run_id = $RunId
    run_attempt = $RunAttempt
    envelope_sha256 = $expectedEnvelopeSha
    descriptor_unchanged = $true
    low_integrity_attacks_denied = $true
    teardown_zero_active = $true
} | ConvertTo-Json -Compress))
