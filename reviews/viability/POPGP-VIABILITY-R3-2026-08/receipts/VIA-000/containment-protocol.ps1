Set-StrictMode -Version Latest

$script:Via000BlockedEnvironmentPatterns = @(
    "GIT_*", "PIP_*", "UV_*", "TEX*", "TEXMF*", "KPATHSEA*",
    "FONTCONFIG*", "LD_*", "DYLD_*", "GITHUB_*", "ACTIONS_*", "RUNNER_*"
)
$script:Via000BlockedEnvironmentNames = @(
    "PATH", "PATHEXT", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    "GITHUB_ENV", "GITHUB_WORKSPACE", "RUNNER_TEMP", "RUNNER_WORKSPACE",
    "BASH_ENV", "ENV", "TMP", "TEMP", "TMPDIR", "HOME", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"
)

function Test-Via000RegularFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not [IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Description path is not absolute"
    }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not ($item -is [IO.FileInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$Description is not one regular non-reparse file"
    }
    $cursor = $item.Directory
    while ($null -ne $cursor) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "$Description has reparse ancestry"
        }
        $cursor = $cursor.Parent
    }
    return $item.FullName
}

function Get-Via000Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Write-Via000CanonicalJsonObject {
    param(
        [Parameter(Mandatory = $true)][Collections.IDictionary]$Document,
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$ReplaceExisting
    )
    if ($null -eq $Document -or -not [IO.Path]::IsPathFullyQualified($Path)) {
        throw "canonical JSON requires one object and an absolute path"
    }
    $exists = Test-Path -LiteralPath $Path
    if ($ReplaceExisting) {
        if (-not $exists) { throw "canonical JSON replacement target is absent" }
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        if (-not ($item -is [IO.FileInfo]) -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "canonical JSON replacement target is not one regular file"
        }
    } elseif ($exists) {
        throw "canonical JSON create target already exists"
    }

    $options = [Text.Json.JsonSerializerOptions]::new()
    $options.WriteIndented = $false
    $payload = [Text.Json.JsonSerializer]::SerializeToUtf8Bytes([object]$Document, $options)
    if ($payload.Length -lt 2 -or $payload[0] -ne 0x7b -or
        $payload[$payload.Length - 1] -ne 0x7d) {
        throw "canonical JSON serialization is not one object"
    }
    foreach ($value in $payload) {
        if ($value -lt 0x20 -or $value -eq 0xef) {
            throw "canonical JSON serialization contains a forbidden raw control or BOM byte"
        }
    }
    $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
    $text = $strictUtf8.GetString($payload)
    $parsed = [Text.Json.JsonDocument]::Parse($text)
    try {
        if ($parsed.RootElement.ValueKind -ne [Text.Json.JsonValueKind]::Object) {
            throw "canonical JSON serialization is not an object"
        }
        $canonical = [Text.Json.JsonSerializer]::SerializeToUtf8Bytes(
            $parsed.RootElement, $options
        )
    } finally {
        $parsed.Dispose()
    }
    if ($canonical.Length -ne $payload.Length) {
        throw "canonical JSON serialization is not stable"
    }
    for ($index = 0; $index -lt $payload.Length; $index++) {
        if ($canonical[$index] -ne $payload[$index]) {
            throw "canonical JSON serialization is not stable"
        }
    }

    $bytes = [byte[]]::new($payload.Length + 1)
    [Array]::Copy($payload, $bytes, $payload.Length)
    $bytes[$bytes.Length - 1] = 0x0a
    $mode = if ($ReplaceExisting) { [IO.FileMode]::Open } else { [IO.FileMode]::CreateNew }
    $stream = [IO.FileStream]::new(
        $Path, $mode, [IO.FileAccess]::Write, [IO.FileShare]::None, 4096,
        [IO.FileOptions]::WriteThrough
    )
    try {
        if ($ReplaceExisting) { $stream.SetLength(0) }
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally {
        $stream.Dispose()
    }
    $readBack = [IO.File]::ReadAllBytes($Path)
    if ($readBack.Length -ne $bytes.Length) {
        throw "canonical JSON read-back length differs"
    }
    for ($index = 0; $index -lt $bytes.Length; $index++) {
        if ($readBack[$index] -ne $bytes[$index]) {
            throw "canonical JSON read-back bytes differ"
        }
    }
    $expectedHash = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($bytes)
    ).ToLowerInvariant()
    $observedHash = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($readBack)
    ).ToLowerInvariant()
    if ($observedHash -cne $expectedHash) {
        throw "canonical JSON read-back hash differs"
    }
}

function Assert-Via000Closure {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Closure,
        [Parameter(Mandatory = $true)][string]$MutableRoot,
        [Parameter(Mandatory = $true)][string]$Moment
    )
    $comparison = if ($IsWindows) {
        [StringComparison]::OrdinalIgnoreCase
    } else {
        [StringComparison]::Ordinal
    }
    $mutable = [IO.Path]::GetFullPath($MutableRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) +
        [IO.Path]::DirectorySeparatorChar
    foreach ($entry in $Closure.GetEnumerator()) {
        $path = Test-Via000RegularFile -Path ([string]$entry.Key) -Description "closure file"
        $full = [IO.Path]::GetFullPath($path)
        if ($full.StartsWith($mutable, $comparison)) {
            throw "trusted closure file is inside the untrusted mutable root"
        }
        $expected = [string]$entry.Value
        if ($expected -cnotmatch '^[0-9a-f]{64}$') {
            throw "trusted closure contains a malformed SHA-256"
        }
        if ((Get-Via000Sha256 -Path $full) -cne $expected) {
            throw "trusted closure changed $Moment containment: $full"
        }
    }
}

function ConvertTo-Via000CleanEnvironment {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Additional,
        [Parameter(Mandatory = $true)][string]$MutableTemp
    )
    $clean = [ordered]@{}
    foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $name = [string]$entry.Key
        $blocked = $name -in $script:Via000BlockedEnvironmentNames
        foreach ($pattern in $script:Via000BlockedEnvironmentPatterns) {
            if ($name -like $pattern) { $blocked = $true; break }
        }
        if (-not $blocked) { $clean[$name] = [string]$entry.Value }
    }
    foreach ($entry in $Additional.GetEnumerator()) {
        $name = [string]$entry.Key
        if ($name -notmatch '^[A-Z][A-Z0-9_]{0,63}$') {
            throw "contained environment name is not canonical: $name"
        }
        if ([string]$entry.Value -match "[\x00\r\n]") {
            throw "contained environment value contains a forbidden control character"
        }
        $clean[$name] = [string]$entry.Value
    }
    foreach ($name in @("TEMP", "TMP", "TMPDIR")) { $clean[$name] = $MutableTemp }
    $clean["HOME"] = $MutableTemp
    $clean["USERPROFILE"] = $MutableTemp
    $clean["GIT_CONFIG_NOSYSTEM"] = "1"
    $clean["GIT_CONFIG_GLOBAL"] = $(if ($IsWindows) { "NUL" } else { "/dev/null" })
    $clean["GIT_NO_REPLACE_OBJECTS"] = "1"
    $clean["GIT_TERMINAL_PROMPT"] = "0"
    return $clean
}

function Initialize-Via000WindowsNative {
    if ("Via000R3.NativeContainment" -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Principal;
using System.Text;
using System.Threading;

namespace Via000R3 {
  public sealed class ContainedResult {
    public int ExitCode { get; set; }
    public bool TimedOut { get; set; }
    public uint ActiveProcessesAfterTermination { get; set; }
    public string Primitive { get; set; } = "windows-low-integrity-restricted-token-job-object";
    public string[] TokenRestrictionFlags { get; set; } = new [] { "DISABLE_MAX_PRIVILEGE" };
    public string TokenIntegritySid { get; set; } = "";
    public string[] EnabledPrivileges { get; set; } = new string[0];
    public string ProtectedLabelPolicy { get; set; } = "medium-integrity-no-write-up-no-read-up";
  }

  public sealed class MandatoryLabelResult {
    public string Sid { get; set; } = "";
    public uint PolicyMask { get; set; }
    public byte AceFlags { get; set; }
    public uint AceCount { get; set; }
  }

  public sealed class ExportAceResult {
    public byte AceType { get; set; }
    public byte AceFlags { get; set; }
    public uint AccessMask { get; set; }
    public string Sid { get; set; } = "";
  }

  public sealed class ExportDescriptorResult {
    public string OwnerSid { get; set; } = "";
    public ushort ControlFlags { get; set; }
    public bool DaclPresent { get; set; }
    public bool DaclDefaulted { get; set; }
    public bool DaclNull { get; set; }
    public uint AceCount { get; set; }
    public ExportAceResult[] Aces { get; set; } = new ExportAceResult[0];
  }

  public static class NativeContainment {
    const UInt32 TOKEN_ASSIGN_PRIMARY = 0x0001;
    const UInt32 TOKEN_DUPLICATE = 0x0002;
    const UInt32 TOKEN_QUERY = 0x0008;
    const UInt32 TOKEN_ADJUST_DEFAULT = 0x0080;
    const UInt32 TOKEN_ADJUST_SESSIONID = 0x0100;
    const UInt32 DISABLE_MAX_PRIVILEGE = 0x1;
    const UInt32 CREATE_SUSPENDED = 0x00000004;
    const UInt32 CREATE_UNICODE_ENVIRONMENT = 0x00000400;
    const UInt32 CREATE_NO_WINDOW = 0x08000000;
    const UInt32 STARTF_USESTDHANDLES = 0x00000100;
    const UInt32 GENERIC_READ = 0x80000000;
    const UInt32 GENERIC_WRITE = 0x40000000;
    const UInt32 FILE_SHARE_READ = 0x1;
    const UInt32 FILE_SHARE_WRITE = 0x2;
    const UInt32 FILE_SHARE_DELETE = 0x4;
    const UInt32 CREATE_ALWAYS = 2;
    const UInt32 OPEN_EXISTING = 3;
    const UInt32 FILE_ATTRIBUTE_NORMAL = 0x80;
    const UInt32 HANDLE_FLAG_INHERIT = 0x1;
    const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    const UInt32 JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400;
    const int JobObjectBasicAccountingInformation = 1;
    const int JobObjectExtendedLimitInformation = 9;
    const int TokenIntegrityLevel = 25;
    const int TokenPrivileges = 3;
    const UInt32 SE_GROUP_INTEGRITY = 0x20;
    const UInt32 SE_PRIVILEGE_ENABLED = 0x2;
    const UInt32 OWNER_SECURITY_INFORMATION = 0x00000001;
    const UInt32 DACL_SECURITY_INFORMATION = 0x00000004;
    const UInt32 PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000;
    const UInt32 WAIT_OBJECT_0 = 0;
    const UInt32 WAIT_TIMEOUT = 258;
    static readonly IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);

    [StructLayout(LayoutKind.Sequential)]
    struct SECURITY_ATTRIBUTES {
      public int nLength;
      public IntPtr lpSecurityDescriptor;
      [MarshalAs(UnmanagedType.Bool)] public bool bInheritHandle;
    }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct STARTUPINFO {
      public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
      public int dwX; public int dwY; public int dwXSize; public int dwYSize;
      public int dwXCountChars; public int dwYCountChars; public int dwFillAttribute;
      public int dwFlags; public short wShowWindow; public short cbReserved2;
      public IntPtr lpReserved2; public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_INFORMATION { public IntPtr hProcess; public IntPtr hThread; public uint dwProcessId; public uint dwThreadId; }
    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS {
      public UInt64 ReadOperationCount, WriteOperationCount, OtherOperationCount;
      public UInt64 ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
      public Int64 PerProcessUserTimeLimit, PerJobUserTimeLimit;
      public UInt32 LimitFlags; public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
      public UInt32 ActiveProcessLimit; public UIntPtr Affinity; public UInt32 PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
      public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
      public IO_COUNTERS IoInfo;
      public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
      public Int64 TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
      public UInt32 TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses;
    }
    [StructLayout(LayoutKind.Sequential)] struct SID_AND_ATTRIBUTES { public IntPtr Sid; public UInt32 Attributes; }
    [StructLayout(LayoutKind.Sequential)] struct TOKEN_MANDATORY_LABEL { public SID_AND_ATTRIBUTES Label; }
    [StructLayout(LayoutKind.Sequential)] struct LUID { public UInt32 LowPart; public Int32 HighPart; }
    [StructLayout(LayoutKind.Sequential)] struct LUID_AND_ATTRIBUTES { public LUID Luid; public UInt32 Attributes; }
    [StructLayout(LayoutKind.Sequential)] struct ACL_SIZE_INFORMATION { public UInt32 AceCount, AclBytesInUse, AclBytesFree; }

    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr GetCurrentProcess();
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool OpenProcessToken(IntPtr p, UInt32 access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool CreateRestrictedToken(IntPtr existing, UInt32 flags, UInt32 ds, IntPtr disable, UInt32 dp, IntPtr delPriv, UInt32 rs, IntPtr restrict, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool SetTokenInformation(IntPtr token, int cls, ref TOKEN_MANDATORY_LABEL info, int len);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetTokenInformation(IntPtr token, int cls, IntPtr info, int len, out int required);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool ConvertStringSidToSid(string value, out IntPtr sid);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool ConvertSidToStringSid(IntPtr sid, out IntPtr value);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool LookupPrivilegeName(string system, ref LUID luid, StringBuilder name, ref int length);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool ConvertStringSecurityDescriptorToSecurityDescriptor(string value, UInt32 revision, out IntPtr descriptor, out UInt32 size);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetSecurityDescriptorSacl(IntPtr descriptor, out bool present, out IntPtr sacl, out bool defaulted);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetSecurityDescriptorOwner(IntPtr descriptor, out IntPtr owner, out bool defaulted);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetSecurityDescriptorDacl(IntPtr descriptor, out bool present, out IntPtr dacl, out bool defaulted);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetSecurityDescriptorControl(IntPtr descriptor, out UInt16 control, out UInt32 revision);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern UInt32 SetNamedSecurityInfo(string name, int objectType, UInt32 information, IntPtr owner, IntPtr group, IntPtr dacl, IntPtr sacl);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern UInt32 GetNamedSecurityInfo(string name, int objectType, UInt32 information, out IntPtr owner, out IntPtr group, out IntPtr dacl, out IntPtr sacl, out IntPtr descriptor);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetAclInformation(IntPtr acl, out ACL_SIZE_INFORMATION information, UInt32 length, int informationClass);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetAce(IntPtr acl, UInt32 index, out IntPtr ace);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcessAsUser(IntPtr token, string app, StringBuilder command, IntPtr pa, IntPtr ta, bool inherit, UInt32 flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, UInt32 len);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job, int cls, out JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info, UInt32 len, IntPtr retLen);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job, UInt32 code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern UInt32 ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError=true)] static extern UInt32 WaitForSingleObject(IntPtr handle, UInt32 timeout);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process, out UInt32 code);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateFile(string name, UInt32 access, UInt32 share, ref SECURITY_ATTRIBUTES attrs, UInt32 creation, UInt32 flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetHandleInformation(IntPtr handle, UInt32 mask, UInt32 flags);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool CloseHandle(IntPtr handle);
    [DllImport("advapi32.dll", SetLastError=true)] static extern UInt32 GetLengthSid(IntPtr sid);
    [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr memory);

    static void Win32(bool ok, string operation) {
      if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error(), operation);
    }
    static string Quote(string value) {
      if (value.Length > 0 && !value.Any(c => char.IsWhiteSpace(c) || c == '"')) return value;
      var output = new StringBuilder("\""); int slashes = 0;
      foreach (char c in value) {
        if (c == '\\') { slashes++; continue; }
        if (c == '"') { output.Append('\\', slashes * 2 + 1); output.Append('"'); slashes = 0; continue; }
        output.Append('\\', slashes); slashes = 0; output.Append(c);
      }
      output.Append('\\', slashes * 2); output.Append('"'); return output.ToString();
    }
    static IntPtr EnvironmentBlock(IDictionary<string,string> environment) {
      string value = string.Join("\0", environment.OrderBy(p => p.Key, StringComparer.OrdinalIgnoreCase).Select(p => p.Key + "=" + p.Value)) + "\0\0";
      return Marshal.StringToHGlobalUni(value);
    }
    static IntPtr OutputHandle(string path) {
      var attrs = new SECURITY_ATTRIBUTES { nLength = Marshal.SizeOf<SECURITY_ATTRIBUTES>(), bInheritHandle = true };
      IntPtr handle = CreateFile(path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, ref attrs, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
      if (handle == INVALID_HANDLE_VALUE) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateFile output");
      return handle;
    }
    static IntPtr InputHandle() {
      var attrs = new SECURITY_ATTRIBUTES { nLength = Marshal.SizeOf<SECURITY_ATTRIBUTES>(), bInheritHandle = true };
      IntPtr handle = CreateFile("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, ref attrs, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
      if (handle == INVALID_HANDLE_VALUE) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateFile NUL");
      return handle;
    }
    static uint Active(IntPtr job) {
      JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info;
      Win32(QueryInformationJobObject(job, JobObjectBasicAccountingInformation, out info, (uint)Marshal.SizeOf<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(), IntPtr.Zero), "QueryInformationJobObject");
      return info.ActiveProcesses;
    }

    static IntPtr TokenInformation(IntPtr token, int cls, out int length) {
      GetTokenInformation(token, cls, IntPtr.Zero, 0, out length);
      int error = Marshal.GetLastWin32Error();
      if (length <= 0 || error != 122) throw new Win32Exception(error, "GetTokenInformation length");
      IntPtr buffer = Marshal.AllocHGlobal(length);
      try {
        Win32(GetTokenInformation(token, cls, buffer, length, out length), "GetTokenInformation");
        return buffer;
      } catch { Marshal.FreeHGlobal(buffer); throw; }
    }
    static string IntegritySid(IntPtr token) {
      IntPtr buffer = IntPtr.Zero, value = IntPtr.Zero;
      try {
        buffer = TokenInformation(token, TokenIntegrityLevel, out _);
        var label = Marshal.PtrToStructure<TOKEN_MANDATORY_LABEL>(buffer);
        Win32(ConvertSidToStringSid(label.Label.Sid, out value), "ConvertSidToStringSid");
        return Marshal.PtrToStringUni(value);
      } finally {
        if (value != IntPtr.Zero) LocalFree(value);
        if (buffer != IntPtr.Zero) Marshal.FreeHGlobal(buffer);
      }
    }
    static string PrivilegeName(LUID luid) {
      int length = 0;
      LookupPrivilegeName(null, ref luid, null, ref length);
      int error = Marshal.GetLastWin32Error();
      if (length <= 0 || error != 122) throw new Win32Exception(error, "LookupPrivilegeName length");
      var name = new StringBuilder(length + 1);
      Win32(LookupPrivilegeName(null, ref luid, name, ref length), "LookupPrivilegeName");
      return name.ToString();
    }
    static string[] EnabledPrivilegeNames(IntPtr token) {
      IntPtr buffer = IntPtr.Zero;
      try {
        buffer = TokenInformation(token, TokenPrivileges, out int length);
        UInt32 count = unchecked((UInt32)Marshal.ReadInt32(buffer));
        int itemSize = Marshal.SizeOf<LUID_AND_ATTRIBUTES>();
        if (count > 1024 || 4L + (long)count * itemSize > length)
          throw new InvalidOperationException("token privilege buffer is malformed");
        var names = new List<string>();
        for (int index = 0; index < count; index++) {
          IntPtr item = IntPtr.Add(buffer, 4 + index * itemSize);
          var privilege = Marshal.PtrToStructure<LUID_AND_ATTRIBUTES>(item);
          if ((privilege.Attributes & SE_PRIVILEGE_ENABLED) != 0)
            names.Add(PrivilegeName(privilege.Luid));
        }
        names.Sort(StringComparer.Ordinal);
        return names.ToArray();
      } finally { if (buffer != IntPtr.Zero) Marshal.FreeHGlobal(buffer); }
    }

    public static void SetIntegrityLabel(string path, bool mutable, bool noReadUp) {
      IntPtr descriptor = IntPtr.Zero;
      try {
        string sddl = mutable ? "S:(ML;OICI;NW;;;LW)" :
          (noReadUp ? "S:(ML;OICI;NWNR;;;ME)" : "S:(ML;OICI;NW;;;ME)");
        Win32(ConvertStringSecurityDescriptorToSecurityDescriptor(sddl, 1, out descriptor, out _), "ConvertStringSecurityDescriptorToSecurityDescriptor");
        Win32(GetSecurityDescriptorSacl(descriptor, out bool present, out IntPtr sacl, out _), "GetSecurityDescriptorSacl");
        if (!present || sacl == IntPtr.Zero) throw new InvalidOperationException("mandatory label SACL is absent");
        UInt32 status = SetNamedSecurityInfo(path, 1, 0x10, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, sacl);
        if (status != 0) throw new Win32Exception((int)status, "SetNamedSecurityInfo mandatory label");
      } finally { if (descriptor != IntPtr.Zero) LocalFree(descriptor); }
    }

    public static MandatoryLabelResult GetMandatoryLabel(string path) {
      IntPtr descriptor = IntPtr.Zero;
      try {
        UInt32 status = GetNamedSecurityInfo(path, 1, 0x10, out _, out _, out _, out IntPtr sacl, out descriptor);
        if (status != 0) throw new Win32Exception((int)status, "GetNamedSecurityInfo mandatory label");
        if (descriptor == IntPtr.Zero || sacl == IntPtr.Zero)
          throw new InvalidOperationException("mandatory label SACL is absent");
        Win32(GetAclInformation(sacl, out ACL_SIZE_INFORMATION info, (uint)Marshal.SizeOf<ACL_SIZE_INFORMATION>(), 2), "GetAclInformation mandatory label");
        if (info.AceCount != 1) throw new InvalidOperationException("mandatory label SACL does not contain exactly one ACE");
        Win32(GetAce(sacl, 0, out IntPtr ace), "GetAce mandatory label");
        if (ace == IntPtr.Zero || Marshal.ReadByte(ace, 0) != 0x11)
          throw new InvalidOperationException("mandatory label ACE type differs");
        return new MandatoryLabelResult {
          Sid = new SecurityIdentifier(IntPtr.Add(ace, 8)).Value,
          PolicyMask = unchecked((uint)Marshal.ReadInt32(ace, 4)),
          AceFlags = Marshal.ReadByte(ace, 1),
          AceCount = info.AceCount
        };
      } finally { if (descriptor != IntPtr.Zero) LocalFree(descriptor); }
    }

    public static void SetExportRootDescriptor(string path, string runnerSid) {
      IntPtr descriptor = IntPtr.Zero;
      try {
        string sddl = "O:" + runnerSid + "D:P(A;OICI;FA;;;" + runnerSid + ")";
        Win32(ConvertStringSecurityDescriptorToSecurityDescriptor(sddl, 1, out descriptor, out _), "ConvertStringSecurityDescriptorToSecurityDescriptor export root");
        Win32(GetSecurityDescriptorOwner(descriptor, out IntPtr owner, out _), "GetSecurityDescriptorOwner export root");
        Win32(GetSecurityDescriptorDacl(descriptor, out bool present, out IntPtr dacl, out bool defaulted), "GetSecurityDescriptorDacl export root");
        if (owner == IntPtr.Zero || !present || defaulted || dacl == IntPtr.Zero)
          throw new InvalidOperationException("constructed export root owner or DACL is absent/defaulted/null");
        UInt32 status = SetNamedSecurityInfo(
          path, 1, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION |
          PROTECTED_DACL_SECURITY_INFORMATION,
          owner, IntPtr.Zero, dacl, IntPtr.Zero);
        if (status != 0) throw new Win32Exception((int)status, "SetNamedSecurityInfo export root owner/protected DACL");
      } finally { if (descriptor != IntPtr.Zero) LocalFree(descriptor); }
    }

    public static void SetExportFileOwner(string path, string runnerSid) {
      IntPtr sid = IntPtr.Zero;
      try {
        Win32(ConvertStringSidToSid(runnerSid, out sid), "ConvertStringSidToSid export file owner");
        UInt32 status = SetNamedSecurityInfo(
          path, 1, OWNER_SECURITY_INFORMATION, sid, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
        if (status != 0) throw new Win32Exception((int)status, "SetNamedSecurityInfo export file owner");
      } finally { if (sid != IntPtr.Zero) LocalFree(sid); }
    }

    public static ExportDescriptorResult GetExportDescriptor(string path) {
      IntPtr descriptor = IntPtr.Zero;
      try {
        UInt32 status = GetNamedSecurityInfo(
          path, 1, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
          out _, out _, out _, out _, out descriptor);
        if (status != 0) throw new Win32Exception((int)status, "GetNamedSecurityInfo export owner/DACL");
        if (descriptor == IntPtr.Zero)
          throw new InvalidOperationException("export security descriptor is absent");
        Win32(GetSecurityDescriptorOwner(descriptor, out IntPtr owner, out _), "GetSecurityDescriptorOwner export query");
        Win32(GetSecurityDescriptorDacl(descriptor, out bool present, out IntPtr dacl, out bool defaulted), "GetSecurityDescriptorDacl export query");
        Win32(GetSecurityDescriptorControl(descriptor, out UInt16 control, out _), "GetSecurityDescriptorControl export query");
        var result = new ExportDescriptorResult {
          OwnerSid = owner == IntPtr.Zero ? "" : new SecurityIdentifier(owner).Value,
          ControlFlags = control,
          DaclPresent = present,
          DaclDefaulted = defaulted,
          DaclNull = dacl == IntPtr.Zero
        };
        if (!present || dacl == IntPtr.Zero) return result;
        Win32(GetAclInformation(dacl, out ACL_SIZE_INFORMATION info, (uint)Marshal.SizeOf<ACL_SIZE_INFORMATION>(), 2), "GetAclInformation export DACL");
        result.AceCount = info.AceCount;
        var aces = new List<ExportAceResult>();
        for (UInt32 index = 0; index < info.AceCount; index++) {
          Win32(GetAce(dacl, index, out IntPtr ace), "GetAce export DACL");
          if (ace == IntPtr.Zero) throw new InvalidOperationException("export DACL ACE is absent");
          byte type = Marshal.ReadByte(ace, 0);
          aces.Add(new ExportAceResult {
            AceType = type,
            AceFlags = Marshal.ReadByte(ace, 1),
            AccessMask = unchecked((uint)Marshal.ReadInt32(ace, 4)),
            Sid = (type == 0x00 || type == 0x01) ? new SecurityIdentifier(IntPtr.Add(ace, 8)).Value : ""
          });
        }
        result.Aces = aces.ToArray();
        return result;
      } finally { if (descriptor != IntPtr.Zero) LocalFree(descriptor); }
    }

    public static ContainedResult Run(string executable, string[] arguments, string workingDirectory, IDictionary<string,string> environment, string stdoutPath, string stderrPath, int timeoutSeconds) {
      IntPtr current = IntPtr.Zero, restricted = IntPtr.Zero, lowSid = IntPtr.Zero, env = IntPtr.Zero;
      IntPtr job = IntPtr.Zero, stdout = IntPtr.Zero, stderr = IntPtr.Zero, stdin = IntPtr.Zero;
      PROCESS_INFORMATION pi = new PROCESS_INFORMATION(); bool created = false;
      try {
        Win32(OpenProcessToken(GetCurrentProcess(), TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT | TOKEN_ADJUST_SESSIONID, out current), "OpenProcessToken");
        Win32(CreateRestrictedToken(current, DISABLE_MAX_PRIVILEGE, 0, IntPtr.Zero, 0, IntPtr.Zero, 0, IntPtr.Zero, out restricted), "CreateRestrictedToken");
        Win32(ConvertStringSidToSid("S-1-16-4096", out lowSid), "ConvertStringSidToSid");
        var label = new TOKEN_MANDATORY_LABEL { Label = new SID_AND_ATTRIBUTES { Sid = lowSid, Attributes = SE_GROUP_INTEGRITY } };
        Win32(SetTokenInformation(restricted, TokenIntegrityLevel, ref label, Marshal.SizeOf<TOKEN_MANDATORY_LABEL>() + (int)GetLengthSid(lowSid)), "SetTokenInformation low integrity");
        string integritySid = IntegritySid(restricted);
        if (!String.Equals(integritySid, "S-1-16-4096", StringComparison.Ordinal))
          throw new InvalidOperationException("restricted token integrity is not exact low integrity");
        string[] enabledPrivileges = EnabledPrivilegeNames(restricted);
        if (enabledPrivileges.Length > 1 ||
            (enabledPrivileges.Length == 1 && !String.Equals(enabledPrivileges[0], "SeChangeNotifyPrivilege", StringComparison.Ordinal)))
          throw new InvalidOperationException("restricted token retained an unexpected enabled privilege");
        job = CreateJobObject(IntPtr.Zero, null); if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject");
        var limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        Win32(SetInformationJobObject(job, JobObjectExtendedLimitInformation, ref limits, (uint)Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()), "SetInformationJobObject");
        stdout = OutputHandle(stdoutPath); stderr = OutputHandle(stderrPath); stdin = InputHandle();
        var si = new STARTUPINFO { cb = Marshal.SizeOf<STARTUPINFO>(), dwFlags = (int)STARTF_USESTDHANDLES, hStdInput = stdin, hStdOutput = stdout, hStdError = stderr };
        env = EnvironmentBlock(environment);
        var command = new StringBuilder(Quote(executable)); foreach (string item in arguments) command.Append(' ').Append(Quote(item));
        Win32(CreateProcessAsUser(restricted, executable, command, IntPtr.Zero, IntPtr.Zero, true, CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, env, workingDirectory, ref si, out pi), "CreateProcessAsUser");
        created = true;
        Win32(AssignProcessToJobObject(job, pi.hProcess), "AssignProcessToJobObject before resume");
        if (ResumeThread(pi.hThread) == UInt32.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread");
        UInt32 wait = WaitForSingleObject(pi.hProcess, checked((uint)timeoutSeconds * 1000));
        bool timedOut = wait == WAIT_TIMEOUT;
        if (wait != WAIT_OBJECT_0 && wait != WAIT_TIMEOUT) throw new Win32Exception(Marshal.GetLastWin32Error(), "WaitForSingleObject");
        UInt32 exitCode = 124;
        if (!timedOut) Win32(GetExitCodeProcess(pi.hProcess, out exitCode), "GetExitCodeProcess");
        Win32(TerminateJobObject(job, 125), "TerminateJobObject complete descendant tree");
        DateTime deadline = DateTime.UtcNow.AddSeconds(15);
        while (Active(job) != 0 && DateTime.UtcNow < deadline) Thread.Sleep(25);
        uint remaining = Active(job);
        if (remaining != 0) throw new InvalidOperationException("job object retained active descendants after termination");
        return new ContainedResult {
          ExitCode = unchecked((int)exitCode), TimedOut = timedOut,
          ActiveProcessesAfterTermination = remaining, TokenIntegritySid = integritySid,
          EnabledPrivileges = enabledPrivileges
        };
      } finally {
        if (created) { if (pi.hThread != IntPtr.Zero) CloseHandle(pi.hThread); if (pi.hProcess != IntPtr.Zero) CloseHandle(pi.hProcess); }
        if (stdin != IntPtr.Zero && stdin != INVALID_HANDLE_VALUE) CloseHandle(stdin);
        if (stdout != IntPtr.Zero && stdout != INVALID_HANDLE_VALUE) CloseHandle(stdout);
        if (stderr != IntPtr.Zero && stderr != INVALID_HANDLE_VALUE) CloseHandle(stderr);
        if (job != IntPtr.Zero) CloseHandle(job);
        if (env != IntPtr.Zero) Marshal.FreeHGlobal(env);
        if (lowSid != IntPtr.Zero) LocalFree(lowSid);
        if (restricted != IntPtr.Zero) CloseHandle(restricted);
        if (current != IntPtr.Zero) CloseHandle(current);
      }
    }
  }
}
'@
}

function Test-Via000ContainmentAvailability {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("ubuntu-latest-x86_64", "windows-x86_64")]
        [string]$PlatformFamily,
        [Parameter(Mandatory = $true)][hashtable]$SystemTools
    )
    if ($PlatformFamily -eq "windows-x86_64") {
        if (-not $IsWindows) { throw "Windows containment requires a Windows host" }
        Initialize-Via000WindowsNative
        return
    }
    if (-not $IsLinux) { throw "Ubuntu containment requires a Linux host" }
    foreach ($name in @("sudo", "systemd_run", "systemctl", "useradd", "userdel", "id")) {
        if (-not $SystemTools.ContainsKey($name)) { throw "missing Ubuntu containment tool $name" }
        [void](Test-Via000RegularFile -Path ([string]$SystemTools[$name]) -Description $name)
    }
    $probe = & ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) show `
        --property=Version --value 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or -not $probe.Trim()) {
        throw "passwordless systemd control is unavailable for fail-closed containment"
    }
}

function Set-Via000RootIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("mutable", "protected", "traverse")][string]$Kind,
        [Parameter(Mandatory = $true)][hashtable]$SystemTools
    )
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not ($item -is [IO.DirectoryInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "containment root is not one regular non-reparse directory"
    }
    if ($IsWindows) {
        Initialize-Via000WindowsNative
        [Via000R3.NativeContainment]::SetIntegrityLabel(
            $item.FullName, ($Kind -eq "mutable"), ($Kind -eq "protected")
        )
        return
    }
    $mode = if ($Kind -eq "mutable") {
        [IO.UnixFileMode]::UserRead -bor [IO.UnixFileMode]::UserWrite -bor [IO.UnixFileMode]::UserExecute -bor
        [IO.UnixFileMode]::GroupRead -bor [IO.UnixFileMode]::GroupWrite -bor [IO.UnixFileMode]::GroupExecute -bor
        [IO.UnixFileMode]::OtherRead -bor [IO.UnixFileMode]::OtherWrite -bor [IO.UnixFileMode]::OtherExecute
    } elseif ($Kind -eq "protected") {
        [IO.UnixFileMode]::UserRead -bor [IO.UnixFileMode]::UserWrite -bor [IO.UnixFileMode]::UserExecute
    } else {
        [IO.UnixFileMode]::UserRead -bor [IO.UnixFileMode]::UserWrite -bor [IO.UnixFileMode]::UserExecute -bor
        [IO.UnixFileMode]::GroupExecute -bor [IO.UnixFileMode]::OtherExecute
    }
    [IO.File]::SetUnixFileMode($item.FullName, $mode)
}

function Set-Via000WindowsExportSecurity {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not $IsWindows) { throw "Windows export security requires a Windows host" }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not ($item -is [IO.DirectoryInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Windows export root is not one ordinary non-reparse directory"
    }
    $runnerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $runnerSid -or $runnerSid.Value -cnotmatch '^S-1-(?:[0-9]+-)+[0-9]+$') {
        throw "trusted Windows runner SID is unavailable"
    }
    Initialize-Via000WindowsNative
    [Via000R3.NativeContainment]::SetExportRootDescriptor(
        $item.FullName, $runnerSid.Value
    )
    [Via000R3.NativeContainment]::SetIntegrityLabel($item.FullName, $false, $false)
    return Assert-Via000WindowsExportSecurity -Path $item.FullName -Kind root
}

function Set-Via000WindowsExportFileOwner {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (-not $IsWindows) { throw "Windows export security requires a Windows host" }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not ($item -is [IO.FileInfo]) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Windows export file is not one ordinary non-reparse file"
    }
    $runnerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $runnerSid -or $runnerSid.Value -cnotmatch '^S-1-(?:[0-9]+-)+[0-9]+$') {
        throw "trusted Windows runner SID is unavailable"
    }
    Initialize-Via000WindowsNative
    [Via000R3.NativeContainment]::SetExportFileOwner(
        $item.FullName, $runnerSid.Value
    )
    return Assert-Via000WindowsExportSecurity -Path $item.FullName `
        -Kind file -ExpectedSha256 $ExpectedSha256
}

function Assert-Via000WindowsExportSecurity {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("root", "file")][string]$Kind,
        [string]$ExpectedSha256 = ""
    )
    if (-not $IsWindows) { throw "Windows export security requires a Windows host" }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (($Kind -eq "root" -and -not ($item -is [IO.DirectoryInfo])) -or
        ($Kind -eq "file" -and -not ($item -is [IO.FileInfo])) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Windows export subject type or reparse identity differs"
    }
    if ($Kind -eq "file") {
        $streams = @(Get-Item -LiteralPath $item.FullName -Stream * -ErrorAction Stop)
        if ($streams.Count -ne 1 -or [string]$streams[0].Stream -cne ':$DATA') {
            throw "Windows export file contains an alternate data stream"
        }
        if ($ExpectedSha256 -cnotmatch '^[0-9a-f]{64}$' -or
            (Get-Via000Sha256 -Path $item.FullName) -cne $ExpectedSha256) {
            throw "Windows export file hash differs"
        }
    } elseif ($ExpectedSha256) {
        throw "Windows export root cannot declare a file hash"
    }

    $runnerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    Initialize-Via000WindowsNative
    $native = [Via000R3.NativeContainment]::GetExportDescriptor($item.FullName)
    $acl = Get-Acl -LiteralPath $item.FullName -ErrorAction Stop
    $ownerSid = $acl.GetOwner([Security.Principal.SecurityIdentifier])
    $rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
    $raw = [Security.AccessControl.RawSecurityDescriptor]::new(
        $acl.GetSecurityDescriptorBinaryForm(), 0
    )
    $expectedInheritance = if ($Kind -eq "root") {
        [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
            [Security.AccessControl.InheritanceFlags]::ObjectInherit
    } else { [Security.AccessControl.InheritanceFlags]::None }
    $expectedInherited = $Kind -eq "file"
    $expectedProtected = $Kind -eq "root"
    $expectedAceFlags = if ($Kind -eq "root") { [byte]3 } else { [byte]16 }
    $expectedControlFlags = if ($Kind -eq "root") { [uint16]37892 } else { [uint16]33796 }
    if ($null -eq $runnerSid -or $ownerSid.Value -cne $runnerSid.Value -or
        [string]$native.OwnerSid -cne $runnerSid.Value -or
        $acl.AreAccessRulesProtected -ne $expectedProtected -or
        [bool]$native.DaclPresent -ne $true -or [bool]$native.DaclDefaulted -ne $false -or
        [bool]$native.DaclNull -ne $false -or $rules.Count -ne 1 -or
        [uint32]$native.AceCount -ne 1 -or @($native.Aces).Count -ne 1 -or
        [uint16]$native.ControlFlags -ne $expectedControlFlags -or
        [uint16]$raw.ControlFlags -ne $expectedControlFlags) {
        throw "Windows export native/managed owner, DACL, or control flags differ"
    }
    $access = $rules[0]
    $nativeAccess = @($native.Aces)[0]
    if ($access.IdentityReference.Value -cne $runnerSid.Value -or
        $access.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
        $access.FileSystemRights -ne [Security.AccessControl.FileSystemRights]::FullControl -or
        $access.InheritanceFlags -ne $expectedInheritance -or
        $access.PropagationFlags -ne [Security.AccessControl.PropagationFlags]::None -or
        $access.IsInherited -ne $expectedInherited -or
        [string]$nativeAccess.Sid -cne $runnerSid.Value -or
        [byte]$nativeAccess.AceType -ne 0 -or
        [uint32]$nativeAccess.AccessMask -ne 2032127 -or
        [byte]$nativeAccess.AceFlags -ne $expectedAceFlags) {
        throw "Windows export DACL grants a subject other than the exact runner SID"
    }
    $label = [Via000R3.NativeContainment]::GetMandatoryLabel($item.FullName)
    if ($label.AceCount -ne 1 -or $label.Sid -cne 'S-1-16-8192' -or
        $label.PolicyMask -ne 1 -or $label.AceFlags -ne $expectedAceFlags) {
        throw "Windows export mandatory label is not exact medium NO_WRITE_UP"
    }
    return [pscustomobject][ordered]@{
        owner_sid = $runnerSid.Value
        dacl_policy = "protected-current-runner-full-control-v1"
        integrity_sid = $label.Sid
        mandatory_policy = "NO_WRITE_UP"
        inherited_file_dacl = $expectedInherited
        control_flags = [uint16]$native.ControlFlags
        dacl_protected = $expectedProtected
        native_ace_count = [uint32]$native.AceCount
        managed_ace_count = $rules.Count
    }
}

function Protect-Via000ReadOnlyClosure {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$SystemTools
    )
    if ($IsWindows) {
        Initialize-Via000WindowsNative
        foreach ($item in @(
            Get-Item -LiteralPath $Path -Force -ErrorAction Stop
            Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction Stop
        )) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "trusted closure contains a symlink or reparse point"
            }
            [Via000R3.NativeContainment]::SetIntegrityLabel($item.FullName, $false, $false)
        }
        return
    }
    foreach ($item in @(Get-ChildItem -LiteralPath $Path -Force -Recurse)) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "trusted closure contains a symlink or reparse point"
        }
        $mode = [IO.File]::GetUnixFileMode($item.FullName)
        $mode = $mode -band (-bnot (
            [IO.UnixFileMode]::UserWrite -bor [IO.UnixFileMode]::GroupWrite -bor [IO.UnixFileMode]::OtherWrite
        ))
        [IO.File]::SetUnixFileMode($item.FullName, $mode)
    }
    $rootMode = [IO.File]::GetUnixFileMode($Path)
    $rootMode = $rootMode -band (-bnot (
        [IO.UnixFileMode]::UserWrite -bor [IO.UnixFileMode]::GroupWrite -bor [IO.UnixFileMode]::OtherWrite
    ))
    [IO.File]::SetUnixFileMode($Path, $rootMode)
}

function Enable-Via000MutableClosure {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$SystemTools
    )
    foreach ($item in @(
        Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction Stop
    )) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "mutable closure contains a symlink or reparse point"
        }
        if ($IsWindows) {
            Initialize-Via000WindowsNative
            [Via000R3.NativeContainment]::SetIntegrityLabel($item.FullName, $true, $false)
        } else {
            $mode = [IO.File]::GetUnixFileMode($item.FullName) -bor
                [IO.UnixFileMode]::UserRead -bor [IO.UnixFileMode]::UserWrite -bor
                [IO.UnixFileMode]::GroupRead -bor [IO.UnixFileMode]::GroupWrite -bor
                [IO.UnixFileMode]::OtherRead -bor [IO.UnixFileMode]::OtherWrite
            if ($item -is [IO.DirectoryInfo]) {
                $mode = $mode -bor [IO.UnixFileMode]::UserExecute -bor
                    [IO.UnixFileMode]::GroupExecute -bor [IO.UnixFileMode]::OtherExecute
            }
            [IO.File]::SetUnixFileMode($item.FullName, $mode)
        }
    }
}

function Assert-Via000UidQuiescent {
    param([Parameter(Mandatory = $true)][string]$Uid)
    $uidPattern = '^Uid:\s+{0}(?:\s|$)' -f [regex]::Escape($Uid)
    foreach ($statusPath in @(Get-ChildItem -LiteralPath /proc -Directory -ErrorAction Stop |
            Where-Object { $_.Name -match '^[0-9]+$' } |
            ForEach-Object { Join-Path $_.FullName "status" })) {
        if ((Test-Path -LiteralPath $statusPath -PathType Leaf) -and
            (Get-Content -LiteralPath $statusPath -ErrorAction SilentlyContinue) -match
                $uidPattern) {
            throw "fresh untrusted service identity retained a process after teardown"
        }
    }
}

function Invoke-Via000ContainedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ContractId,
        [Parameter(Mandatory = $true)]
        [ValidateSet("ubuntu-latest-x86_64", "windows-x86_64")]
        [string]$PlatformFamily,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$MutableRoot,
        [Parameter(Mandatory = $true)][string]$TrustedRoot,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath,
        [Parameter(Mandatory = $true)][string]$ResultPath,
        [Parameter(Mandatory = $true)][hashtable]$Environment,
        [Parameter(Mandatory = $true)][hashtable]$Closure,
        [Parameter(Mandatory = $true)][hashtable]$SystemTools,
        [ValidateRange(1, 7200)][int]$TimeoutSeconds = 3600
    )
    foreach ($path in @($FilePath, $WorkingDirectory, $MutableRoot, $TrustedRoot, $StdoutPath, $StderrPath, $ResultPath)) {
        if (-not [IO.Path]::IsPathFullyQualified($path)) { throw "contained path is not absolute: $path" }
    }
    $executable = Test-Via000RegularFile -Path $FilePath -Description "contained executable"
    $mutable = (Get-Item -LiteralPath $MutableRoot -Force -ErrorAction Stop).FullName
    $trusted = (Get-Item -LiteralPath $TrustedRoot -Force -ErrorAction Stop).FullName
    if ($mutable -eq $trusted) { throw "mutable and trusted roots must be disjoint" }
    $stageTemp = Join-Path $mutable "temp"
    New-Item -ItemType Directory -Path $stageTemp -Force | Out-Null
    Set-Via000RootIntegrity -Path $stageTemp -Kind mutable -SystemTools $SystemTools
    $cleanEnvironment = ConvertTo-Via000CleanEnvironment -Additional $Environment -MutableTemp $stageTemp
    Assert-Via000Closure -Closure $Closure -MutableRoot $mutable -Moment "before"
    $started = [DateTimeOffset]::UtcNow
    $staging = Join-Path $mutable (".via000-" + $Label + "-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $staging | Out-Null
    Set-Via000RootIntegrity -Path $staging -Kind mutable -SystemTools $SystemTools
    $untrustedStdout = Join-Path $staging "stdout.txt"
    $untrustedStderr = Join-Path $staging "stderr.txt"
    $primitive = ""
    $tokenRestrictionFlags = @()
    $tokenIntegritySid = ""
    $enabledPrivileges = @()
    $protectedLabelPolicy = $(if ($PlatformFamily -eq "windows-x86_64") {
        "medium-integrity-no-write-up-no-read-up"
    } else { "owner-only-protected-root" })
    $quiescent = $false
    $exitCode = 125
    $timeout = $false
    $unit = ""
    $serviceUser = ""
    $serviceUserCreated = $false
    $serviceUid = ""
    try {
        if ($PlatformFamily -eq "windows-x86_64") {
            Initialize-Via000WindowsNative
            $dictionary = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
            foreach ($entry in $cleanEnvironment.GetEnumerator()) { $dictionary[[string]$entry.Key] = [string]$entry.Value }
            $result = [Via000R3.NativeContainment]::Run(
                $executable, $Arguments, $WorkingDirectory, $dictionary,
                $untrustedStdout, $untrustedStderr, $TimeoutSeconds
            )
            $exitCode = [int]$result.ExitCode
            $timeout = [bool]$result.TimedOut
            $quiescent = ([uint32]$result.ActiveProcessesAfterTermination -eq 0)
            $primitive = [string]$result.Primitive
            $tokenRestrictionFlags = @($result.TokenRestrictionFlags)
            $tokenIntegritySid = [string]$result.TokenIntegritySid
            $enabledPrivileges = @($result.EnabledPrivileges)
            if ($tokenRestrictionFlags.Count -ne 1 -or
                [string]$tokenRestrictionFlags[0] -cne "DISABLE_MAX_PRIVILEGE" -or
                $tokenIntegritySid -cne "S-1-16-4096" -or
                $enabledPrivileges.Count -gt 1 -or
                ($enabledPrivileges.Count -eq 1 -and
                    [string]$enabledPrivileges[0] -cne "SeChangeNotifyPrivilege")) {
                throw "restricted token evidence differs from the frozen policy"
            }
        } else {
            $unit = "via000-r3-$($Label.ToLowerInvariant().Replace('_','-'))-$([Guid]::NewGuid().ToString('N'))"
            $serviceUser = "via000r3$([Guid]::NewGuid().ToString('N').Substring(0,12))"
            & ([string]$SystemTools.sudo) -n ([string]$SystemTools.useradd) `
                --system --no-create-home --home-dir /nonexistent `
                --shell /usr/sbin/nologin --gid nogroup $serviceUser
            if ($LASTEXITCODE -ne 0) { throw "could not create fresh untrusted service identity" }
            $serviceUserCreated = $true
            $serviceUid = (& ([string]$SystemTools.id) -u $serviceUser 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -ne 0 -or $serviceUid -cnotmatch '^[1-9][0-9]*$') {
                throw "fresh untrusted service identity has no canonical UID"
            }
            $systemdArguments = @(
                "-n", [string]$SystemTools.systemd_run,
                "--quiet", "--wait", "--pipe", "--service-type=exec", "--unit=$unit",
                "--property=User=$serviceUser", "--property=Group=nogroup",
                "--property=KillMode=control-group",
                "--property=SendSIGKILL=yes", "--property=TimeoutStopSec=15s",
                "--property=PrivateTmp=no",
                "--property=ProtectSystem=no", "--property=ProtectHome=no",
                "--property=NoNewPrivileges=yes", "--property=RestrictSUIDSGID=yes",
                "--property=LockPersonality=yes", "--working-directory=$WorkingDirectory"
            )
            foreach ($entry in $cleanEnvironment.GetEnumerator()) {
                $systemdArguments += "--setenv=$([string]$entry.Key)=$([string]$entry.Value)"
            }
            $systemdArguments += @("--", $executable)
            $systemdArguments += $Arguments
            $process = [Diagnostics.Process]::new()
            $process.StartInfo.FileName = [string]$SystemTools.sudo
            $process.StartInfo.UseShellExecute = $false
            $process.StartInfo.RedirectStandardOutput = $true
            $process.StartInfo.RedirectStandardError = $true
            foreach ($argument in $systemdArguments) { [void]$process.StartInfo.ArgumentList.Add($argument) }
            if (-not $process.Start()) { throw "could not start transient systemd service" }
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
                $timeout = $true
                & ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) kill --kill-whom=all --signal=KILL $unit | Out-Null
                [void]$process.WaitForExit(15000)
            }
            [IO.File]::WriteAllText($untrustedStdout, $stdoutTask.GetAwaiter().GetResult(), [Text.UTF8Encoding]::new($false))
            [IO.File]::WriteAllText($untrustedStderr, $stderrTask.GetAwaiter().GetResult(), [Text.UTF8Encoding]::new($false))
            $exitCode = if ($timeout) { 124 } else { $process.ExitCode }
            $active = (& ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) show $unit --property=ActiveState --value 2>&1 | Out-String).Trim()
            $sub = (& ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) show $unit --property=SubState --value 2>&1 | Out-String).Trim()
            $controlGroup = (& ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) show $unit --property=ControlGroup --value 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -ne 0) { throw "could not query transient containment unit" }
            $cgroupProcs = if ($controlGroup) { "/sys/fs/cgroup$controlGroup/cgroup.procs" } else { "" }
            $remaining = @()
            if ($cgroupProcs -and (Test-Path -LiteralPath $cgroupProcs -PathType Leaf)) {
                $remaining = @(Get-Content -LiteralPath $cgroupProcs | Where-Object { $_ })
            }
            $quiescent = ($active -in @("inactive", "failed") -and
                $sub -in @("dead", "failed") -and $remaining.Count -eq 0)
            if ($quiescent) { Assert-Via000UidQuiescent -Uid $serviceUid }
            $primitive = "ubuntu-systemd-ephemeral-user-control-group"
            & ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) reset-failed $unit 2>$null | Out-Null
        }
        if (-not $quiescent) { throw "contained descendant tree is not proven quiescent" }
        Assert-Via000Closure -Closure $Closure -MutableRoot $mutable -Moment "after"
        foreach ($pair in @(@($untrustedStdout, $StdoutPath), @($untrustedStderr, $StderrPath))) {
            if (-not (Test-Path -LiteralPath $pair[0] -PathType Leaf)) {
                [IO.File]::WriteAllText($pair[0], "", [Text.UTF8Encoding]::new($false))
            }
            Copy-Item -LiteralPath $pair[0] -Destination $pair[1]
        }
        $finished = [DateTimeOffset]::UtcNow
        $record = [ordered]@{
            schema_version = 1
            label = $Label
            contract_id = $ContractId
            primitive = $primitive
            file = $executable
            executable = $executable
            executable_sha256 = Get-Via000Sha256 -Path $executable
            arguments = $Arguments
            working_directory = $WorkingDirectory
            started_at = $started.ToString("O")
            finished_at = $finished.ToString("O")
            duration_seconds = ($finished - $started).TotalSeconds
            direct_exit_code = $exitCode
            exit_code = $exitCode
            timed_out = $timeout
            descendants_quiescent = $quiescent
            active_processes_after_teardown = 0
            mutable_root = $mutable
            trusted_root = $trusted
            environment_leaks_scrubbed = $true
            privilege_separation = $(if ($IsWindows) {
                "low-integrity-restricted-token"
            } else {
                "systemd-ephemeral-user"
            })
            token_restriction_flags = @($tokenRestrictionFlags)
            token_integrity_sid = $tokenIntegritySid
            enabled_privilege_count = $enabledPrivileges.Count
            enabled_privileges = @($enabledPrivileges)
            protected_label_policy = $protectedLabelPolicy
            unit = $unit
            stdout_sha256 = Get-Via000Sha256 -Path $StdoutPath
            stderr_sha256 = Get-Via000Sha256 -Path $StderrPath
        }
        if ($PlatformFamily -eq "ubuntu-latest-x86_64") {
            $record["ephemeral_identity_uid"] = $serviceUid
            $record["ephemeral_identity_processes_empty"] = $true
            $record["ephemeral_identity_removed"] = $false
        }
        Write-Via000CanonicalJsonObject -Document $record -Path $ResultPath
        if ($PlatformFamily -eq "ubuntu-latest-x86_64") {
            Assert-Via000UidQuiescent -Uid $serviceUid
            & ([string]$SystemTools.sudo) -n ([string]$SystemTools.userdel) $serviceUser
            if ($LASTEXITCODE -ne 0) { throw "could not remove fresh untrusted service identity" }
            $serviceUserCreated = $false
            $record["ephemeral_identity_removed"] = $true
            Write-Via000CanonicalJsonObject -Document $record -Path $ResultPath `
                -ReplaceExisting
        }
    } finally {
        if ($serviceUserCreated) {
            if ($unit) {
                & ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) kill `
                    --kill-whom=all --signal=KILL $unit 2>$null | Out-Null
                & ([string]$SystemTools.sudo) -n ([string]$SystemTools.systemctl) reset-failed `
                    $unit 2>$null | Out-Null
            }
            & ([string]$SystemTools.sudo) -n ([string]$SystemTools.userdel) $serviceUser `
                2>$null | Out-Null
            $serviceUserCreated = $false
        }
        if (Test-Path -LiteralPath $staging) {
            Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ($timeout) { throw "contained command $Label timed out and was terminated" }
    if ($exitCode -ne 0) { throw "contained command $Label failed with exit code $exitCode" }
}
