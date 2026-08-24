Set-StrictMode -Version Latest

function Initialize-Via000Rr13NativeDiagnostic {
    if ("Via000Rr13.NativeDiagnostic" -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace Via000Rr13 {
  public sealed class DiagnosticResult {
    public int ExitCode { get; set; }
    public bool TimedOut { get; set; }
    public uint ActiveProcessesAfterTermination { get; set; }
    public string Primitive { get; set; } = "windows-low-integrity-restricted-token-job-object";
    public string Phase { get; set; } = "not-started";
    public string TokenFlags { get; set; } = "";
    public string EnvironmentConstruction { get; set; } = "";
    public string IntegritySid { get; set; } = "";
    public bool PrivilegesDisabled { get; set; }
  }

  public static class NativeDiagnostic {
    const UInt32 TOKEN_ASSIGN_PRIMARY = 0x0001;
    const UInt32 TOKEN_DUPLICATE = 0x0002;
    const UInt32 TOKEN_QUERY = 0x0008;
    const UInt32 TOKEN_ADJUST_DEFAULT = 0x0080;
    const UInt32 TOKEN_ADJUST_SESSIONID = 0x0100;
    const UInt32 DISABLE_MAX_PRIVILEGE = 0x1;
    const UInt32 LUA_TOKEN = 0x4;
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
    const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    const UInt32 JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400;
    const UInt32 SE_PRIVILEGE_ENABLED = 0x2;
    const int JobObjectBasicAccountingInformation = 1;
    const int JobObjectExtendedLimitInformation = 9;
    const int TokenPrivileges = 3;
    const int TokenIntegrityLevel = 25;
    const UInt32 SE_GROUP_INTEGRITY = 0x20;
    const UInt32 WAIT_OBJECT_0 = 0;
    const UInt32 WAIT_TIMEOUT = 258;
    const int ERROR_INSUFFICIENT_BUFFER = 122;
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

    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr GetCurrentProcess();
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool OpenProcessToken(IntPtr p, UInt32 access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool CreateRestrictedToken(IntPtr existing, UInt32 flags, UInt32 ds, IntPtr disable, UInt32 dp, IntPtr delPriv, UInt32 rs, IntPtr restrict, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool SetTokenInformation(IntPtr token, int cls, ref TOKEN_MANDATORY_LABEL info, int len);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetTokenInformation(IntPtr token, int cls, IntPtr info, int len, out int required);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool ConvertStringSidToSid(string value, out IntPtr sid);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool ConvertSidToStringSid(IntPtr sid, out IntPtr value);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool LookupPrivilegeName(string system, ref LUID luid, StringBuilder name, ref int length);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcessAsUser(IntPtr token, string app, StringBuilder command, IntPtr pa, IntPtr ta, bool inherit, UInt32 flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
    [DllImport("userenv.dll", SetLastError=true)] static extern bool CreateEnvironmentBlock(out IntPtr environment, IntPtr token, bool inherit);
    [DllImport("userenv.dll", SetLastError=true)] static extern bool DestroyEnvironmentBlock(IntPtr environment);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, UInt32 len);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job, int cls, out JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info, UInt32 len, IntPtr retLen);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job, UInt32 code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern UInt32 ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError=true)] static extern UInt32 WaitForSingleObject(IntPtr handle, UInt32 timeout);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process, out UInt32 code);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateFile(string name, UInt32 access, UInt32 share, ref SECURITY_ATTRIBUTES attrs, UInt32 creation, UInt32 flags, IntPtr template);
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
    static IntPtr SerializeEnvironment(IDictionary<string,string> environment) {
      string value = string.Join("\0", environment.OrderBy(p => p.Key, StringComparer.OrdinalIgnoreCase).Select(p => p.Key + "=" + p.Value)) + "\0\0";
      return Marshal.StringToHGlobalUni(value);
    }
    static bool BlockedName(string name) {
      string upper = name.ToUpperInvariant();
      return upper.StartsWith("GITHUB_") || upper.StartsWith("ACTIONS_") ||
        upper.StartsWith("RUNNER_") || upper.StartsWith("GIT_") ||
        upper.StartsWith("PIP_") || upper.StartsWith("UV_") ||
        upper.StartsWith("TEX") || upper.StartsWith("TEXMF") ||
        upper.StartsWith("KPATHSEA") || upper.StartsWith("FONTCONFIG") ||
        upper.StartsWith("LD_") || upper.StartsWith("DYLD_") ||
        upper.StartsWith("PYTHON") || upper == "VIRTUAL_ENV" ||
        upper.Contains("CREDENTIAL") || upper.Contains("SECRET") ||
        upper.Contains("PASSWORD") || upper.EndsWith("_TOKEN");
    }
    static Dictionary<string,string> TokenEnvironment(IntPtr token, string mutableTemp, string sentinel, string trustedPath) {
      IntPtr block = IntPtr.Zero;
      var parsed = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
      try {
        Win32(CreateEnvironmentBlock(out block, token, false), "CreateEnvironmentBlock restricted token");
        IntPtr cursor = block;
        while (true) {
          string entry = Marshal.PtrToStringUni(cursor);
          if (entry == null) throw new InvalidOperationException("token environment contains a null entry");
          if (entry.Length == 0) break;
          cursor = IntPtr.Add(cursor, checked((entry.Length + 1) * 2));
          if (entry.IndexOfAny(new[] {'\0', '\r', '\n'}) >= 0)
            throw new InvalidOperationException("token environment contains a control byte");
          if (entry[0] == '=') {
            if (!Regex.IsMatch(entry, "^=[A-Za-z]:=.*$"))
              throw new InvalidOperationException("token environment pseudo-variable is malformed");
            continue;
          }
          int split = entry.IndexOf('=');
          if (split <= 0) throw new InvalidOperationException("token environment entry is malformed");
          string name = entry.Substring(0, split);
          string value = entry.Substring(split + 1);
          if (!Regex.IsMatch(name, "^[A-Za-z_][A-Za-z0-9_()]{0,127}$") ||
              value.IndexOfAny(new[] {'\0', '\r', '\n'}) >= 0)
            throw new InvalidOperationException("token environment name or value is noncanonical");
          if (parsed.ContainsKey(name)) throw new InvalidOperationException("token environment has a duplicate name");
          if (BlockedName(name)) throw new InvalidOperationException("token environment contains a blocked name");
          parsed.Add(name, value);
        }
      } finally {
        if (block != IntPtr.Zero) Win32(DestroyEnvironmentBlock(block), "DestroyEnvironmentBlock");
      }
      string[] allow = {
        "SystemRoot", "windir", "SystemDrive", "ComSpec", "OS",
        "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "ProgramFiles", "ProgramW6432",
        "CommonProgramFiles", "CommonProgramW6432", "ProgramFiles(x86)",
        "CommonProgramFiles(x86)", "ALLUSERSPROFILE", "PUBLIC"
      };
      var selected = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
      foreach (string name in allow) if (parsed.TryGetValue(name, out string value)) selected[name] = value;
      foreach (string required in new[] {"SystemRoot", "windir", "SystemDrive", "ComSpec", "OS", "PROCESSOR_ARCHITECTURE"})
        if (!selected.ContainsKey(required)) throw new InvalidOperationException("token environment omits required reviewed essential " + required);
      string profile = System.IO.Path.Combine(mutableTemp, "profile");
      string roaming = System.IO.Path.Combine(profile, "AppData", "Roaming");
      string local = System.IO.Path.Combine(profile, "AppData", "Local");
      System.IO.Directory.CreateDirectory(roaming);
      System.IO.Directory.CreateDirectory(local);
      selected["PATH"] = trustedPath;
      selected["PATHEXT"] = ".COM;.EXE;.BAT;.CMD";
      selected["TEMP"] = mutableTemp;
      selected["TMP"] = mutableTemp;
      selected["HOME"] = profile;
      selected["USERPROFILE"] = profile;
      selected["APPDATA"] = roaming;
      selected["LOCALAPPDATA"] = local;
      selected["VIA000_SENTINEL"] = sentinel;
      return selected;
    }
    static string VerifyToken(IntPtr token) {
      int required = 0;
      if (GetTokenInformation(token, TokenIntegrityLevel, IntPtr.Zero, 0, out required) ||
          Marshal.GetLastWin32Error() != ERROR_INSUFFICIENT_BUFFER || required <= 0)
        throw new Win32Exception(Marshal.GetLastWin32Error(), "GetTokenInformation integrity size");
      IntPtr buffer = Marshal.AllocHGlobal(required);
      IntPtr sidText = IntPtr.Zero;
      try {
        Win32(GetTokenInformation(token, TokenIntegrityLevel, buffer, required, out required), "GetTokenInformation integrity");
        var label = Marshal.PtrToStructure<TOKEN_MANDATORY_LABEL>(buffer);
        Win32(ConvertSidToStringSid(label.Label.Sid, out sidText), "ConvertSidToStringSid integrity");
        string sid = Marshal.PtrToStringUni(sidText);
        if (sid != "S-1-16-4096") throw new InvalidOperationException("restricted token is not low integrity");
        return sid;
      } finally {
        if (sidText != IntPtr.Zero) LocalFree(sidText);
        Marshal.FreeHGlobal(buffer);
      }
    }
    static bool VerifyPrivilegesDisabled(IntPtr token) {
      int required = 0;
      if (GetTokenInformation(token, TokenPrivileges, IntPtr.Zero, 0, out required) ||
          Marshal.GetLastWin32Error() != ERROR_INSUFFICIENT_BUFFER || required < 4)
        throw new Win32Exception(Marshal.GetLastWin32Error(), "GetTokenInformation privileges size");
      IntPtr buffer = Marshal.AllocHGlobal(required);
      try {
        Win32(GetTokenInformation(token, TokenPrivileges, buffer, required, out required), "GetTokenInformation privileges");
        UInt32 count = unchecked((UInt32)Marshal.ReadInt32(buffer));
        if (4L + 12L * count > required) throw new InvalidOperationException("restricted privilege buffer is malformed");
        int enabledCount = 0;
        for (int index = 0; index < count; index++) {
          var luid = new LUID {
            LowPart = unchecked((UInt32)Marshal.ReadInt32(buffer, checked(4 + index * 12))),
            HighPart = Marshal.ReadInt32(buffer, checked(4 + index * 12 + 4))
          };
          UInt32 attributes = unchecked((UInt32)Marshal.ReadInt32(buffer, checked(4 + index * 12 + 8)));
          if ((attributes & SE_PRIVILEGE_ENABLED) != 0) {
            int nameLength = 0;
            LookupPrivilegeName(null, ref luid, null, ref nameLength);
            if (nameLength <= 0) throw new Win32Exception(Marshal.GetLastWin32Error(), "LookupPrivilegeName size");
            var name = new StringBuilder(nameLength + 1);
            Win32(LookupPrivilegeName(null, ref luid, name, ref nameLength), "LookupPrivilegeName");
            if (name.ToString() != "SeChangeNotifyPrivilege")
              throw new InvalidOperationException("restricted token retained an unexpected enabled privilege");
            enabledCount++;
          }
        }
        if (enabledCount > 1) throw new InvalidOperationException("restricted token privilege state is noncanonical");
        return true;
      } finally { Marshal.FreeHGlobal(buffer); }
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

    public static DiagnosticResult Run(string executable, string[] arguments,
        string workingDirectory, IDictionary<string,string> manualEnvironment,
        string stdoutPath, string stderrPath, int timeoutSeconds, string factor,
        string mutableTemp, string sentinel, string trustedPath) {
      if (factor != "token-environment-pwsh" && factor != "no-lua-low-il-pwsh")
        throw new ArgumentException("unsupported RR13 native factor");
      IntPtr current = IntPtr.Zero, restricted = IntPtr.Zero, lowSid = IntPtr.Zero, env = IntPtr.Zero;
      IntPtr job = IntPtr.Zero, stdout = IntPtr.Zero, stderr = IntPtr.Zero, stdin = IntPtr.Zero;
      PROCESS_INFORMATION pi = new PROCESS_INFORMATION(); bool created = false;
      string phase = "not-started";
      try {
        Win32(OpenProcessToken(GetCurrentProcess(), TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT | TOKEN_ADJUST_SESSIONID, out current), "OpenProcessToken");
        phase = "current-token-opened";
        UInt32 tokenFlags = factor == "no-lua-low-il-pwsh" ? DISABLE_MAX_PRIVILEGE : DISABLE_MAX_PRIVILEGE | LUA_TOKEN;
        Win32(CreateRestrictedToken(current, tokenFlags, 0, IntPtr.Zero, 0, IntPtr.Zero, 0, IntPtr.Zero, out restricted), "CreateRestrictedToken");
        phase = "restricted-token-created";
        Win32(ConvertStringSidToSid("S-1-16-4096", out lowSid), "ConvertStringSidToSid");
        var label = new TOKEN_MANDATORY_LABEL { Label = new SID_AND_ATTRIBUTES { Sid = lowSid, Attributes = SE_GROUP_INTEGRITY } };
        Win32(SetTokenInformation(restricted, TokenIntegrityLevel, ref label, Marshal.SizeOf<TOKEN_MANDATORY_LABEL>() + (int)GetLengthSid(lowSid)), "SetTokenInformation low integrity");
        string integrity = VerifyToken(restricted);
        bool disabled = VerifyPrivilegesDisabled(restricted);
        phase = "restricted-token-verified";
        job = CreateJobObject(IntPtr.Zero, null); if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject");
        var limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        Win32(SetInformationJobObject(job, JobObjectExtendedLimitInformation, ref limits, (uint)Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()), "SetInformationJobObject");
        phase = "job-configured";
        stdout = OutputHandle(stdoutPath); stderr = OutputHandle(stderrPath); stdin = InputHandle();
        var si = new STARTUPINFO { cb = Marshal.SizeOf<STARTUPINFO>(), dwFlags = (int)STARTF_USESTDHANDLES, hStdInput = stdin, hStdOutput = stdout, hStdError = stderr };
        IDictionary<string,string> childEnvironment = factor == "token-environment-pwsh" ?
          TokenEnvironment(restricted, mutableTemp, sentinel, trustedPath) : manualEnvironment;
        env = SerializeEnvironment(childEnvironment);
        phase = "environment-ready";
        var command = new StringBuilder(Quote(executable)); foreach (string item in arguments) command.Append(' ').Append(Quote(item));
        Win32(CreateProcessAsUser(restricted, executable, command, IntPtr.Zero, IntPtr.Zero, true, CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, env, workingDirectory, ref si, out pi), "CreateProcessAsUser");
        created = true;
        phase = "process-created-suspended";
        Win32(AssignProcessToJobObject(job, pi.hProcess), "AssignProcessToJobObject before resume");
        phase = "job-assigned-before-resume";
        if (ResumeThread(pi.hThread) == UInt32.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread");
        phase = "process-resumed";
        UInt32 wait = WaitForSingleObject(pi.hProcess, checked((uint)timeoutSeconds * 1000));
        bool timedOut = wait == WAIT_TIMEOUT;
        if (wait != WAIT_OBJECT_0 && wait != WAIT_TIMEOUT) throw new Win32Exception(Marshal.GetLastWin32Error(), "WaitForSingleObject");
        UInt32 exitCode = 124;
        if (!timedOut) Win32(GetExitCodeProcess(pi.hProcess, out exitCode), "GetExitCodeProcess");
        phase = "direct-child-exited";
        Win32(TerminateJobObject(job, 125), "TerminateJobObject complete descendant tree");
        DateTime deadline = DateTime.UtcNow.AddSeconds(15);
        while (Active(job) != 0 && DateTime.UtcNow < deadline) Thread.Sleep(25);
        uint remaining = Active(job);
        if (remaining != 0) throw new InvalidOperationException("job object retained active descendants after termination");
        phase = "post-teardown-quiescent";
        return new DiagnosticResult {
          ExitCode = unchecked((int)exitCode), TimedOut = timedOut,
          ActiveProcessesAfterTermination = remaining, Phase = phase,
          TokenFlags = factor == "no-lua-low-il-pwsh" ? "DISABLE_MAX_PRIVILEGE" : "DISABLE_MAX_PRIVILEGE|LUA_TOKEN",
          EnvironmentConstruction = factor == "token-environment-pwsh" ? "restricted-token-CreateEnvironmentBlock-reviewed-allowlist" : "manual-filtered-parent",
          IntegritySid = integrity, PrivilegesDisabled = disabled
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
