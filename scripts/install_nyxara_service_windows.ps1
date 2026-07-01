# =============================================================================
# NYXARA - install the always-on service on Windows (SYSTEM scheduled task).
# -----------------------------------------------------------------------------
# Registers NYXARA (HTTP server + background mind) as a Scheduled Task that:
#   * starts automatically at boot, as SYSTEM, BEFORE anyone logs in,
#   * restarts itself if it crashes or is killed,
#   * runs 24/7 while the laptop is on.
#
# A process cannot run while the machine is powered OFF - this delivers
# "always alive" the only way software can: auto-start on boot + auto-restart.
#
# Run in an ELEVATED PowerShell (Run as administrator):
#   powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1 -Uninstall
#
# Requires Python 3.11+ with NYXARA installed (pip install -e .).
# =============================================================================
param(
    [switch]$Uninstall,
    [string]$TaskName = "NYXARA"
)

$ErrorActionPreference = "Stop"

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "This installer must run in an elevated (Administrator) PowerShell."
        exit 1
    }
}

Assert-Admin

if ($Uninstall) {
    Write-Host "==> Removing scheduled task '$TaskName' ..."
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "==> Removed. (Your ~\.nyxara data was left intact.)"
    } else {
        Write-Host "==> No task named '$TaskName' found; nothing to do."
    }
    exit 0
}

# --- Repo root and the command that launches the daemon ------------------------------
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Prefer pythonw.exe (no console window) so the daemon runs silently in the background.
$Pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue)
$Python  = (Get-Command python.exe  -ErrorAction SilentlyContinue)
if ($Pythonw) {
    $Exe = $Pythonw.Source
} elseif ($Python) {
    $Exe = $Python.Source
} else {
    Write-Error "Python not found on PATH. Install NYXARA first:  py -m pip install -e ."
    exit 1
}
# -u = unbuffered; -m nyxara.daemon turns the background mind on and serves the API.
$Arguments = "-u -m nyxara.daemon"

Write-Host "==> Repo:   $RepoRoot"
Write-Host "==> Launch: $Exe $Arguments"

# --- Build the task (trigger at startup, run as SYSTEM, restart on failure) ----------
$Action = New-ScheduledTaskAction -Execute $Exe -Argument $Arguments -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Turn the background mind on for the task's process. (Set your own env in .env / system env.)
[Environment]::SetEnvironmentVariable("NYXARA_SERVER__AUTONOMIC", "true", "Machine")

$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)   # 0 = run indefinitely

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "==> Task '$TaskName' exists - replacing it."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings `
    -Description "NYXARA - sovereign cognitive daemon (API + background mind). Always-on." | Out-Null

Write-Host "==> Starting it now ..."
Start-ScheduledTask -TaskName $TaskName

Write-Host ""
Write-Host "Done. NYXARA is now an always-on task (auto-start at boot, auto-restart on failure)."
Write-Host "  Status:    Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Stop now:  Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  Start now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Uninstall: powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1 -Uninstall"
Write-Host ""
Write-Host "Tip: set NYXARA_SERVER__API_TOKEN and any LLM keys as Machine environment"
Write-Host "     variables (or in $RepoRoot\.env). Keep the bind host at 127.0.0.1"
Write-Host "     unless you intend to expose the API to your network."
Write-Host ""
Write-Host "Alternative: for a true Windows *service* (not a task), use NSSM:"
Write-Host "     nssm install NYXARA `"$Exe`" `"$Arguments`""
