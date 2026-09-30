<#
.SYNOPSIS
Registers (or with -Unregister removes) a scheduled task that starts a computer's llama.cpp server at logon.

.DESCRIPTION
The task runs the given start script (e.g. ..\llama-cpp-t550\windows\start.ps1) hidden in the background as the
current user when they log on, and restarts it up to three times if it fails. The server's output goes to the log
file. The task is a per-user task, so this needs no administrator rights.

.EXAMPLE
.\register-autostart.ps1 -StartScript ..\llama-cpp-t550\windows\start.ps1
.\register-autostart.ps1 -Unregister
Start-ScheduledTask -TaskName "llama.cpp server"      # start it now
.\stop-llama-server.ps1                               # stop the server (ends the task first)
#>
param(
    [string]$StartScript = "",
    [string]$TaskName = "llama.cpp server",
    [string]$LogFile = "$env:LOCALAPPDATA\llama.cpp\llama-server.log",
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed the scheduled task '$TaskName'."
    return
}
if (-not $StartScript) { throw "Give the computer's start script with -StartScript." }
$StartScript = (Resolve-Path $StartScript).Path
$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = (Get-Command powershell).Source }

$action = New-ScheduledTaskAction -Execute $pwsh `
    -Argument "-NoProfile -WindowStyle Hidden -File `"$StartScript`" -LogFile `"$LogFile`"" `
    -WorkingDirectory (Split-Path $StartScript)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
# No time limit, keep running on battery, restart after failures.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "llama.cpp router server ($StartScript)" -Force | Out-Null
Write-Output "Registered the scheduled task '$TaskName' (at logon of $env:USERNAME). Log: $LogFile"
