<#
.SYNOPSIS
Stops the native llama.cpp server: the router and its model instances (llama-server.exe).

.DESCRIPTION
If the server runs from the scheduled task of register-autostart.ps1, the task is ended first, as its
"restart on failure" setting would otherwise start the server again. Then all llama-server.exe processes that run from
the install directory are stopped, which frees the VRAM and the RAM of the loaded model. Nothing needs to be saved:
the server keeps no state apart from the prompt cache, which is lost anyway when a model is unloaded.

.EXAMPLE
.\stop-llama-server.ps1
#>
param(
    [string]$TaskName = "llama.cpp server",
    [string]$InstallDir = "$env:LOCALAPPDATA\llama.cpp"
)
$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task -and $task.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName
    Write-Output "Ended the scheduled task '$TaskName'."
}

$installPath = (Resolve-Path $InstallDir -ErrorAction SilentlyContinue).Path
function Get-ServerProcesses {
    @(Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { -not $installPath -or $_.Path -like "$installPath\*" })
}
$procs = Get-ServerProcesses
if ($procs.Count -eq 0) {
    Write-Output "llama-server is not running."
    return
}
$count = $procs.Count
$mb = [int](($procs | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
# Stop the router first: when a model instance dies while the router still runs, the router may start a new one.
# Repeat until none is left, in case one was started in between.
for ($round = 0; $round -lt 5 -and $procs.Count -gt 0; $round++) {
    $children = @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | ForEach-Object { [int]$_.ParentProcessId })
    $routers = @($procs | Where-Object { $children -contains $_.Id })
    $routers + @($procs | Where-Object { $routers -notcontains $_ }) | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 20 -and @(Get-Process -Id $procs.Id -ErrorAction SilentlyContinue).Count -gt 0; $i++) { Start-Sleep -Milliseconds 500 }
    Start-Sleep 1
    $procs = Get-ServerProcesses
    $count += $procs.Count
}
if ($procs.Count -gt 0) { throw "$($procs.Count) llama-server process(es) did not stop." }
Write-Output "Stopped $count llama-server process(es) (working set $mb MB)."
