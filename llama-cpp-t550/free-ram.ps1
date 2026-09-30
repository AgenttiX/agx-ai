<#
.SYNOPSIS
Shows, and with -Stop closes, background programs and services that are not needed while a large model runs, to free RAM.

.DESCRIPTION
Closes apps of the current user that restart normally from the Start menu or at the next login, stops Docker Desktop
and WSL 2, and stops some services until the next restart. Stopping services needs an administrator PowerShell;
in a normal one, the script prints the command for them instead.
It does not touch the security software, the VPN, the password manager, Claude, or the apps that should stay open
(e.g. messaging, file synchronization and time management programs).

.EXAMPLE
.\free-ram.ps1          # show what would be closed and how much memory it uses
.\free-ram.ps1 -Stop    # close and stop them
#>
param([switch]$Stop)

# Apps of the current user: process name patterns (regular expressions).
$apps = [ordered]@{
    "Intel Driver & Support Assistant tray" = '^DSATray$'
    "Copilot"                               = '^mscopilot_proxy$|^Copilot$'
    "Widgets"                               = '^Widget(Board|Service)$'
    "Phone Link"                            = '^PhoneExperienceHost$|^CrossDeviceResume$'
    # Mouse button and gesture customizations do not work while it is closed.
    "Logitech Options+"                     = '^logioptionsplus_.+$'
}
# Services: display name -> service name patterns. They restart at the next boot (all are set to start automatically).
$services = [ordered]@{
    # A software licensing server: programs that are licensed through it do not start while it is stopped.
    "CodeMeter (incl. CmWebAdmin)"           = '^(CodeMeter\.exe|CmWebAdmin\.exe|CmLogger)$'
    "Intel Driver & Support Assistant"       = '^DSA(Update)?Service$'
    "PostgreSQL servers"                     = '^postgresql-'
}

function Get-WorkingSetMB($procs) { [int](($procs | Measure-Object WorkingSet64 -Sum).Sum / 1MB) }
# An elevated (administrator) PowerShell has the high integrity level (S-1-16-12288).
$isAdmin = [bool]((& "$env:SystemRoot\System32\whoami.exe" /groups) -match 'S-1-16-12288')
$total = 0

foreach ($app in $apps.Keys) {
    $procs = @(Get-Process | Where-Object { $_.ProcessName -match $apps[$app] })
    if ($procs.Count -eq 0) { continue }
    $mb = Get-WorkingSetMB $procs; $total += $mb
    if ($Stop) { $procs | Stop-Process -Force -ErrorAction SilentlyContinue; "{0,-40} {1,6} MB  closed" -f $app, $mb }
    else { "{0,-40} {1,6} MB" -f $app, $mb }
}

# Docker Desktop (~0.9 GB) and the WSL 2 VM (vmmemWSL, 2.6 GB when idle on 2026-09-30), which also holds the
# Linux page cache, plus the WSL helper processes. This also stops the Ubuntu distribution.
$docker = @(Get-Process | Where-Object { $_.ProcessName -match '^Docker Desktop$|^com\.docker\.|^docker-agent$' })
$vm = @(Get-Process | Where-Object { $_.ProcessName -match '^vmmemWSL$|^wsl$|^wslhost$|^wslrelay$' })
$mb = (Get-WorkingSetMB $docker) + (Get-WorkingSetMB $vm)
if ($docker.Count -gt 0 -or $mb -gt 0) {
    $total += $mb
    if ($Stop) {
        if ($docker.Count -gt 0) { docker desktop stop 2>&1 | Out-Null }
        wsl --shutdown
        "{0,-40} {1,6} MB  stopped" -f "Docker Desktop and WSL 2", $mb
    } else {
        "{0,-40} {1,6} MB" -f "Docker Desktop and WSL 2", $mb
    }
}

$toStop = @()
$running = @(Get-CimInstance Win32_Service | Where-Object { $_.State -eq 'Running' })
$allProcs = @(Get-CimInstance Win32_Process)
foreach ($name in $services.Keys) {
    $s = @($running | Where-Object { $_.Name -match $services[$name] })
    if ($s.Count -eq 0) { continue }
    # The service processes and all their descendants (e.g. the PostgreSQL server and its workers).
    $ids = @($s.ProcessId)
    do {
        $n = $ids.Count
        $ids = @($ids + @($allProcs | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId }) | Sort-Object -Unique)
    } while ($ids.Count -gt $n)
    $mb = Get-WorkingSetMB @($ids | Sort-Object -Unique | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    $total += $mb
    if ($Stop -and $isAdmin) {
        $s | ForEach-Object { Stop-Service -Name $_.Name -Force -ErrorAction SilentlyContinue }
        "{0,-40} {1,6} MB  stopped" -f $name, $mb
    } else {
        "{0,-40} {1,6} MB  (service)" -f $name, $mb
        $toStop += $s.Name
    }
}

"{0,-40} {1,6} MB{2}" -f "Total (working sets)", $total, $(if ($Stop) { "" } else { " - run with -Stop to close these" })
if ($toStop.Count -gt 0 -and ($Stop -or -not $isAdmin)) {
    ""
    "Stopping the services needs an administrator PowerShell (they start again at the next boot):"
    "    Stop-Service -Force '$($toStop -join "', '")'"
}
