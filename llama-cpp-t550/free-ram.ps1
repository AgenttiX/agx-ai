<#
.SYNOPSIS
Shows, and with -Stop closes, background apps that are not needed while a large model runs, to free RAM.

.DESCRIPTION
Only closes apps of the current user that restart normally from the Start menu or at the next login.
It does not touch security software, the VPN, the password manager, Docker Desktop or Claude.
The PostgreSQL services need an administrator shell; the script prints the command for them.

.EXAMPLE
.\free-ram.ps1          # show what would be closed and how much memory it uses
.\free-ram.ps1 -Stop    # close the apps
#>
param([switch]$Stop)

# Process name patterns (regular expressions) of apps to close.
$apps = [ordered]@{
    "Slack"                  = '^slack$'
    "Microsoft Teams"        = '^ms-teams$'
    "OneDrive"               = '^OneDrive(\.Sync\.Service)?$'
    "Phone Link"             = '^PhoneExperienceHost$'
    "PowerToys"              = '^PowerToys(\..+)?$|^Microsoft\.CmdPal\..+$'
    "SyncTrayzor, Syncthing" = '^SyncTrayzor$|^syncthing$'
    "ActivityWatch"          = '^aw-(qt|server|watcher-.+)$'
    "Logitech Options+"      = '^logioptionsplus_.+$'
    "Widgets"                = '^Widget(Board|Service)$'
}

$total = 0
foreach ($app in $apps.Keys) {
    $procs = @(Get-Process | Where-Object { $_.ProcessName -match $apps[$app] })
    if ($procs.Count -eq 0) { continue }
    $mb = [int](($procs | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
    $total += $mb
    if ($Stop) {
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        "{0,-24} {1,6} MB  closed" -f $app, $mb
    } else {
        "{0,-24} {1,6} MB" -f $app, $mb
    }
}
"{0,-24} {1,6} MB{2}" -f "Total", $total, $(if ($Stop) { " freed" } else { " (run with -Stop to close these)" })

$pg = @(Get-Service -Name 'postgresql*' -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'Running' })
if ($pg.Count -gt 0) {
    ""
    "PostgreSQL services running: $($pg.Name -join ', ')"
    "To stop them until the next restart, run in an administrator PowerShell:"
    "    Stop-Service $($pg.Name -join ', ')"
}
