<#
.SYNOPSIS
    Configure Claude Code
#>

$GitDir = (Get-Item "${PSScriptRoot}").Parent.Parent.FullName
. "${GitDir}\windows-scripts\Utils.ps1"

$ClaudeDir = "${UserDir}\.claude"
$ClaudeSettings = "${ClaudeDir}\settings.json"

if (! (Test-Path "${ClaudeDir}")) {
    Show-Output "Claude configuration directory was not found at `"${ClaudeDir}`"."
    exit 1
}
if (Test-Path "${ClaudeSettings}") {
    if (Test-IsLink "${ClaudeSettings}") {
        Show-Output "Claude settings.json is already linked."
    } else {
        Move-Item "${ClaudeSettings}" "${ClaudeDir}\settings-old.json"
    }
}
New-Item -ItemType SymbolicLink "${ClaudeSettings}" -Target "${PSScriptRoot}\settings.json"
