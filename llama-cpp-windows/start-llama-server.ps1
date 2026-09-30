<#
.SYNOPSIS
Starts the llama.cpp router server (llama-server.exe) natively on Windows with the configuration of a computer.

.DESCRIPTION
The configuration directory contains the same files as the Docker configurations of the other computers:
- config.ini: server-wide settings in a [*] section. In the Docker image llama-server reads them from
  /etc/llama.cpp/config.ini; on Windows there is no such default, so this script passes them as arguments.
  A relative models-preset path is relative to the configuration directory.
- the preset file that config.ini names (models-preset): the models and their settings.
- an env file (default llama-cpp.env in the configuration directory) with KEY=VALUE lines, e.g. LLAMA_API_KEY.
  They are set as environment variables of the server and never printed.

The models are downloaded to their own cache (default <InstallDir>\models, via HF_HUB_CACHE and LLAMA_CACHE),
not to %USERPROFILE%\.cache\huggingface\hub: if Docker containers have used that one, its snapshot entries are
Linux symbolic links that Windows programs cannot open.

The server runs in the foreground until it is stopped (Ctrl+C), so that the console or the scheduled task
(register-autostart.ps1) controls it.

.EXAMPLE
.\start-llama-server.ps1 -ConfigDir ..\llama-cpp-t550\windows -EnvFile ..\llama-cpp-t550\llama-cpp.env
#>
param(
    [Parameter(Mandatory = $true)][string]$ConfigDir,
    [string]$EnvFile = "",
    [string]$InstallDir = "$env:LOCALAPPDATA\llama.cpp",
    [string]$Version = "",       # installed version directory, e.g. b11262-cuda-13.4 (default: current.txt)
    [string]$LogFile = "",       # llama-server's own log file, in addition to the console
    [string[]]$ExtraArgs = @()   # further llama-server arguments
)
$ErrorActionPreference = "Stop"
$ConfigDir = (Resolve-Path $ConfigDir).Path

if (-not $Version) { $Version = (Get-Content (Join-Path $InstallDir "current.txt") -ErrorAction Stop).Trim() }
$exe = Get-ChildItem -Path (Join-Path $InstallDir $Version) -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $exe) { throw "llama-server.exe not found in $InstallDir\$Version; run install-llama-cpp.ps1 first." }

# Environment: the env file, then defaults for what it does not set.
if (-not $EnvFile) { $EnvFile = Join-Path $ConfigDir "llama-cpp.env" }
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'"))
        }
    }
    Write-Output "Environment from $EnvFile"
} else {
    Write-Output "No env file ($EnvFile); the server runs without an API key unless LLAMA_API_KEY is set."
}
$models = Join-Path $InstallDir "models"
if (-not $env:HF_HUB_CACHE) { $env:HF_HUB_CACHE = $models }
if (-not $env:LLAMA_CACHE) { $env:LLAMA_CACHE = $models }
# Keep more JIT-compiled CUDA kernels than the driver's default, for GPUs without native kernels in the build.
if (-not $env:CUDA_CACHE_MAXSIZE) { $env:CUDA_CACHE_MAXSIZE = "4294967296" }

# config.ini [*] -> arguments. An option is a flag if llama-server's help lists it without an argument placeholder.
$help = (& $exe.FullName --help 2>&1) -join "`n"
function Test-TakesValue([string]$name) {
    $m = [regex]::Match($help, "(?m)(?:^|[ ,])--$([regex]::Escape($name))(?=[ ,\n])( [^ ,\n-][^ ,\n]*)?")
    if (-not $m.Success) { throw "llama-server has no option --$name (from config.ini)" }
    return $m.Groups[1].Success
}
$serverArgs = @()
$inGlobal = $false
foreach ($line in Get-Content (Join-Path $ConfigDir "config.ini")) {
    $line = $line.Trim()
    if ($line -eq "" -or $line.StartsWith(";") -or $line.StartsWith("#")) { continue }
    if ($line -match '^\[(.+)\]$') { $inGlobal = ($Matches[1] -eq "*"); continue }
    if (-not $inGlobal -or $line -notmatch '^([a-z0-9-]+)\s*=\s*(.*)$') { continue }
    $key = $Matches[1]; $value = $Matches[2].Trim()
    if ($key -eq "models-preset" -and -not (Split-Path -IsAbsolute $value)) { $value = Join-Path $ConfigDir $value }
    if (Test-TakesValue $key) {
        $serverArgs += @("--$key", $value)
    } elseif ($value -match '^(1|true|on|yes)$') {
        $serverArgs += "--$key"
    }
}
if ($LogFile) { $serverArgs += @("--log-file", $LogFile) }
$serverArgs += $ExtraArgs

Write-Output "Starting $($exe.FullName)"
Write-Output "  $($serverArgs -join ' ')"
Write-Output "  models: $env:HF_HUB_CACHE"
& $exe.FullName @serverArgs
exit $LASTEXITCODE
