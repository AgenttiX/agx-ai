<#
.SYNOPSIS
Downloads a llama.cpp release for Windows from GitHub, verifies it and installs it next to the previous versions.

.DESCRIPTION
Downloads the release archive (and for CUDA the matching CUDA runtime archive) from
https://github.com/ggml-org/llama.cpp/releases, checks the SHA-256 digests that GitHub publishes for the assets,
extracts both into <InstallDir>\<build>-<backend> and makes that the current version (<InstallDir>\current.txt,
used by start-llama-server.ps1). Previous versions are kept, so going back is a matter of editing current.txt.
Needs no administrator rights. CUDA builds need only the NVIDIA driver, not the CUDA toolkit.

.PARAMETER Build
Release tag, e.g. b11262, or "latest".

.PARAMETER Backend
cuda-13.4, cuda-12.4, vulkan or cpu (the part of the asset name after "bin-win-"; see the release page for others).
A CUDA build runs with a driver that supports at least the same major CUDA version (CUDA minor version
compatibility), but the most reliable choice is a build whose CUDA version the driver supports ("CUDA Version" in
nvidia-smi).

.EXAMPLE
.\install-llama-cpp.ps1                                # latest release, CUDA 13.4
.\install-llama-cpp.ps1 -Build b11262 -Backend cuda-12.4
#>
param(
    [string]$Build = "latest",
    [string]$Backend = "cuda-13.4",
    [string]$InstallDir = "$env:LOCALAPPDATA\llama.cpp"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # the progress bar makes Invoke-WebRequest very slow

$api = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
if ($Build -eq "latest") {
    # GitHub's "latest" release can be a version release without the build assets, so take the newest release
    # that has this backend. Old build releases are deleted after a while.
    $release = $null
    foreach ($r in (Invoke-RestMethod "${api}?per_page=30")) {
        if ($r.assets.name -contains "llama-$($r.tag_name)-bin-win-$Backend-x64.zip") { $release = $r; break }
    }
    if (-not $release) { throw "None of the 30 newest releases has a $Backend build for Windows x64." }
} else {
    $release = Invoke-RestMethod "$api/tags/$Build"
}
$tag = $release.tag_name
$names = @("llama-$tag-bin-win-$Backend-x64.zip")
if ($Backend -like "cuda-*") { $names += "cudart-llama-bin-win-$Backend-x64.zip" }

$target = Join-Path $InstallDir "$tag-$Backend"
$downloads = Join-Path $InstallDir "downloads"
New-Item -ItemType Directory -Force $downloads | Out-Null
if (Test-Path $target) { Write-Output "$target exists already; reinstalling it." ; Remove-Item -Recurse -Force $target }
New-Item -ItemType Directory -Force $target | Out-Null

foreach ($name in $names) {
    $asset = $release.assets | Where-Object { $_.name -eq $name }
    if (-not $asset) {
        $available = ($release.assets | Where-Object { $_.name -like "*win*" } | ForEach-Object { $_.name }) -join "`n  "
        throw "Release $tag has no asset $name. Windows assets:`n  $available"
    }
    $file = Join-Path $downloads $name
    Write-Output ("Downloading {0} ({1:N0} MiB)..." -f $name, ($asset.size / 1MB))
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $file
    $expected = ($asset.digest -replace '^sha256:', '').ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
    if (-not $expected) { throw "GitHub publishes no digest for $name; not installing an unverified file." }
    if ($actual -ne $expected) { throw "SHA-256 mismatch for ${name}: expected $expected, got $actual" }
    Write-Output "  SHA-256 OK"
    Expand-Archive -Path $file -DestinationPath $target -Force
    Remove-Item $file
}

$exe = Get-ChildItem -Path $target -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $exe) { throw "llama-server.exe not found in $target" }
Set-Content -Path (Join-Path $InstallDir "current.txt") -Value (Split-Path -Leaf $target)
Write-Output "Installed $tag ($Backend) to $target, now the current version."
Write-Output "Checking that it runs (this also lists the GPUs it can use):"
& $exe.FullName --version 2>&1 | ForEach-Object { "  $_" }
& $exe.FullName --list-devices 2>&1 | ForEach-Object { "  $_" }
