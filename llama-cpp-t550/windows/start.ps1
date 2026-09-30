# Starts llama.cpp natively on Windows with this computer's configuration (see ../README.md).
# Install llama.cpp first with ../../llama-cpp-windows/install-llama-cpp.ps1. Stop the Docker container first
# (docker compose stop in ..), as both use the GPU and port 9931.
param([string]$LogFile = "")
& "$PSScriptRoot\..\..\llama-cpp-windows\start-llama-server.ps1" -ConfigDir $PSScriptRoot `
    -EnvFile "$PSScriptRoot\..\llama-cpp.env" -LogFile $LogFile
exit $LASTEXITCODE
