# AGENTS.md

## Project description
This repository contains the configs of the personal AI server and other AI tools of Mika "AgenttiX" Mäki,
abbreviated as `agx` in the hostnames.

## General instructions
- Before starting your work, run `git pull --rebase` to ensure you have the latest changes.
- Committing directly to main is OK in this project.

## Creating a new LLM backend configuration
- Create the new configuration in its own directory, following the naming convention of other configurations.
- Create a `README.md` in the configuration directory.
- Document at least the following: OS name and version, CPU model, RAM size, RAM type, RAM frequency, NPU model (if any), GPU model, GPU architecture, VRAM size, VRAM type.
  For laptops with a discrete GPU (dGPU), also document the following: specifications of both iGPU and dGPU, dGPU power limit, dGPU PCIe link width.
- For setups that require a custom Docker container,
  create a workflow in `./.github/workflows/` for building the container.

## Benchmarking
- Use the scripts in `./benchmark/` for benchmarking. You may update the scripts as needed.

## Commands
- Run all lints and type checks: `./lint.sh`
  - This runs `ruff check`, `pyrefly check` and `pyrefly coverage check`.
    All checks are run even if some of them fail. The exit code is 0 if all checks pass,
    the exit code of the failed check if exactly one check fails, and 100 if multiple checks fail.
