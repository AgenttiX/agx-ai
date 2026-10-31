# AGENTS.md

## Project description
This repository contains the configs of the personal AI server and other AI tools of Mika "AgenttiX" Mäki,
abbreviated as `agx` in the hostnames.

## General instructions
- Before starting your work, run `git pull --rebase` to ensure you have the latest changes.
- Use `./hostname.py` to get the hostname of this computer to be used in the repository.
  If it returns "censored", don't write the hostname in this repository.
  Instead, use its hardware model in place of the hostname for e.g. the benchmark files.
- Committing directly to main is OK in this project.

## Creating a new LLM backend configuration
- Use the directory naming convention of `BACKEND-ACCELERATOR`, e.g. `vllm-radeon-vii` by default.
  If the configuration is specific to a particular computer (e.g. multi-GPU),
  use the naming convention of `BACKEND-HOSTNAME`,
  unless `./hostname.py` returns "censored".
  (Many of the existing configurations intentionally deviate from these naming conventions.)
- Create a `README.md` in the configuration directory.
- Document at least the following: OS name and version, CPU model, RAM size, RAM type, RAM frequency, NPU model (if any), GPU model, GPU architecture, VRAM size, VRAM type.
  For laptops with a discrete GPU (dGPU), also document the following: specifications of both iGPU and dGPU, dGPU power limit, dGPU PCIe link width.
  For computers for which `./hostname.py` does not return "censored", document also the computer/motherboard/laptop model.
- For setups that require a custom Docker container that is not already available in a container registry,
  create a workflow in `./.github/workflows/` for building the container.
- For setups that modify the vLLM Docker image but don't use a different base image,
  prefer to modify the generic image in `./vllm` instead of creating a new container.
  You may update the version of its base image as needed.
  However, keep that image compatible with the other configurations.

## Updating an LLM backend configuration
- Maintain a clear summary table of the current configuration in the start of its `README.md`.
  For each model in the configuration, include the model, context size, VRAM use, RAM use,
  token generation speed, prompt processing speed, and notes (if there are any significant remarks).
  If the same model works significantly better on the same hardware with a different backend, mention this.
- Keep `./llama-cpp/OPTIMIZATIONS.md` up to date with all llama.cpp optimizations, and `./vllm/OPTIMIZATIONS.md` with all vLLM optimizations.
- When achieving a new performance record, check whether the computer is listed in `./structure/generate.py`, and whether it has the LLM model listed.
  If yes, then update the performance numbers in it.
  If the computer is listed but the LLM model is not, then mention this to the user.

## Benchmarking
- Use the scripts in `./benchmark/` for benchmarking.
  You may update the scripts as needed, but keep them compatible with the other backends.
- If you crate new testing scripts, store them in `./benchmark/`.
  However, if they are specific to a particular backend configuration, store them in its directory instead.

## Commands
- Run all lints and type checks: `./lint.sh`
  - Run this after creating or modifying Python code.
  - This runs `ruff check`, `pyrefly check` and `pyrefly coverage check`.
    All checks are run even if some of them fail. The exit code is 0 if all checks pass,
    the exit code of the failed check if exactly one check fails, and 100 if multiple checks fail.
