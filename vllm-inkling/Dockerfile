# vLLM nightly with Inkling 2 Mini (vllm-project/vllm#69028, merged at ab18334a4f60 on 2026-11-21, after v0.34.0) and
# inkling-sm86.patch, which lets the model run on Ampere (sm_80/86/89, e.g. the RTX 3090), where upstream requires SM90:
# - the minimum capability of the inkling2 quantization config is 80 instead of 90,
# - below SM90, the model takes the SM12x fallbacks that need no Hopper features: Triton short conv + AttnRes instead
#   of the inkling2 CUDA kernels (the wheel builds them for 9.0a/10.0f/10.3f only, so on sm_86 the ops exist but have
#   no kernel image), no TMA linear, FA2 for the vision tower; the routed experts stay on the Hopper path (Humming,
#   NVFP4/MXFP8 weights with bf16 activations), as the SM12x experts use native FP4 MMA,
# - below SM90, Triton's programmatic dependent launch is disabled (gdc_wait/gdc_launch_dependents compile to
#   nothing, launch_pdl is dropped): ptxas rejects the griddepcontrol PTX for sm_86.
# Other GPUs take the same paths as upstream. ../vllm/vllm.patch is not applied: its fixes are for Qwen3.5/3.8,
# Qwen4Exp, Gemma 4, Triton attention and CPU/UVA offloading, none of which Inkling 2 uses (it has its own attention).
#
# The first nightly with the model (2026-11-22, 8 commits after ab18334a4f60). Python 3.14, CUDA 13.
ARG VLLM_NIGHTLY_TAG=nightly-647a76866b4559a74e27d1c6b55fbffe85c07ce4
FROM vllm/vllm-openai:${VLLM_NIGHTLY_TAG}

COPY inkling-sm86.patch /tmp/inkling-sm86.patch
# The nightly uses Python 3.14 (v0.33.0: 3.12), so the package directory is looked up instead of hard-coded. find_spec
# does not import vllm.
RUN VLLM_DIR="$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')" \
    && test -f "$VLLM_DIR/models/inkling2/quant_config.py" \
    && patch -p1 -d "$VLLM_DIR" < /tmp/inkling-sm86.patch \
    && rm /tmp/inkling-sm86.patch
