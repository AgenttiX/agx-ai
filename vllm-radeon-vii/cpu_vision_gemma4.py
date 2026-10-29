# SPDX-License-Identifier: Apache-2.0
"""The vision tower of Gemma 4 on the CPU (agx-ai addition to the gfx906 image, see README.md).

The Radeon VII's 16 GB leave no room for the vision tower (1.1 GiB of weights in FP16 and its activations) next to the
model and its KV cache. Here transformers' Gemma 4 vision model, the same module that vLLM uses on the GPU, is built on
the CPU in FP32 and computes there; only its output soft tokens (1152 values per token) are copied to the GPU, where
vLLM's embedder projects them into the language model's embedding space. The parameters are registered under the same
names as on the GPU, so that vLLM's weight loader fills them, but they are on the CPU and so outside vLLM's memory
accounting. The model sets mm_encoder_inputs_on_cpu, so that the model runner (cpu-vision.patch) leaves the pixel
values on the CPU. FP32, as Zen 2 has no fast BF16 or FP16, and the linear layers and the attention with oneDNN
(cpu_onednn.py).
Selected with GEMMA4_VISION_DEVICE=cpu, with GEMMA4_VISION_THREADS threads (default: PyTorch's).
"""

from collections.abc import Iterator, Sequence
import contextlib
import os
from typing import Any

# The module runs in the vLLM image, which has these packages; the development environment does not have them.
import torch  # pyrefly: ignore[missing-import]
from torch import nn  # pyrefly: ignore[missing-import]
from transformers import AutoModel  # pyrefly: ignore[missing-import]
from vllm.model_executor.models import agx_cpu_onednn as cpu_onednn  # pyrefly: ignore[missing-import]

DEVICE_ENV = "GEMMA4_VISION_DEVICE"
THREADS_ENV = "GEMMA4_VISION_THREADS"
# Patches per encoder call: the CPU attention of an image of 2520 patches takes ~0.4 GB, so a few images at a time.
CHUNK_PATCHES = 10080


def cpu_vision_enabled() -> bool:
    """Whether the vision tower should run on the CPU (GEMMA4_VISION_DEVICE=cpu)."""
    return os.environ.get(DEVICE_ENV, "gpu").strip().lower() == "cpu"


def build_cpu_vision_tower(vision_config: Any) -> nn.Module:
    """The Gemma 4 vision model of transformers in FP32 on the CPU (vLLM builds the model under a GPU device)."""
    with torch.device("cpu"):
        tower = AutoModel.from_config(
            config=vision_config, dtype=torch.float32, attn_implementation=cpu_onednn.attn_implementation()
        )
    return tower.to(device="cpu", dtype=torch.float32).eval()



@contextlib.contextmanager
def _threads() -> Iterator[None]:
    threads = int(os.environ.get(THREADS_ENV, "0"))
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _encode_batch(tower: nn.Module, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> list[torch.Tensor]:
    """The soft tokens of a batch of items with the same number of patches, as vLLM's GPU path computes them."""
    padding = (position_ids == -1).all(dim=-1)
    embeds = tower.patch_embedder(pixel_values, position_ids, padding)
    hidden = tower.encoder(
        inputs_embeds=embeds, attention_mask=~padding, pixel_position_ids=position_ids
    ).last_hidden_state
    pooled, valid = tower.pooler(
        hidden_states=hidden,
        pixel_position_ids=position_ids,
        padding_positions=padding,
        output_length=pixel_values.shape[1] // tower.config.pooling_kernel_size**2,
    )
    states = []
    for i in range(pixel_values.shape[0]):
        item = pooled[i][valid[i]]
        if getattr(tower.config, "standardize", False):
            item = (item - tower.std_bias) * tower.std_scale
        states.append(item)
    return states


def encode(
    tower: nn.Module,
    pixel_values: torch.Tensor | Sequence[torch.Tensor],
    position_ids: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """The FP32 soft tokens of each image or video frame (each item: patches x values) on the CPU."""
    if not getattr(tower, "_agx_onednn", False):
        cpu_onednn.use_onednn_linears(tower)
        tower._agx_onednn = True  # noqa: SLF001
    items = [(pv.to("cpu", torch.float32), pp.to("cpu")) for pv, pp in zip(pixel_values, position_ids, strict=True)]
    states: list[torch.Tensor | None] = [None] * len(items)
    by_patches: dict[int, list[int]] = {}
    for i, (pv, _) in enumerate(items):
        by_patches.setdefault(pv.shape[0], []).append(i)
    with _threads(), torch.inference_mode():
        for patches, indices in by_patches.items():
            chunk = max(1, CHUNK_PATCHES // max(patches, 1))
            for start in range(0, len(indices), chunk):
                batch = indices[start : start + chunk]
                out = _encode_batch(
                    tower,
                    torch.stack([items[i][0] for i in batch]),
                    torch.stack([items[i][1] for i in batch]),
                )
                for i, s in zip(batch, out, strict=True):
                    states[i] = s
    return [s for s in states if s is not None]
