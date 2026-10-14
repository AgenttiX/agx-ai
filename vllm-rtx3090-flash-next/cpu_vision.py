# SPDX-License-Identifier: Apache-2.0
"""The vision tower of Qwen3.8-Flash-Next on the CPU (agx-ai addition to the qwen38-flash-next-3090 runtime).

The runtime keeps every MiB of VRAM for the language model: the hot experts, the KV cache and the expert streaming of
the prefill. On the GPU, the vision tower would need 0.84 GiB for its weights and ~1 GiB of activations for a
1280x960 image. Here it runs with transformers' Qwen3-VL vision model (the same architecture and weight names) in FP32
on the CPU, outside vLLM's memory accounting, CUDA graphs and torch.compile. The model sets mm_encoder_inputs_on_cpu,
so that the encoder runner (vllm-mm.patch) leaves the pixel values on the CPU, and only the embeddings (2560 values
per token in BF16) are copied to the GPU. Zen 2 has no fast BF16, so FP32 is 3.5 times as fast as BF16 there; the
pixel values come from the processor in FP32 anyway.

The engine runs the encoder of a request before its prefill, while the CPU expert pool is idle (one request at a
time), so the vision threads do not compete with the decode of the same server. Videos go through the same forward
pass: transformers' model attends within each frame (pair), like vLLM's.
"""

from collections.abc import Iterable
import os
from typing import Any

# The module runs in the vLLM image, which has these packages; the development environment does not have them.
import torch  # pyrefly: ignore[missing-import]
from torch import nn  # pyrefly: ignore[missing-import]
from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # pyrefly: ignore[missing-import]
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # pyrefly: ignore[missing-import]

DEVICE_ENV = "QWEN38_VISION_DEVICE"
THREADS_ENV = "QWEN38_VISION_THREADS"


def cpu_vision_enabled() -> bool:
    """Whether the vision tower should run on the CPU (QWEN38_VISION_DEVICE=cpu)."""
    return os.environ.get(DEVICE_ENV, "gpu").strip().lower() == "cpu"


class CpuVisionTower(nn.Module):
    """Drop-in replacement for vLLM's Qwen3_VisionTransformer that computes on the CPU."""

    def __init__(self, vision_config: Any) -> None:  # a transformers config or a mapping
        super().__init__()
        cfg_dict = vision_config.to_dict() if hasattr(vision_config, "to_dict") else dict(vision_config)
        cfg_dict.pop("model_type", None)
        cfg = Qwen3VLVisionConfig(**cfg_dict, attn_implementation="sdpa")
        if cfg.deepstack_visual_indexes:
            raise ValueError("The CPU vision tower does not implement deepstack features")
        # vLLM builds the model under a CUDA default device and a BF16 default dtype.
        with torch.device("cpu"):
            model = Qwen3VLVisionModel(cfg).to(dtype=torch.float32).eval()
        # Not a registered submodule: vLLM must neither see these parameters nor move them.
        object.__setattr__(self, "_hf", model)
        self.spatial_merge_size: int = cfg.spatial_merge_size
        self.out_hidden_size: int = cfg.out_hidden_size
        self.threads: int | None = int(os.environ.get(THREADS_ENV, "0")) or None

    @property
    def dtype(self) -> torch.dtype:
        """The dtype that the caller casts the pixel values to."""
        return torch.float32

    @property
    def device(self) -> torch.device:
        """Where the vision tower computes."""
        return torch.device("cpu")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint's visual.* weights (names relative to the tower) in FP32."""
        params = dict(self._hf.named_parameters())
        loaded = set()
        for name, tensor in weights:
            param = params.get(name)
            if param is None:
                raise KeyError(f"unexpected vision tower weight {name!r}")
            with torch.no_grad():
                param.copy_(tensor.to(device="cpu", dtype=torch.float32))
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise KeyError(f"vision tower weights missing from the checkpoint: {sorted(missing)[:5]}")
        return loaded

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Image or video embeddings (merged patches) in BF16 on the GPU."""
        if grid_thw is None:
            raise ValueError("the CPU vision tower needs grid_thw")
        target = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else pixel_values.device
        grid = torch.as_tensor(grid_thw).to("cpu")
        previous = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            with torch.inference_mode():
                out = self._hf(pixel_values.to("cpu", torch.float32), grid_thw=grid)
        finally:
            torch.set_num_threads(previous)
        embeds = out.pooler_output if hasattr(out, "pooler_output") else out
        # BF16 on the CPU first: half the bytes over PCIe and no FP32 copy on the GPU.
        return embeds.to(dtype=torch.bfloat16).to(device=target, non_blocking=False)
