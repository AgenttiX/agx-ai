# SPDX-License-Identifier: Apache-2.0
"""The vision tower of Qwen3.8-27B on the CPU (agx-ai addition to the gfx906 image, see README.md).

The Radeon VII's 16 GB leave no room for the vision tower (0.86 GiB of weights, ~1 GiB of activations for a large
image) next to the model and its KV cache. Here it runs with transformers' Qwen3-VL vision model (the same architecture
and weight names; Qwen3.8 has no deepstack features) in FP32 on the CPU, outside vLLM's memory accounting, CUDA graphs
and torch.compile. The model sets mm_encoder_inputs_on_cpu, so that the model runner (cpu-vision.patch) leaves the
pixel values on the CPU, and only the embeddings (5120 values per token in the model's dtype, FP16 on gfx906) are
copied to the GPU. Selected with QWEN38_VISION_DEVICE=cpu, with QWEN38_VISION_THREADS threads (default: PyTorch's).
The same module as ../vllm-rtx3090-hyperqwen/cpu_vision.py, except for the output dtype and the linear layers and
the attention with oneDNN (cpu_onednn.py).
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
from vllm.model_executor.models import agx_cpu_onednn as cpu_onednn  # pyrefly: ignore[missing-import]

DEVICE_ENV = "QWEN38_VISION_DEVICE"
THREADS_ENV = "QWEN38_VISION_THREADS"


def cpu_vision_enabled() -> bool:
    """Whether the vision tower should run on the CPU (QWEN38_VISION_DEVICE=cpu)."""
    return os.environ.get(DEVICE_ENV, "gpu").strip().lower() == "cpu"



class CpuVisionTower(nn.Module):
    """Drop-in replacement for vLLM's Qwen3_VisionTransformer that computes on the CPU."""

    def __init__(self, vision_config: Any, out_dtype: torch.dtype = torch.bfloat16) -> None:
        """vision_config: a transformers config or a mapping; out_dtype: that of the language model."""
        super().__init__()
        cfg_dict = vision_config.to_dict() if hasattr(vision_config, "to_dict") else dict(vision_config)
        cfg_dict.pop("model_type", None)
        cfg = Qwen3VLVisionConfig(**cfg_dict, attn_implementation=cpu_onednn.attn_implementation())
        if cfg.deepstack_visual_indexes:
            raise ValueError("The CPU vision tower does not implement deepstack features")
        # vLLM builds the model under a CUDA default device and a BF16 default dtype.
        with torch.device("cpu"):
            model = Qwen3VLVisionModel(cfg).to(dtype=torch.float32).eval()
        # Not a registered submodule: vLLM must neither see these parameters nor move them.
        object.__setattr__(self, "_hf", model)
        self.spatial_merge_size: int = cfg.spatial_merge_size
        self.out_hidden_size: int = cfg.out_hidden_size
        self.out_dtype: torch.dtype = out_dtype
        self._onednn = False
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
        """Image or video embeddings (merged patches) in the language model's dtype on the GPU."""
        if grid_thw is None:
            raise ValueError("the CPU vision tower needs grid_thw")
        target = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else pixel_values.device
        grid = torch.as_tensor(grid_thw).to("cpu")
        previous = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        if not self._onednn:
            cpu_onednn.use_onednn_linears(self._hf)
            self._onednn = True
        try:
            with torch.inference_mode():
                out = self._hf(pixel_values.to("cpu", torch.float32), grid_thw=grid)
        finally:
            torch.set_num_threads(previous)
        embeds = out.pooler_output if hasattr(out, "pooler_output") else out
        # The output dtype on the CPU first: half the bytes over PCIe and no FP32 copy on the GPU.
        return embeds.to(dtype=self.out_dtype).to(device=target, non_blocking=False)
