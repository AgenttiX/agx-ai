# SPDX-License-Identifier: Apache-2.0
"""The vision tower of Qwen3.8-27B on the CPU (agx-ai addition to the HyperQwen image, see README.md).

The configuration keeps every MiB of VRAM for the language model, the DFlash2 drafter and the KV cache: on the GPU,
the vision tower would need 0.84 GiB for its weights (HyperQwen's VISION=1 keeps them in RAM, but runs them on the
GPU) and ~1 GiB of activations for a large image, which would cost ~25k tokens of context. Here it runs with
transformers' Qwen3-VL vision model (the same architecture and weight names; Qwen3.8 has no deepstack features) in
FP32 on the CPU, outside vLLM's memory accounting, CUDA graphs and torch.compile. The model sets
mm_encoder_inputs_on_cpu, so that the encoder runner (hyperqwen-vision.patch) leaves the pixel values on the CPU, and
only the embeddings (5120 values per token in BF16) are copied to the GPU. Zen 2 has no fast BF16, so FP32 is 3.5
times as fast as BF16 there; the pixel values come from the processor in FP32 anyway. Selected with
QWEN38_VISION_DEVICE=cpu, with QWEN38_VISION_THREADS threads (default: PyTorch's). The same module as in
../vllm-rtx3090-flash-next, plus tensor parallelism (../vllm-rtx-a4000): the ranks split the frames of the videos
and the images between them and exchange the embeddings.
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
from vllm.distributed import parallel_state  # pyrefly: ignore[missing-import]

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
        tp_group = _tp_group()
        if tp_group is None:
            return self._compute(pixel_values, grid).to(device=target, non_blocking=False)
        # With tensor parallelism, every rank runs the encoder with the same inputs. The frames of the videos and the
        # images are split between the ranks, each rank computes its share, and the ranks exchange the embeddings
        # over NCCL. The tower attends within each frame, and its position embeddings depend only on the height and
        # width of a frame, so the result is the same as computing all frames at once.
        shares = _split_frames(grid, tp_group.world_size)
        merge = self.spatial_merge_size**2
        # Each rank computes its own share first, so that the ranks compute at the same time, and then all shares
        # are broadcast in order.
        # QWEN38_VISION_THREADS is the budget per rank: a rank that computes alone (e.g. a single image) gets the
        # threads of the idle ranks too.
        busy = sum(1 for rows, _ in shares if rows.stop > rows.start)
        threads = self.threads * tp_group.world_size // busy if self.threads else None
        rows, share_grid = shares[tp_group.rank_in_group]
        own = self._compute(pixel_values[rows], share_grid, threads) if rows.stop > rows.start else None
        embeds = []
        for rank, (rows, _) in enumerate(shares):
            tokens = (rows.stop - rows.start) // merge
            if tokens == 0:
                continue
            if rank == tp_group.rank_in_group and own is not None:
                part = own.to(device=target, non_blocking=False)
            else:
                part = torch.empty((tokens, self.out_hidden_size), dtype=torch.bfloat16, device=target)
            embeds.append(tp_group.broadcast(part.contiguous(), src=rank))
        return torch.cat(embeds)

    def _compute(self, pixel_values: torch.Tensor, grid: torch.Tensor, threads: int | None = None) -> torch.Tensor:
        """The embeddings of the given patches on the CPU, in BF16, with the given number of threads."""
        threads = threads or self.threads
        previous = torch.get_num_threads()
        if threads:
            torch.set_num_threads(threads)
        try:
            with torch.inference_mode():
                out = self._hf(pixel_values.to("cpu", torch.float32), grid_thw=grid)
        finally:
            torch.set_num_threads(previous)
        embeds = out.pooler_output if hasattr(out, "pooler_output") else out
        # BF16 on the CPU first: half the bytes over PCIe and no FP32 copy on the GPU.
        return embeds.to(dtype=torch.bfloat16)


def _split_frames(grid: torch.Tensor, parts: int) -> list[tuple[slice, torch.Tensor]]:
    """Split the frames of grid_thw (t, h, w per image or video) into contiguous shares of about equal patch counts.

    Returns the rows of pixel_values and the grid_thw of each share; a share may be empty.
    """
    frames = [(int(h), int(w)) for t, h, w in grid.tolist() for _ in range(int(t))]
    items = [i for i, (t, _, _) in enumerate(grid.tolist()) for _ in range(int(t))]
    total = sum(h * w for h, w in frames)
    shares: list[tuple[slice, torch.Tensor]] = []
    frame = row = 0
    for part in range(parts):
        first_frame, first_row = frame, row
        # A frame goes to this share if its middle is before the share's end.
        end = total * (part + 1) / parts
        while frame < len(frames) and (part == parts - 1 or row + frames[frame][0] * frames[frame][1] / 2 <= end):
            row += frames[frame][0] * frames[frame][1]
            frame += 1
        share_grid: list[list[int]] = []
        for f in range(first_frame, frame):
            h, w = frames[f]
            if share_grid and items[f] == items[f - 1] and f > first_frame:
                share_grid[-1][0] += 1
            else:
                share_grid.append([1, h, w])
        shares.append((slice(first_row, row), torch.tensor(share_grid, dtype=grid.dtype).reshape(-1, 3)))
    return shares


def _tp_group() -> Any:
    """The tensor-parallel group of vLLM if the model runs on more than one GPU, else None."""
    if not parallel_state.model_parallel_is_initialized():
        return None
    group = parallel_state.get_tp_group()
    return group if group.world_size > 1 else None
