# SPDX-License-Identifier: Apache-2.0
"""oneDNN for the CPU vision towers of the gfx906 image (agx-ai addition, see README.md).

The PyTorch of the gfx906 image has no MKL or other BLAS library, so that its CPU matrix multiplications fall back to a
single-threaded loop (~7 GFLOP/s on the Threadripper 3970X), and its SDPA reaches only ~100 GFLOP/s. oneDNN, which it
has, computes a linear layer ~100 times and the attention of a vision tower ~1.9 times as fast. With a PyTorch that has
MKL, these functions change nothing.
"""

from typing import Any

# The module runs in the vLLM image, which has these packages; the development environment does not have them.
import torch  # pyrefly: ignore[missing-import]
from torch import nn  # pyrefly: ignore[missing-import]
from torch.utils.mkldnn import MkldnnLinear  # pyrefly: ignore[missing-import]
from transformers import AttentionInterface  # pyrefly: ignore[missing-import]

ATTENTION = "agx_onednn"


def needed() -> bool:
    """Whether PyTorch lacks MKL but has oneDNN."""
    return not torch.backends.mkl.is_available() and torch.backends.mkldnn.is_available()


def attn_implementation() -> str:
    """The attention implementation for a transformers vision model on the CPU."""
    return ATTENTION if needed() else "sdpa"


def use_onednn_linears(module: nn.Module) -> None:
    """Compute the linear layers of the module with oneDNN; after loading the weights, which are copied."""
    if not needed():
        return
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                setattr(parent, name, MkldnnLinear(child, torch.float32))


def _matmul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The product of a and the transpose of b with oneDNN (2-D FP32 tensors)."""
    return torch._C._nn.mkldnn_linear(a.contiguous().to_mkldnn(), b.contiguous().to_mkldnn(), None).to_dense()  # noqa: SLF001


def onednn_attention(
    module: nn.Module,  # the signature of transformers' attention functions
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,  # inference only
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attention of (batch, heads, tokens, dim) tensors, one head at a time; returns (batch, tokens, heads, dim)."""
    batch, heads, tokens, dim = query.shape
    groups = heads // key.shape[1]
    scale = dim**-0.5 if scaling is None else scaling
    out = torch.empty(batch, heads, tokens, value.shape[-1], dtype=query.dtype)
    for b in range(batch):
        mask = None if attention_mask is None else attention_mask[b if attention_mask.shape[0] > 1 else 0]
        for h in range(heads):
            scores = _matmul_t(query[b, h] * scale, key[b, h // groups])
            if mask is not None:
                m = mask[h if mask.shape[0] > 1 else 0, : scores.shape[0], : scores.shape[1]]
                scores = scores.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else scores + m
            out[b, h] = _matmul_t(scores.softmax(dim=-1), value[b, h // groups].transpose(0, 1))
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(ATTENTION, onednn_attention)
