#!/usr/bin/env python3
"""Quantize the BF16 linear layers of the W4A16 checkpoint of Qwen3.8-Flash-Next, so that its GPU part fits in 8 GB.

`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` has the routed experts in int4 (Intel's AutoRound, GPTQ packing, group
size 128) and the n-gram embedding table in FP8, but every other weight in BF16: 9.2 GiB, of which the embedding
(1.2 GiB) is only looked up and can stay in RAM. The rest does not fit on an 8 GB GPU next to anything else.
This script quantizes the largest of them with round-to-nearest to --bits (symmetric, group size 128, the GPTQ
packing of the experts, which vLLM runs with its Marlin kernels on Ampere):
- the Gated DeltaNet projections (linear_attn.in_proj_qkv, in_proj_z and out_proj, 3.9 GiB),
- the attention projections (self_attn.q_proj, k_proj, v_proj and o_proj, 1.1 GiB),
- the shared experts (mlp.shared_expert.gate_proj, up_proj and down_proj, 0.4 GiB),
- the lm_head (1.2 GiB), with --lm-head-bits.
The small or numerically sensitive weights stay in BF16: in_proj_a/b, the convolutions, the sparse-attention
indexer, the routers, the hyper-connections and the MTP module.

The output directory gets new shards in place of the quantized ones (~78 GB) and symbolic links to the n-gram table
shards (51 GB) and the other files of the source, which therefore has to stay in the Hugging Face cache.
With --separate-shard, it gets one new shard with only the quantized tensors (~3 GB) and symbolic links to all
source shards instead: the index then refers to the new tensors in it, and to the other tensors in the source
shards, which still hold the BF16 originals of the quantized ones. Only loaders that honor the index at the tensor
level can load that, such as the vLLM build of ../vllm-rtx3090-flash-next (stock vLLM 0.31 reads every tensor of a
listed shard). --keep-shared-experts leaves the shared experts in BF16 (that runtime fuses them into the MoE).
The quantization_config of config.json gets the new bits for the quantized layers.

Run it in the vLLM image, which has torch and safetensors (see README.md).
"""

import argparse
import json
import os
from pathlib import Path
import re

# The script runs in the vLLM image, which has these packages; the development environment does not have them.
from safetensors import safe_open  # pyrefly: ignore[missing-import]
from safetensors.torch import save_file  # pyrefly: ignore[missing-import]
import torch  # pyrefly: ignore[missing-import]

SRC = (
    "/root/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/"
    "ef554143369a706525336f6b42a09094835dc077"
)
GROUP_SIZE = 128
# Linear layers of the language model to quantize (the MTP module, "mtp.layers.N", is not matched)
TARGET: re.Pattern[str] = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.[qkvo]_proj|mlp\.shared_expert\.(gate|up|down)_proj)$"
)
LM_HEAD = "lm_head"


def quantize_gptq(w: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    """Symmetric round-to-nearest in the GPTQ format of the checkpoint: qweight, qzeros (zero - 1) and scales.

    w is [out, in]. qweight is [in * bits / 32, out] (packed along the input dimension), qzeros
    [in / GROUP_SIZE, out * bits / 32] (packed along the output dimension) and scales [in / GROUP_SIZE, out].
    """
    out_features, in_features = w.shape
    assert in_features % GROUP_SIZE == 0, w.shape
    assert out_features % (32 // bits) == 0, w.shape
    maxq = 2**bits - 1
    zero = 2 ** (bits - 1)
    g = w.float().t().reshape(in_features // GROUP_SIZE, GROUP_SIZE, out_features)  # [groups, group, out]
    scales = (g.abs().amax(dim=1) * 2 / maxq).clamp(min=1e-8)  # [groups, out], as in GPTQ with sym=True
    q = (torch.round(g / scales[:, None, :]) + zero).clamp(0, maxq).to(torch.int64)
    q = q.reshape(in_features, out_features)
    per = 32 // bits
    shifts = torch.arange(per, dtype=torch.int64) * bits
    qweight = (q.reshape(in_features // per, per, out_features) << shifts[None, :, None]).sum(dim=1)
    zeros = torch.full((in_features // GROUP_SIZE, out_features // per, per), zero - 1, dtype=torch.int64)
    qzeros = (zeros << shifts[None, None, :]).sum(dim=2)
    return {"qweight": to_int32(qweight), "qzeros": to_int32(qzeros), "scales": scales.to(torch.float16)}


def to_int32(x: torch.Tensor) -> torch.Tensor:
    """Reinterpret unsigned 32-bit values (in int64) as signed int32, like the packed tensors of the checkpoint."""
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def main() -> None:
    """Build the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="output model directory")
    parser.add_argument("--src", default=SRC, type=Path, help="source snapshot directory (default %(default)s)")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--lm-head-bits", type=int, default=8, choices=[4, 8, 16], help="16 = keep BF16")
    parser.add_argument("--separate-shard", action="store_true",
                        help="write only the quantized tensors, into one new shard, and link all source shards")
    parser.add_argument("--keep-shared-experts", action="store_true", help="keep the shared experts in BF16")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    quantized: dict[str, int] = {}  # module -> bits

    def bits_for(module: str) -> int:
        return args.lm_head_bits if module == LM_HEAD else args.bits

    def is_target(name: str) -> bool:
        module = name.removesuffix(".weight")
        is_linear = TARGET.match(module) is not None or module == LM_HEAD
        if args.keep_shared_experts and ".shared_expert." in module:
            return False
        return name.endswith(".weight") and is_linear and bits_for(module) < 16

    if args.separate_shard:
        new_shard = f"model-dense-int{args.bits}.safetensors"
        tensors = {}
        for shard in shards:
            dst = args.out / shard
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.realpath(args.src / shard))
            targets = [n for n, s in weight_map.items() if s == shard and is_target(n)]
            if not targets:
                continue
            with safe_open(args.src / shard, "pt") as f:
                for name in targets:
                    module = name.removesuffix(".weight")
                    for suffix, t in quantize_gptq(f.get_tensor(name), bits_for(module)).items():
                        tensors[f"{module}.{suffix}"] = t
                        weight_map[f"{module}.{suffix}"] = new_shard
                    del weight_map[name]
                    quantized[module] = bits_for(module)
            print(f"{shard}: {len(targets)} tensors quantized", flush=True)
        save_file(tensors, args.out / new_shard, metadata={"format": "pt"})
        shards = []  # all done

    for shard in shards:
        names = [n for n, s in weight_map.items() if s == shard]
        targets = [n for n in names if is_target(n)]
        dst = args.out / shard
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if not targets:
            dst.symlink_to(os.path.realpath(args.src / shard))
            continue
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(args.src / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
                if name in targets:
                    module = name.removesuffix(".weight")
                    for suffix, t in quantize_gptq(f.get_tensor(name), bits_for(module)).items():
                        tensors[f"{module}.{suffix}"] = t
                        weight_map[f"{module}.{suffix}"] = shard
                    del weight_map[name]
                    quantized[module] = bits_for(module)
                else:
                    tensors[name] = f.get_tensor(name)
        save_file(tensors, dst, metadata={"format": "pt"})
        print(f"{shard}: {len(targets)} tensors quantized", flush=True)

    config = json.loads((args.src / "config.json").read_text())
    extra: dict[str, dict] = config["quantization_config"]["extra_config"]
    for module, bits in quantized.items():
        # The exact keys are checked before the regular expressions.
        extra[module] = {"bits": bits, "group_size": GROUP_SIZE, "sym": True, "data_type": "int"}
    # These patterns would also match the quantized layers; the exact keys cover all layers anyway.
    for pattern in (".*linear_attn.*",) if args.keep_shared_experts else (".*linear_attn.*", ".*shared_expert.*"):
        extra.pop(pattern, None)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for p in args.src.iterdir():
        dst = args.out / p.name
        if p.is_file() and not p.name.endswith(".safetensors") and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(os.path.realpath(p))
    print(f"Wrote {args.out}: {len(quantized)} layers quantized")


if __name__ == "__main__":
    main()
