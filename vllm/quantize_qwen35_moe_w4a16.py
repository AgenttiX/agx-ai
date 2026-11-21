r"""Quantize a Qwen3.5/3.6 MoE checkpoint (e.g. Phai-A1-35B-1M) for vLLM on a 24 GB GPU, without calibration.

llm-compressor's model_free_ptq rounds the tensors of the checkpoint to the nearest quantization level, shard by shard
on the GPU (minutes, no calibration data, little RAM). Unlike oneshot, it also quantizes the MTP layer, which
transformers does not load. Compressed-tensors pack-quantized, symmetric, group size 128:

- the routed experts (also those of the MTP layer, 97 % of the parameters): int4,
- the attention, Gated DeltaNet and shared-expert projections, the MTP fc and the lm_head: int8,
- the token embedding: int4 (it needs the embedding fix of ../vllm/vllm.patch),
- the router gates, the small in_proj_a/in_proj_b of the Gated DeltaNet layers and the vision tower: unquantized.

Runs in a container from the image of ../vllm, with llm-compressor installed into it. The input may be a checkpoint
directory with symlinks into the Hugging Face cache (e.g. from graft_mtp.py), so the cache is mounted at its host path:

    docker run --rm --gpus all -v /home/mika/.cache:/home/mika/.cache -v ~/.cache/vllm-models:/models \
        -v $PWD:/work -w /work --entrypoint sh ghcr.io/agenttix/agx-ai/vllm:latest \
        -c "pip install -q llmcompressor==0.15.0 && python3 quantize_qwen35_moe_w4a16.py \
            /home/mika/.cache/vllm-models/Phai-A1-35B-1M-mtp /models/Phai-A1-35B-1M-W4A16-mtp"
"""

import argparse

# The module runs in a vLLM image with llm-compressor installed; the development environment does not have them.
from compressed_tensors.quantization import (  # pyrefly: ignore[missing-import]
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from llmcompressor import model_free_ptq  # pyrefly: ignore[missing-import]

EXPERTS = [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"]
DENSE = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
    r"re:^mtp\.fc$",
    r"re:.*lm_head$",
]
EMBEDDING = [r"re:.*embed_tokens$"]


def weights(bits: int) -> QuantizationArgs:
    """Symmetric integer weights with a scale per group of 128."""
    return QuantizationArgs(num_bits=bits, type="int", symmetric=True, strategy="group", group_size=128)


def main() -> None:
    """Quantize the checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model id or path of the BF16 checkpoint")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--dense-bits", type=int, default=8, choices=[4, 8],
                        help="bits of the non-expert projections and the lm_head (default %(default)s)")
    args = parser.parse_args()
    config = QuantizationConfig(
        format="pack-quantized",
        config_groups={
            "experts": QuantizationScheme(targets=EXPERTS, weights=weights(4)),
            "dense": QuantizationScheme(targets=DENSE, weights=weights(args.dense_bits)),
            "embedding": QuantizationScheme(targets=EMBEDDING, weights=weights(4)),
        },
    )
    model_free_ptq(model_stub=args.model, save_directory=args.output, config=config, device="cuda:0")


if __name__ == "__main__":
    main()
