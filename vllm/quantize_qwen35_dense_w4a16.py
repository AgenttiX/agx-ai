r"""Quantize a dense Qwen3.5 text checkpoint (e.g. the Qwen3.5-4B draft model of Qwen3.8-27B) to int4 for vLLM.

Made for nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft, an external draft model (Qwen3_5ForCausalLM) that has to fit on the
RTX 3090 next to its 13.1 GiB int4 target (see ../vllm-rtx3090/README.md). In BF16 it takes 8.4 GB.

llm-compressor's model_free_ptq rounds the tensors of the checkpoint to the nearest quantization level, shard by shard
on the GPU (a minute, no calibration data). Compressed-tensors pack-quantized, symmetric, group size 128:

- the attention, Gated DeltaNet and MLP projections: int4 (``--bits 8`` for int8),
- the token embedding, which is tied to the lm_head (0.64 B of the 4.21 B parameters): unquantized (BF16), as vLLM
  would apply the quantization of the embedding to the tied lm_head as well,
- the small in_proj_a/in_proj_b of the Gated DeltaNet layers, the norms and the convolutions: unquantized.

Runs in a container from the image of ../vllm, with llm-compressor installed into it:

    docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/vllm-models:/models \
        -v $PWD:/work -w /work --entrypoint sh ghcr.io/agenttix/agx-ai/vllm:latest \
        -c "pip install -q llmcompressor==0.15.0 && python3 quantize_qwen35_dense_w4a16.py \
            nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft /models/Qwen3.5-4B-Qwen3.8-27B-Draft-W4A16"
"""

import argparse

# The module runs in a vLLM image with llm-compressor installed; the development environment does not have them.
from compressed_tensors.quantization import (  # pyrefly: ignore[missing-import]
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from llmcompressor import model_free_ptq  # pyrefly: ignore[missing-import]

LINEAR = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*mlp\.(gate|up|down)_proj$",
]


def main() -> None:
    """Quantize the checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model id or path of the BF16 checkpoint")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                        help="bits of the projections (default %(default)s)")
    args = parser.parse_args()
    weights = QuantizationArgs(num_bits=args.bits, type="int", symmetric=True, strategy="group", group_size=128)
    config = QuantizationConfig(
        format="pack-quantized",
        config_groups={"linear": QuantizationScheme(targets=LINEAR, weights=weights)},
    )
    model_free_ptq(model_stub=args.model, save_directory=args.output, config=config, device="cuda:0")


if __name__ == "__main__":
    main()
