r"""Quantize a Qwen3.5/3.8-27B checkpoint (e.g. Teriminal-RSI/Terminal-RSI-27B) to W4A16 for HyperQwen, with GPTQ.

The output has the format of the base checkpoint that HyperQwen's prepare/ scripts extend
(dbirks/Qwen3.8-27B-W4A16-AutoRound, a compressed-tensors re-export): compressed-tensors pack-quantized, int4,
symmetric, group size 128, on every Linear layer of the language model except the small in_proj_a/in_proj_b of the
Gated DeltaNet layers. The vision tower, lm_head, embedding and MTP module stay in BF16; HyperQwen's prepare/ then
quantizes the lm_head, the embedding and the MTP module to int8 and builds the drafter vocabulary (see README.md).
transformers does not load the MTP module; llm-compressor 0.15 copies its BF16 tensors unchanged
(model_mtp.safetensors), and otherwise this script copies them from the source checkpoint into
model_extra_tensors.safetensors.

GPTQ with static activation ordering (no runtime cost; the columns are permuted only during the quantization) on
chat and code calibration samples, layer by layer on the GPU with the model in RAM (llm-compressor's sequential
pipeline; 40 min for Terminal-RSI-27B on the RTX 3090). Runs in a container from the image of ../vllm (PyTorch with
CUDA, transformers >= 5.8), with llm-compressor installed into it:

    docker run --rm --gpus all --shm-size 16g -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/vllm-models:/models -v $PWD:/work -w /work --entrypoint sh ghcr.io/agenttix/agx-ai/vllm:latest \
        -c "pip install -q llmcompressor==0.15.0 && python3 quantize_w4a16.py Teriminal-RSI/Terminal-RSI-27B \
            /models/hyperqwen/Terminal-RSI-27B-W4A16"
"""

import argparse
import json
from pathlib import Path
import random
import shutil

# The module runs in a vLLM image with llm-compressor installed; the development environment does not have them.
from datasets import Dataset, load_dataset  # pyrefly: ignore[missing-import]
from huggingface_hub import snapshot_download  # pyrefly: ignore[missing-import]
from llmcompressor import oneshot  # pyrefly: ignore[missing-import]
from llmcompressor.modifiers.quantization import GPTQModifier  # pyrefly: ignore[missing-import]
from safetensors import safe_open  # pyrefly: ignore[missing-import]
from safetensors.torch import save_file  # pyrefly: ignore[missing-import]
import torch  # pyrefly: ignore[missing-import]
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration  # pyrefly: ignore[missing-import]

IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:^mtp\\..*",
    "re:.*linear_attn\\.in_proj_a$",
    "re:.*linear_attn\\.in_proj_b$",
]


def calibration_texts(tokenizer: object, samples: int, seed: int) -> list[str]:
    """Chat-formatted samples: half general chat (UltraChat), half code instructions with answers (Magicoder)."""
    rng = random.Random(seed)
    template = tokenizer.apply_chat_template  # pyrefly: ignore[missing-attribute]
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{samples * 4}]")
    texts = [template(row["messages"], tokenize=False) for row in rng.sample(list(chat), samples // 2)]
    code = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=f"train[:{samples * 4}]")
    texts += [
        template(
            [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}],
            tokenize=False,
        )
        for row in rng.sample(list(code), samples - samples // 2)
    ]
    rng.shuffle(texts)
    return texts


def copy_mtp_and_files(model: str, output: Path) -> None:
    """Add the BF16 MTP module (which transformers does not load) and the remaining small files of the source.

    The mtp.* tensors go to model_extra_tensors.safetensors and into the weight index, where HyperQwen's
    prepare/quant_mtp.py and vLLM's Qwen3.5 MTP drafter look for them.
    """
    src = Path(model) if Path(model).is_dir() else Path(snapshot_download(model))
    src_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    mtp: dict[str, torch.Tensor] = {}
    for shard in sorted({v for k, v in src_map.items() if k.startswith("mtp.")}):
        with safe_open(src / shard, "pt") as f:
            mtp.update({key: f.get_tensor(key) for key in f.keys() if key.startswith("mtp.")})  # noqa: SIM118
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if mtp and not any(k.startswith("mtp.") for k in index["weight_map"]):
        save_file(mtp, output / "model_extra_tensors.safetensors", metadata={"format": "pt"})
        index["weight_map"].update(dict.fromkeys(mtp, "model_extra_tensors.safetensors"))
        index_path.write_text(json.dumps(index, indent=2))
    print(f"MTP tensors: {len(mtp)}")
    # The chat template, generation and preprocessor configs that save_pretrained did not write.
    for path in src.iterdir():
        if (
            path.suffix in {".json", ".jinja", ".txt"}
            and not path.name.startswith("model.safetensors")
            and not (output / path.name).exists()
        ):
            shutil.copy(path, output / path.name)


def main() -> None:
    """Quantize the checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model id or path of the BF16 checkpoint")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--samples", type=int, default=256, help="calibration samples (default %(default)s)")
    parser.add_argument("--max-len", type=int, default=2048, help="tokens per sample (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # On the CPU in BF16; the sequential pipeline moves one decoder layer at a time to the GPU.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")
    texts = calibration_texts(tokenizer, args.samples, args.seed)
    rows = [dict(tokenizer(t, max_length=args.max_len, truncation=True, add_special_tokens=False)) for t in texts]
    oneshot(
        model=model,
        dataset=Dataset.from_list(rows),
        recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE, actorder="static"),
        max_seq_length=args.max_len,
        num_calibration_samples=len(rows),
        sequential_targets=["Qwen3_5DecoderLayer"],
    )
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.model).save_pretrained(args.output)
    copy_mtp_and_files(args.model, args.output)


if __name__ == "__main__":
    main()
