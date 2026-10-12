#!/usr/bin/env python3
"""Quantize the embedding and the lm_head of Google's compressed-tensors QAT checkpoint of Gemma 4, so that it fits.

`google/gemma-4-31B-it-qat-w4a16-ct` has all linear weights of the language model in int4 (group size 32, the grid of
Google's quantization-aware training), but the embedding and the lm_head in BF16: 2.6 GiB each, 20.6 GiB on the GPU
in total, which leaves no room for a KV cache in 21 GB. This script copies the checkpoint and quantizes the two with
round-to-nearest (--embed-bits, --lm-head-bits; group size 32 for int4 and 128 for int8), with the functions of
../vllm-radeon-vii/repack_gemma4_qat.py. The embedding and the lm_head stay untied, because vLLM cannot use a
quantized embedding as the lm_head. All other tensors (the int4 linear weights, the norms and the vision tower) are
copied unchanged. A tensor that is already quantized (e.g. the int8 lm_head of a checkpoint built by
repack_gemma4_qat.py, given as a directory with --repo) is copied unchanged too.

Run it in the vLLM image, which has torch, safetensors, huggingface_hub and compressed-tensors (see README.md).
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

# The script runs in the vLLM image, which has these packages; the development environment does not have them.
from huggingface_hub import snapshot_download  # pyrefly: ignore[missing-import]
from safetensors import safe_open  # pyrefly: ignore[missing-import]
from safetensors.torch import save_file  # pyrefly: ignore[missing-import]
import torch  # pyrefly: ignore[missing-import]

REPO = "google/gemma-4-31B-it-qat-w4a16-ct"
EMBED = "model.language_model.embed_tokens"


def load_repack_module():
    """Import ../vllm-radeon-vii/repack_gemma4_qat.py (quant_sym, packed, group_args and group_size)."""
    path = Path(__file__).resolve().parent.parent / "vllm-radeon-vii" / "repack_gemma4_qat.py"
    if not path.exists():  # in a container, the script directories may be mounted side by side
        path = Path(__file__).resolve().parent / "repack_gemma4_qat.py"
    spec = importlib.util.spec_from_file_location("repack_gemma4_qat", path)
    assert spec is not None, path
    assert spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Build the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="output model directory")
    parser.add_argument(
        "--repo", default=REPO, help="compressed-tensors checkpoint, a repository or a directory (default %(default)s)"
    )
    parser.add_argument("--embed-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--lm-head-bits", type=int, default=8, choices=[4, 8])
    args = parser.parse_args()
    repack = load_repack_module()

    src = Path(args.repo) if Path(args.repo).is_dir() else Path(snapshot_download(args.repo))
    weights = sorted(src.glob("*.safetensors"))
    tensors: dict[str, torch.Tensor] = {}
    for path in weights:
        with safe_open(path, framework="pt") as f:
            names = list(f.keys())
            # The lm_head is tied to the embedding in the original model, but stored separately in some checkpoints.
            head_source = "lm_head.weight" if "lm_head.weight" in names else EMBED + ".weight"
            has_quantized_head = any(n.startswith("lm_head.weight_") for n in names)
            for name in names:
                t = f.get_tensor(name)
                if name == head_source and not has_quantized_head:
                    tensors.update(repack.packed("lm_head", *repack.quant_sym(t, args.lm_head_bits), args.lm_head_bits))
                if name == EMBED + ".weight":
                    tensors.update(repack.packed(EMBED, *repack.quant_sym(t, args.embed_bits), args.embed_bits))
                elif name != "lm_head.weight":
                    tensors[name] = t
                print(name, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "model.safetensors", metadata={"format": "pt"})
    for p in src.iterdir():
        if p.suffix in (".json", ".jinja", ".txt", ".model") and p.name != "model.safetensors.index.json":
            (args.out / p.name).write_bytes(p.read_bytes())

    config = json.loads((src / "config.json").read_text())
    config["tie_word_embeddings"] = config["text_config"]["tie_word_embeddings"] = False
    q = config["quantization_config"]
    q["config_groups"]["group_embed"] = repack.group_args(
        args.embed_bits, repack.group_size(args.embed_bits), [f"re:.*{EMBED.rsplit('.', 1)[1]}$"]
    )
    q["config_groups"]["group_lm_head"] = repack.group_args(
        args.lm_head_bits, repack.group_size(args.lm_head_bits), ["re:.*lm_head$"]
    )
    q["ignore"] = [n for n in q.get("ignore", []) if n != "lm_head"]
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    size = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    print(f"Wrote {args.out}: {size:.2f} GiB")


if __name__ == "__main__":
    main()
