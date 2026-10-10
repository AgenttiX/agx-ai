#!/usr/bin/env python3
"""Build a vLLM checkpoint of Gemma 4 26B-A4B QAT that is bit-exact with Google's Q4_0 and fits in 16 GB.

Google's quantization-aware training (QAT) targets ggml's Q4_0: int4 with an fp16 scale per block of 32 weights,
symmetric. That is the same format as compressed-tensors' int4 "pack-quantized" with group size 32, which vLLM runs
with its exllama and gfx906 W4A16 MoE kernels. The checkpoint is built from Google's two QAT repositories:
- every linear weight of the language model (attention, dense MLP and experts) from the Q4_0 GGUF
  `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, converted without any rounding,
- the norms, routers, embedding and vision tower from `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`, in fp16.
  Only these ~2.5 GB of the 52 GB repository are downloaded, with HTTP range requests.
The embedding and the lm_head are untied, because vLLM cannot use a quantized embedding as the lm_head. Both are
quantized with round-to-nearest (--embed-bits, --lm-head-bits; group size 32 for int4 and 128 for int8), as the
fp16 versions (1.4 GiB each) do not fit. The vision tower stays fp16 (vLLM skips it with --language-model-only).

The ready-made int4 checkpoints of this model on Hugging Face (e.g. cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4) are not
on Google's grid: their quantizer chose other scales, so only 60-70 % of the int4 values are the same (see README.md).

Run it in the vLLM image, which has torch, numpy, safetensors, huggingface_hub and compressed-tensors (see README.md).
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import requests
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from huggingface_hub import hf_hub_download, hf_hub_url, snapshot_download
from safetensors.torch import save_file

GGUF_REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
GGUF_FILE = "gemma-4-26B_q4_0-it.gguf"
HF_REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
PREFIX = "model.language_model."
GGML_Q4_0 = 2

# GGUF tensor -> HF module, per layer
LINEAR = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
# HF tensors that come from the GGUF instead
FROM_GGUF = re.compile(r"\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|experts)\.")


def read_gguf(path: Path) -> dict[str, tuple[list[int], int, np.ndarray]]:
    """Minimal GGUF reader: tensor name -> (dims, ggml type, raw bytes of a Q4_0 tensor)."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    buf = memoryview(mm)
    pos = 4

    def rd(fmt: str):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, buf, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def rstr() -> str:
        nonlocal pos
        n = rd("Q")
        s = bytes(buf[pos : pos + n]).decode()
        pos += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t: int):
        if t == 8:
            return rstr()
        if t == 9:
            at, n = rd("I"), rd("Q")
            return [rval(at) for _ in range(n)]
        return rd(scalar[t])

    assert bytes(buf[:4]) == b"GGUF"
    rd("I")  # version
    n_tensors, n_kv = rd("Q"), rd("Q")
    alignment = 32
    for _ in range(n_kv):
        key = rstr()
        value = rval(rd("I"))
        if key == "general.alignment":
            alignment = value
    infos = []
    for _ in range(n_tensors):
        name = rstr()
        dims = [rd("Q") for _ in range(rd("I"))]
        infos.append((name, dims, rd("I"), rd("Q")))
    base = (pos + alignment - 1) // alignment * alignment
    out = {}
    for name, dims, ggml_type, offset in infos:
        nbytes = int(np.prod(dims)) // 32 * 18 if ggml_type == GGML_Q4_0 else 0
        out[name] = (dims, ggml_type, mm[base + offset : base + offset + nbytes])
    return out


def fetch_tensors(repo: str, keep) -> dict[str, torch.Tensor]:
    """The tensors of a safetensors repository whose names satisfy keep(name), downloaded with range requests."""
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json")).read_text())
    shards: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if keep(name):
            shards.setdefault(shard, []).append(name)
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    out = {}
    with requests.Session() as session:
        for shard, names in shards.items():
            url = hf_hub_url(repo, shard)

            def get(start: int, end: int) -> bytes:
                r = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=600)
                r.raise_for_status()
                assert len(r.content) == end - start
                return r.content

            n = struct.unpack("<Q", get(0, 8))[0]
            header = json.loads(get(8, 8 + n))
            for name in names:
                info = header[name]
                a, b = info["data_offsets"]
                t = torch.frombuffer(bytearray(get(8 + n + a, 8 + n + b)), dtype=dtypes[info["dtype"]])
                out[name] = t.reshape(info["shape"]).to(torch.float16)
            print(f"{shard}: {len(names)} tensors", flush=True)
    return out


def q4_0_to_int(raw: np.ndarray, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Q4_0 blocks of a [rows, cols] matrix -> (int8 values in -8..7, fp16 scales [rows, cols / 32])."""
    blocks = np.asarray(raw).reshape(rows, cols // 32, 18)
    scales = blocks[:, :, :2].copy().view(np.float16)[..., 0]
    qs = blocks[:, :, 2:]
    # Within a block, the low nibbles hold elements 0-15 and the high nibbles elements 16-31.
    q = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.int8) - 8
    return torch.from_numpy(q.reshape(rows, cols)), torch.from_numpy(scales.copy())


def group_size(bits: int) -> int:
    return 32 if bits == 4 else 128


def quant_sym(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest symmetric quantization with absmax scales (for the embedding and the lm_head)."""
    group = group_size(bits)
    out_f, in_f = w.shape
    qmax = 2 ** (bits - 1) - 1
    b = w.float().reshape(out_f, in_f // group, group)
    s = b.abs().amax(dim=-1, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s).half().float()
    q = torch.clamp(torch.round(b / s), -qmax - 1, qmax)
    return q.reshape(out_f, in_f).to(torch.int8), s.squeeze(-1).to(torch.float16)


def packed(name: str, q: torch.Tensor, s: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    return {
        f"{name}.weight_packed": pack_to_int32(q, bits),
        f"{name}.weight_scale": s,
        f"{name}.weight_shape": torch.tensor(list(q.shape), dtype=torch.int64),
    }


def group_args(bits: int, group: int, targets: list[str]) -> dict:
    return {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": targets,
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": group,
            "num_bits": bits,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "group",
            "symmetric": True,
            "type": "int",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="output model directory")
    parser.add_argument("--lm-head-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--embed-bits", type=int, default=4, choices=[4, 8, 16], help="16 = keep fp16")
    args = parser.parse_args()

    gguf = read_gguf(Path(hf_hub_download(GGUF_REPO, GGUF_FILE)))
    tensors: dict[str, torch.Tensor] = {}

    # Linear weights of the language model from the GGUF
    n_layers = 1 + max(int(m.group(1)) for n in gguf if (m := re.match(r"blk\.(\d+)\.", n)))
    for i in range(n_layers):
        layer = f"{PREFIX}layers.{i}."
        for g, hf in LINEAR.items():
            name = f"blk.{i}.{g}.weight"
            if name not in gguf:  # the full-attention layers have no attn_v (attention_k_eq_v)
                continue
            (cols, rows), ggml_type, raw = gguf[name]
            assert ggml_type == GGML_Q4_0, name
            tensors.update(packed(layer + hf, *q4_0_to_int(raw, rows, cols), 4))
        (cols, rows2, n_exp), ggml_type, raw = gguf[f"blk.{i}.ffn_gate_up_exps.weight"]
        assert ggml_type == GGML_Q4_0
        q, s = q4_0_to_int(raw, n_exp * rows2, cols)
        inter = rows2 // 2  # each expert's rows are [gate; up]
        for e in range(n_exp):
            r = e * rows2
            tensors.update(packed(f"{layer}experts.{e}.gate_proj", q[r : r + inter], s[r : r + inter], 4))
            tensors.update(packed(f"{layer}experts.{e}.up_proj", q[r + inter : r + rows2], s[r + inter : r + rows2], 4))
        (cols, rows, n_exp), ggml_type, raw = gguf[f"blk.{i}.ffn_down_exps.weight"]
        assert ggml_type == GGML_Q4_0
        q, s = q4_0_to_int(raw, n_exp * rows, cols)
        for e in range(n_exp):
            r = e * rows
            tensors.update(packed(f"{layer}experts.{e}.down_proj", q[r : r + rows], s[r : r + rows], 4))
        print(f"GGUF layer {i}", flush=True)

    # Everything else from the unquantized QAT checkpoint
    for name, t in fetch_tensors(HF_REPO, lambda n: not FROM_GGUF.search(n)).items():
        if name == PREFIX + "embed_tokens.weight":
            if args.embed_bits == 16:
                tensors[name] = t
            else:
                tensors.update(packed(PREFIX + "embed_tokens", *quant_sym(t, args.embed_bits), args.embed_bits))
            tensors.update(packed("lm_head", *quant_sym(t, args.lm_head_bits), args.lm_head_bits))
        else:
            tensors[name] = t

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "model.safetensors", metadata={"format": "pt"})

    src = Path(snapshot_download(HF_REPO, allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"]))
    for p in src.iterdir():
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json"):
            (args.out / p.name).write_bytes(p.read_bytes())
    config = json.loads((src / "config.json").read_text())
    config["dtype"] = config["text_config"]["dtype"] = "float16"  # gfx906 has no native bf16
    config["tie_word_embeddings"] = config["text_config"]["tie_word_embeddings"] = False
    groups = {
        "group_0": group_args(4, 32, ["Linear"]),
        "group_lm_head": group_args(args.lm_head_bits, group_size(args.lm_head_bits), ["re:.*lm_head$"]),
    }
    if args.embed_bits != 16:
        groups["group_embed"] = group_args(args.embed_bits, group_size(args.embed_bits), ["re:.*embed_tokens$"])
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": groups,
        "ignore": ["re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*router.*"],
        "kv_cache_scheme": None,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    size = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    print(f"Wrote {args.out}: {size:.2f} GiB")


if __name__ == "__main__":
    main()
