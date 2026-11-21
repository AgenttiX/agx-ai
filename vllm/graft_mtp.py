r"""Graft the MTP head of a base checkpoint onto a fine-tuned checkpoint that was published without it.

Some fine-tunes keep ``mtp_num_hidden_layers`` in their config but ship no ``mtp.*`` tensors, e.g.
PhaiModelLab/Phai-A1-35B-1M (a fine-tune of Qwen/Qwen3.6-35B-A3B). The base model's MTP head still drafts well for
them (it reads the fine-tune's own hidden states and embeddings), so this copies the ``mtp.*`` tensors of the base
checkpoint into a new shard and writes a checkpoint directory that vLLM and llama.cpp's convert_hf_to_gguf.py load
like the original. The other files are symlinks to the target checkpoint (absolute paths, resolved through the
Hugging Face cache), so the output takes only the size of the MTP head. Mount the Hugging Face cache at the same path
when the output is used in a container.

Only the Python standard library is needed: the tensors are copied byte for byte from the safetensors files.

    uv run vllm/graft_mtp.py PhaiModelLab/Phai-A1-35B-1M Qwen/Qwen3.6-35B-A3B \
        ~/.cache/vllm-models/Phai-A1-35B-1M-mtp
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys

INDEX = "model.safetensors.index.json"
MTP_SHARD = "model-mtp.safetensors"
# The prefixes of the MTP tensors in the checkpoints of Qwen3.5/3.6/3.8 and Qwen3-Next.
MTP_PREFIXES = ("mtp.",)


def hf_snapshot(model: str) -> Path:
    """The local directory of a model: a path as is, or the newest snapshot of a model id in the Hugging Face cache.

    :param model: a directory or a Hugging Face model id (``org/name``), which must already be downloaded
    :return: the directory
    :raises FileNotFoundError: if the model is not in the cache
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snapshots = cache / f"models--{model.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"{model} is not in {cache}; download it first, e.g. with hf download")
    return max(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)


def read_header(path: Path) -> tuple[dict, int]:
    """The JSON header of a safetensors file and the offset of its data section.

    :param path: the safetensors file
    :return: the header and the data offset
    """
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size)), 8 + size


def copy_tensors(sources: dict[str, Path], output: Path) -> int:
    """Write the given tensors, byte for byte, into a new safetensors file.

    :param sources: the tensor names and the safetensors files that hold them
    :param output: the new file
    :return: the number of bytes of tensor data
    """
    header: dict = {"__metadata__": {"format": "pt"}}
    chunks: list[tuple[Path, int, int]] = []
    offset = 0
    for name in sorted(sources):
        src_header, data_start = read_header(sources[name])
        info = src_header[name]
        begin, end = info["data_offsets"]
        header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
        chunks.append((sources[name], data_start + begin, end - begin))
        offset += end - begin
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # The data section is aligned to 8 bytes.
    with output.open("wb") as out:
        out.write(struct.pack("<Q", len(raw)) + raw)
        for path, start, length in chunks:
            with path.open("rb") as f:
                f.seek(start)
                out.write(f.read(length))
    return offset


def graft(target: Path, base: Path, output: Path) -> int:
    """Write the grafted checkpoint.

    :param target: the checkpoint without the MTP head
    :param base: the checkpoint whose MTP head is copied
    :param output: the new directory (created; must not contain other files)
    :return: the number of MTP tensors
    :raises ValueError: if the target already has an MTP head, or the base has none
    """
    target_index = json.loads((target / INDEX).read_text())
    if any(k.startswith(MTP_PREFIXES) for k in target_index["weight_map"]):
        raise ValueError(f"{target} already has MTP tensors")
    base_map = json.loads((base / INDEX).read_text())["weight_map"]
    mtp = {k: base / v for k, v in base_map.items() if k.startswith(MTP_PREFIXES)}
    if not mtp:
        raise ValueError(f"{base} has no MTP tensors")
    output.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        link = output / path.name
        if path.name != INDEX and not link.exists():
            link.symlink_to(path.resolve())
    size = copy_tensors(mtp, output / MTP_SHARD)
    target_index["weight_map"].update(dict.fromkeys(mtp, MTP_SHARD))
    target_index.setdefault("metadata", {})["total_size"] = target_index["metadata"].get("total_size", 0) + size
    (output / INDEX).write_text(json.dumps(target_index, indent=2))
    return len(mtp)


def main() -> None:
    """Graft the checkpoints given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="the fine-tuned checkpoint without an MTP head (directory or model id)")
    parser.add_argument("base", help="the checkpoint with the MTP head (directory or model id)")
    parser.add_argument("output", type=Path, help="the output directory")
    args = parser.parse_args()
    try:
        count = graft(hf_snapshot(args.target), hf_snapshot(args.base), args.output.expanduser())
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    print(f"Grafted {count} MTP tensors into {args.output}")


if __name__ == "__main__":
    main()
