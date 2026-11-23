"""The vision tower of Qwen3.8 (Qwen3-VL architecture) on the Radeon VII, as a service for the vLLM container.

The vLLM container on the RTX 3090 runs the vision tower outside its GPU (cpu_vision.py, QWEN38_VISION_DEVICE=rvii):
it normalizes the pixel values and sends them here over a Unix socket, and gets the embeddings back, so that the
tower takes no VRAM of the RTX 3090 and an image takes a fraction of a second instead of ~7 s on the CPU. The tower
runs in FP16 (gfx906 has fast FP16 but no fast BF16) with transformers' Qwen3VLVisionModel, in the ROCm image of
../../vllm-radeon-vii (ghcr.io/agenttix/agx-ai/vllm-gfx906), with this file bind-mounted. See ../README.md.

Protocol (one request at a time per connection, any number of requests per connection):
  request:  uint32 little-endian header length, header JSON {"rows": N, "cols": D, "grid": [[t, h, w], ...]},
            then N * D float16 values (the normalized pixel patches, row-major);
  response: uint32 header length, header JSON {"tokens": K, "dim": E} or {"error": "..."},
            then K * E bfloat16 values (the merged-patch embeddings).

Environment: QWEN_VISION_MODEL (checkpoint directory with config.json and the safetensors of model.visual.*),
QWEN_VISION_SOCKET (default /run/qwen-vision/vision.sock), QWEN_VISION_PREFIX (default "model.visual.").
"""

import json
import os
from pathlib import Path
import socket
import struct
import threading
import time

from safetensors import safe_open  # pyrefly: ignore[missing-import]

# The module runs in the ROCm vLLM image, which has these packages; the development environment does not have them.
import torch  # pyrefly: ignore[missing-import]
from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # pyrefly: ignore[missing-import]
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # pyrefly: ignore[missing-import]

DTYPE = torch.float16


def load_model(model_dir: str, prefix: str) -> Qwen3VLVisionModel:
    """The checkpoint's vision tower on the GPU in FP16."""
    with (Path(model_dir) / "config.json").open() as fh:
        config = json.load(fh)
    cfg_dict = dict(config["vision_config"])
    cfg_dict.pop("model_type", None)
    cfg = Qwen3VLVisionConfig(**cfg_dict, attn_implementation="sdpa")
    if cfg.deepstack_visual_indexes:
        raise ValueError("deepstack features are not implemented")
    with torch.device("cuda"):
        model = Qwen3VLVisionModel(cfg).to(DTYPE).eval()
    params = dict(model.named_parameters())
    with (Path(model_dir) / "model.safetensors.index.json").open() as fh:
        weight_map = json.load(fh)["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        if name.startswith(prefix):
            by_file.setdefault(file, []).append(name)
    loaded = set()
    for file, names in by_file.items():
        with safe_open(str(Path(model_dir) / file), framework="pt") as f:
            for name in names:
                short = name[len(prefix):]
                if short not in params:
                    raise KeyError(f"unexpected vision tower weight {name!r}")
                with torch.no_grad():
                    params[short].copy_(f.get_tensor(name).to(device="cuda", dtype=DTYPE))
                loaded.add(short)
    missing = set(params) - loaded
    if missing:
        raise KeyError(f"vision tower weights missing from the checkpoint: {sorted(missing)[:5]}")
    # The patch embedding is a Conv3d whose kernel equals its stride, i.e. a linear layer over each flattened patch
    # (channels, time, height, width, the order of the pixel rows). MIOpen runs that Conv3d with a naive kernel on
    # gfx906: 2.0 of 3.1 s for a 1-megapixel image, against milliseconds as a GEMM.
    proj = model.patch_embed.proj
    weight = proj.weight.detach().reshape(proj.out_channels, -1)
    bias = proj.bias.detach() if proj.bias is not None else None

    def patch_embed(hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.reshape(-1, weight.shape[1]).to(DTYPE), weight, bias)

    model.patch_embed.forward = patch_embed
    return model


def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Exactly n bytes from the connection (b"" if it closed before the first byte)."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0:
            if got == 0:
                return b""
            raise ConnectionError("connection closed in the middle of a message")
        got += k
    return bytes(buf)


def send_message(conn: socket.socket, header: dict, payload: bytes = b"") -> None:
    """A length-prefixed JSON header and a payload."""
    head = json.dumps(header).encode()
    conn.sendall(struct.pack("<I", len(head)) + head)
    if payload:
        conn.sendall(payload)


def serve(model: Qwen3VLVisionModel, path: str) -> None:
    """Accept connections on the Unix socket and compute embeddings for their requests, one at a time."""
    lock = threading.Lock()
    sock_path = Path(path)
    sock_path.unlink(missing_ok=True)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    sock_path.chmod(0o666)
    server.listen(8)
    print(f"[rvii-vision] listening on {path}", flush=True)

    def handle(conn: socket.socket) -> None:
        with conn:
            while True:
                raw = recv_exact(conn, 4)
                if not raw:
                    return
                header = json.loads(recv_exact(conn, struct.unpack("<I", raw)[0]))
                rows, cols = int(header["rows"]), int(header["cols"])
                data = recv_exact(conn, rows * cols * 2)
                try:
                    t0 = time.perf_counter()
                    pixels = torch.frombuffer(bytearray(data), dtype=torch.float16).view(rows, cols)
                    grid = torch.tensor(header["grid"], dtype=torch.int64)
                    with lock, torch.inference_mode():
                        out = model(pixels.to("cuda"), grid_thw=grid.to("cuda"))
                        embeds = out.pooler_output if hasattr(out, "pooler_output") else out
                        embeds = embeds.to(torch.bfloat16).cpu().contiguous()
                    tokens, dim = embeds.shape
                    send_message(conn, {"tokens": tokens, "dim": dim},
                                 embeds.view(torch.int16).numpy().tobytes())
                    print(f"[rvii-vision] {rows} patches {header['grid']} -> {tokens} tokens in "
                          f"{time.perf_counter() - t0:.3f} s", flush=True)
                except Exception as exc:  # noqa: BLE001 - reported to the client, which falls back to the CPU
                    send_message(conn, {"error": repr(exc)})

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main() -> None:
    """Load the tower and serve it."""
    model_dir = os.environ["QWEN_VISION_MODEL"]
    path = os.environ.get("QWEN_VISION_SOCKET", "/run/qwen-vision/vision.sock")
    prefix = os.environ.get("QWEN_VISION_PREFIX", "model.visual.")
    t0 = time.perf_counter()
    model = load_model(model_dir, prefix)
    print(f"[rvii-vision] loaded {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters from "
          f"{model_dir} in {time.perf_counter() - t0:.1f} s on {torch.cuda.get_device_name()}", flush=True)
    # The first forward pass selects and compiles the GPU kernels (2.8 instead of 1.1 s for a 1-megapixel image).
    t0 = time.perf_counter()
    with torch.inference_mode():
        for h, w in ((32, 32), (64, 64)):
            patches = torch.zeros(h * w, 3 * 2 * 16 * 16, dtype=DTYPE, device="cuda")
            model(patches, grid_thw=torch.tensor([[1, h, w]], device="cuda"))
    torch.cuda.synchronize()
    print(f"[rvii-vision] warmed up in {time.perf_counter() - t0:.1f} s", flush=True)
    serve(model, path)


if __name__ == "__main__":
    main()
