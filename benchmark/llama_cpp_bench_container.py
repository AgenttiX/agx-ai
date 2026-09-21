#!/usr/bin/env python3
"""Run llama.cpp's own benchmark (``llama bench``, formerly llama-bench) inside a running llama-server container.

Unlike ``llama_cpp_bench_http.py``, which measures the server as it serves, this measures the raw model with the
flags given here, independently of the server configuration. That makes it the tool for comparing quantisations,
builds and hardware.

The benchmark loads the model itself, so the server's copy of the models is unloaded first through the router API
to free the GPU memory, and the models are reloaded at the end. The GGUF path defaults to the one the server is
serving, read from ``/props``; the Hugging Face cache is mounted at the same path inside the container.

Note: LiteLLM on the personal agx-ai server (``../litellm/config.yaml``, ``health_check_interval: 60``) probes the
models every minute over the network and makes the router reload models that were just unloaded. Turn it off there
(``background_health_checks: false``) or stop that LiteLLM while benchmarking; otherwise the benchmark may run with
the server's models back in VRAM.

Examples (arguments after ``--`` go to ``llama bench``; match the server preset's device flags for comparability):
    ./llama_cpp_bench_container.py --env-file ../llama-cpp-big-machine/llama-cpp.env --label "UD-Q4_K_M" \\
        -- -sm tensor -ts 1/1 -fa on -p 4096 -n 256
    ./llama_cpp_bench_container.py --container llama-cpp-gpu --model /root/.cache/huggingface/hub/.../model.gguf -- -p 2048

The Markdown table is printed and the JSON lines that ``llama bench`` emits are appended, with the label and date,
to ``results/<hostname>-llama-bench.jsonl``. Only the Python standard library is needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llama_cpp_bench_http import Server, read_env_key


def pick_chat_model(models: list[dict[str, Any]]) -> str:
    rank = {"loaded": 0, "sleeping": 1}
    ordered = sorted(models, key=lambda m: rank.get(m.get("status", {}).get("value"), 2))
    try:
        return next(m["id"] for m in ordered if "embed" not in m["id"].lower())
    except StopIteration:
        raise SystemExit("No chat model found on the server; pass --model-id or --model") from None


def model_status(server: Server, model_id: str) -> str:
    return next((m.get("status", {}).get("value", "") for m in server.models() if m["id"] == model_id), "")


def unload_models(server: Server, model_ids: list[str]) -> None:
    for model_id in model_ids:
        print(f"Unloading {model_id} from the server", file=sys.stderr)
        try:
            server.request("/models/unload", {"model": model_id})
        except SystemExit as e:
            print(f"WARNING: could not unload {model_id}: {e}", file=sys.stderr)
    time.sleep(3)


def reload_models(server: Server, model_ids: list[str]) -> None:
    """The router usually reloads load-on-startup models by itself; only ask for the ones it has not started on."""
    for model_id in model_ids:
        status = model_status(server, model_id)
        if status in ("loaded", "loading"):
            print(f"{model_id}: {status}", file=sys.stderr)
            continue
        print(f"Reloading {model_id} on the server", file=sys.stderr)
        try:
            server.request("/models/load", {"model": model_id})
        except SystemExit:
            pass
        print(f"{model_id}: {model_status(server, model_id)}", file=sys.stderr)


def run_bench(container: str, model_path: str, bench_args: list[str], label: str, output: Path) -> int:
    """One ``llama bench`` invocation: Markdown to stdout, JSON lines (via stderr) to the results file."""
    cmd = ["docker", "exec", container, "/app/llama", "bench", "-m", model_path, "-o", "md", "-oe", "jsonl", *bench_args]
    print("Running:", " ".join(cmd[3:]), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    records = []
    for line in proc.stderr.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            rec["label"] = label
            rec["date"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            records.append(rec)
        elif line.strip():
            print(line, file=sys.stderr)  # llama.cpp logs
    if records:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(rec) + "\n" for rec in records)
        print(f"Appended {len(records)} records to {output}", file=sys.stderr)
    if proc.returncode != 0:
        print(f"ERROR: llama bench exited with {proc.returncode}", file=sys.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--container", default="llama-cpp", help="container name (default %(default)s)")
    parser.add_argument("--url", default="http://localhost:9931", help="llama-server URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="GGUF path inside the container (default: the served chat model's path)")
    parser.add_argument("--model-id", help="router model id whose path is used (default: first non-embedding model)")
    parser.add_argument("--label", default="", help="label stored with the result")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="do not unload the server's models first (only if there is enough GPU memory)")
    parser.add_argument("--output", type=Path, help="JSON lines file (default results/<hostname>-llama-bench.jsonl)")
    parser.add_argument("bench_args", nargs="*", help="arguments for llama bench, after --")
    args = parser.parse_args()

    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get("LLAMA_API_KEY")
    server = Server(args.url, api_key, timeout=60.0)
    models = server.models()
    model_id = args.model_id or pick_chat_model(models)
    model_path = args.model or server.props(model_id)["model_path"]
    print(f"Model: {model_id} -> {model_path}", file=sys.stderr)

    loaded = [] if args.keep_loaded else [
        m["id"] for m in models if m.get("status", {}).get("value") in ("loaded", "sleeping")
    ]
    if loaded:
        unload_models(server, loaded)
    output = args.output or Path(__file__).resolve().parent / "results" / f"{socket.gethostname().split('.')[0]}-llama-bench.jsonl"
    try:
        return run_bench(args.container, model_path, args.bench_args, args.label, output)
    finally:
        if loaded:
            reload_models(server, loaded)


if __name__ == "__main__":
    sys.exit(main())
