#!/usr/bin/env python3
"""Benchmark a running llama.cpp server (llama-server) over its HTTP API.

Measures, for the serving configuration as it actually runs (router presets, slots, KV cache, speculative decoding):
- generation speed (tokens/s) for a single request,
- prompt processing speed (tokens/s) for a long prompt,
- throughput with concurrent requests (aggregate tokens/s and per-request speed),
- embedding throughput (tokens/s) for the embedding model, if one is served,
- peak GPU memory during the run (nvidia-smi, if available on this machine).

Speeds for the LLM come from the ``timings`` object that llama-server attaches to every completion (server-side
prompt and generation timings, plus draft-token acceptance when speculative decoding is on). Embedding timings are
wall-clock, with token counts from ``/tokenize``. Nothing but the Python standard library is needed.

Examples:
    ./llama_cpp_bench_http.py --env-file ../llama-cpp-big-machine/llama-cpp.env --label "Qwen3.8-27B ctx 65536"
    ./llama_cpp_bench_http.py --url http://agx-z2e:9932 --api-key ... --model gemma --no-embedding
    ./llama_cpp_bench_http.py --concurrency 1 2 4 8 --n-predict 512 --prompt-tokens 8192 --mixed

For the raw model speed independent of the server configuration, see ``llama_cpp_bench_container.py``.

Results are printed as a Markdown table and appended as one JSON line per run to ``results/<hostname>.jsonl``,
so that runs with different models, settings or hardware can be compared afterwards.

Note: LiteLLM on the personal agx-ai server (``../litellm/config.yaml``, ``health_check_interval: 60``) sends a health
check request to every model once a minute over the network, which adds noise to the measurements and reloads
unloaded models. Turn it off there (``background_health_checks: false``) or stop that LiteLLM while benchmarking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

PARAGRAPH = (
    "A first-order phase transition in the early universe proceeds through the nucleation, expansion and collision "
    "of bubbles of the new phase. The expanding bubbles set the surrounding plasma in motion, and the resulting sound "
    "waves source a stochastic background of gravitational waves whose spectrum depends on the transition strength, "
    "the wall speed, the mean bubble separation and the equation of state of the plasma. "
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, url: str, api_key: str | None, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data else "GET")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise SystemExit(f"HTTP {e.code} for {path}: {body}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach {self.url}{path}: {e.reason}") from e

    def models(self) -> list[dict[str, Any]]:
        return self.request("/v1/models").get("data", [])

    def props(self, model: str) -> dict[str, Any]:
        return self.request(f"/props?model={urllib.request.quote(model, safe='')}")

    def tokenize(self, model: str, text: str) -> list[int]:
        return self.request("/tokenize", {"model": model, "content": text})["tokens"]

    def detokenize(self, model: str, tokens: list[int]) -> str:
        return self.request("/detokenize", {"model": model, "tokens": tokens})["content"]

    def completion(self, model: str, prompt: str, n_predict: int) -> dict[str, Any]:
        """Raw completion (no chat template, so no thinking mode); returns the server's timings."""
        payload = {
            "model": model,
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": False,  # measure real prompt processing, not a KV-cache hit
            "ignore_eos": True,  # always generate exactly n_predict tokens
        }
        return self.request("/completion", payload)

    def embeddings(self, model: str, texts: list[str]) -> dict[str, Any]:
        return self.request("/v1/embeddings", {"model": model, "input": texts, "encoding_format": "float"})


def build_prompt(server: Server, model: str, n_tokens: int, salt: str = "") -> tuple[str, int]:
    """A prompt of about n_tokens tokens (measured with the server's tokenizer), unique per salt."""
    text = salt + PARAGRAPH * (n_tokens // 20 + 1)
    tokens = server.tokenize(model, text)
    if len(tokens) < n_tokens:
        raise SystemExit(f"Could not build a {n_tokens}-token prompt (got {len(tokens)})")
    prompt = server.detokenize(model, tokens[:n_tokens])
    return prompt, len(server.tokenize(model, prompt))


# ---------------------------------------------------------------------------
# GPU memory sampling
# ---------------------------------------------------------------------------

class GpuSampler:
    """Samples GPU memory use with nvidia-smi in a background thread; records the peak per GPU."""

    def __init__(self, interval: float = 0.5) -> None:
        self.available = shutil.which("nvidia-smi") is not None
        self.interval = interval
        self.peak: list[int] = []
        self.names: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> list[int]:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout
        return [int(x) for x in out.split()]

    def __enter__(self) -> Self:
        if not self.available:
            return self
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout
        self.names = [line.strip() for line in out.splitlines() if line.strip()]
        self.peak = self._query()

        def loop() -> None:
            while not self._stop.is_set():
                self.peak = [max(a, b) for a, b in zip(self.peak, self._query(), strict=False)]
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def timings_summary(t: dict[str, Any]) -> dict[str, Any]:
    out = {
        "prompt_tokens": t.get("prompt_n"),
        "prompt_tps": round(t.get("prompt_per_second") or 0, 1),
        "generated_tokens": t.get("predicted_n"),
        "generation_tps": round(t.get("predicted_per_second") or 0, 1),
    }
    if t.get("draft_n"):
        out["draft_acceptance"] = round(100 * t.get("draft_n_accepted", 0) / t["draft_n"], 1)
    return out


def test_generation(server: Server, model: str, n_predict: int, repeat: int) -> dict[str, Any]:
    prompt, _ = build_prompt(server, model, 64)
    runs = [timings_summary(server.completion(model, prompt, n_predict)["timings"]) for _ in range(repeat)]
    best = max(runs, key=lambda r: r["generation_tps"])
    return {**best, "runs": len(runs), "generation_tps_all": [r["generation_tps"] for r in runs]}


def test_prompt_processing(server: Server, model: str, prompt_tokens: int, repeat: int) -> dict[str, Any]:
    prompt, n = build_prompt(server, model, prompt_tokens)
    runs = [timings_summary(server.completion(model, prompt, 16)["timings"]) for _ in range(repeat)]
    best = max(runs, key=lambda r: r["prompt_tps"])
    return {**best, "prompt_tokens": n, "runs": len(runs), "prompt_tps_all": [r["prompt_tps"] for r in runs]}


def test_concurrency(
    server: Server,
    model: str,
    concurrency: int,
    prompt_tokens: int,
    n_predict: int,
    embedding: tuple[str, list[str], int] | None = None,
) -> dict[str, Any]:
    """Concurrent completions; with ``embedding`` (model, texts, tokens) embedding batches run at the same time.

    The mixed load reproduces how PaperQA2 uses the server (evidence summaries while new papers are embedded) and
    gives the peak GPU memory of both models under load.
    """
    prompts = [build_prompt(server, model, prompt_tokens, salt=f"Request {i}. ")[0] for i in range(concurrency)]
    stop = threading.Event()
    embed_stats = {"batches": 0, "tokens": 0, "seconds": 0.0}

    def embed_loop() -> None:
        assert embedding is not None
        emb_model, texts, tokens = embedding
        while not stop.is_set():
            t0 = time.perf_counter()
            server.embeddings(emb_model, texts)
            embed_stats["seconds"] += time.perf_counter() - t0
            embed_stats["batches"] += 1
            embed_stats["tokens"] += tokens

    embed_thread = threading.Thread(target=embed_loop, daemon=True) if embedding else None
    start = time.perf_counter()
    if embed_thread:
        embed_thread.start()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: server.completion(model, p, n_predict), prompts))
    wall = time.perf_counter() - start
    stop.set()
    if embed_thread:
        embed_thread.join()
    t = [r["timings"] for r in results]
    generated = sum(x["predicted_n"] for x in t)
    prompt_total = sum(x["prompt_n"] for x in t)
    out = {
        "concurrency": concurrency,
        "prompt_tokens_each": prompt_tokens,
        "generated_tokens_each": n_predict,
        "wall_s": round(wall, 1),
        "aggregate_generation_tps": round(generated / wall, 1),
        "aggregate_prompt_tps": round(prompt_total / wall, 1),
        "per_request_generation_tps": [round(x["predicted_per_second"], 1) for x in t],
        "per_request_prompt_tps": [round(x["prompt_per_second"], 1) for x in t],
    }
    if embedding:
        out["concurrent_embedding"] = {
            "batches": embed_stats["batches"],
            "tokens_per_s": round(embed_stats["tokens"] / embed_stats["seconds"], 1) if embed_stats["seconds"] else None,
        }
    return out


def test_embeddings(server: Server, model: str, n_texts: int, tokens_each: int) -> dict[str, Any]:
    texts = [build_prompt(server, model, tokens_each, salt=f"Chunk {i}. ")[0] for i in range(n_texts)]
    total_tokens = sum(len(server.tokenize(model, t)) for t in texts)
    start = time.perf_counter()
    single = server.embeddings(model, texts[:1])
    single_s = time.perf_counter() - start
    start = time.perf_counter()
    batch = server.embeddings(model, texts)
    wall = time.perf_counter() - start
    return {
        "texts": n_texts,
        "tokens_each": tokens_each,
        "dimension": len(batch["data"][0]["embedding"]),
        "single_text_s": round(single_s, 2),
        "batch_wall_s": round(wall, 1),
        "tokens_per_s": round(total_tokens / wall, 1),
        "reported_prompt_tokens": (batch.get("usage") or {}).get("prompt_tokens") or (single.get("usage") or {}).get("prompt_tokens"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_env_key(path: Path, name: str = "LLAMA_API_KEY") -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("'\"")
    return None


def pick_models(models: list[dict[str, Any]], chat: str | None, embedding: str | None) -> tuple[str | None, str | None]:
    """Choose the chat and embedding model ids: explicit arguments, else the loaded (or first) ones by name."""
    def is_embedding(m: dict[str, Any]) -> bool:
        return "embed" in m["id"].lower() or "--embeddings" in (m.get("status", {}).get("args") or [])

    def rank(m: dict[str, Any]) -> int:
        return {"loaded": 0, "sleeping": 1}.get(m.get("status", {}).get("value"), 2)

    ordered = sorted(models, key=rank)
    if chat is None:
        chat = next((m["id"] for m in ordered if not is_embedding(m)), None)
    if embedding is None:
        embedding = next((m["id"] for m in ordered if is_embedding(m) and rank(m) < 2), None)
    return chat, embedding


def server_args(models: list[dict[str, Any]], model: str) -> list[str] | None:
    """The effective llama-server arguments of a router model (records ctx-size, parallel, etc. with the result)."""
    for m in models:
        if m["id"] == model:
            args = m.get("status", {}).get("args")
            return args[1:] if args else None  # drop the binary path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9931", help="llama-server base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="chat model id (default: first loaded non-embedding model)")
    parser.add_argument("--embedding-model", help="embedding model id (default: first loaded model with 'embed' in its id)")
    parser.add_argument("--no-embedding", action="store_true", help="skip the embedding test")
    parser.add_argument("--n-predict", type=int, default=256, help="tokens to generate per request (default %(default)s)")
    parser.add_argument("--prompt-tokens", type=int, default=4096, help="prompt length for the prompt-processing test")
    parser.add_argument("--concurrency", type=int, nargs="*", help="concurrent request counts (default: 1 and the slot count)")
    parser.add_argument("--concurrent-prompt-tokens", type=int, default=1024, help="prompt length per concurrent request")
    parser.add_argument("--embed-texts", type=int, default=16, help="texts in the embedding batch (default %(default)s)")
    parser.add_argument("--embed-tokens", type=int, default=1024, help="tokens per embedded text (default %(default)s)")
    parser.add_argument("--mixed", action="store_true",
                        help="run embedding batches concurrently with the highest concurrency test (PaperQA2-like load)")
    parser.add_argument("--repeat", type=int, default=3, help="repetitions of the single-request tests; the best is reported")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout per request in seconds")
    parser.add_argument("--label", default="", help="free-text label stored with the result, e.g. what was changed")
    parser.add_argument("--output", type=Path, help="JSON lines file to append to (default results/<hostname>.jsonl)")
    args = parser.parse_args()

    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get("LLAMA_API_KEY")
    server = Server(args.url, api_key, args.timeout)
    models = server.models()
    chat_model, embedding_model = pick_models(models, args.model, args.embedding_model)
    if chat_model is None:
        raise SystemExit("No chat model found; pass --model")
    if args.no_embedding:
        embedding_model = None

    props = server.props(chat_model)
    slots = int(props.get("total_slots") or 1)
    concurrency = args.concurrency if args.concurrency is not None else sorted({1, slots})
    print(f"server {server.url}  build {props.get('build_info')}  chat model {chat_model} ({slots} slots)"
          f"  embedding model {embedding_model or '-'}", file=sys.stderr)

    result: dict[str, Any] = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "client_platform": platform.platform(),
        "label": args.label,
        "url": server.url,
        "build_info": props.get("build_info"),
        "chat_model": chat_model,
        "chat_model_path": props.get("model_path"),
        "chat_server_args": server_args(models, chat_model),
        "embedding_model": embedding_model,
        "embedding_server_args": server_args(models, embedding_model) if embedding_model else None,
    }

    with GpuSampler() as gpu:
        print("warm-up...", file=sys.stderr)
        server.completion(chat_model, "Hello", 8)
        if embedding_model:
            server.embeddings(embedding_model, ["hello"])
        print(f"generation: {args.n_predict} tokens x{args.repeat}...", file=sys.stderr)
        result["generation"] = test_generation(server, chat_model, args.n_predict, args.repeat)
        print(f"prompt processing: {args.prompt_tokens} tokens x{args.repeat}...", file=sys.stderr)
        result["prompt_processing"] = test_prompt_processing(server, chat_model, args.prompt_tokens, args.repeat)
        result["concurrency"] = []
        mixed_embedding = None
        if args.mixed and embedding_model:
            texts = [build_prompt(server, embedding_model, args.embed_tokens, salt=f"Chunk {i}. ")[0] for i in range(args.embed_texts)]
            mixed_embedding = (embedding_model, texts, sum(len(server.tokenize(embedding_model, t)) for t in texts))
        for c in concurrency:
            mixed = mixed_embedding if c == max(concurrency) else None
            print(f"concurrency {c}: {args.concurrent_prompt_tokens}-token prompts, {args.n_predict} tokens each"
                  f"{' + concurrent embeddings' if mixed else ''}...", file=sys.stderr)
            result["concurrency"].append(
                test_concurrency(server, chat_model, c, args.concurrent_prompt_tokens, args.n_predict, mixed)
            )
        if embedding_model:
            print(f"embeddings: {args.embed_texts} texts x {args.embed_tokens} tokens...", file=sys.stderr)
            result["embeddings"] = test_embeddings(server, embedding_model, args.embed_texts, args.embed_tokens)
    if gpu.available:
        result["gpu"] = {"names": gpu.names, "peak_memory_used_mib": gpu.peak}

    output = args.output or Path(__file__).resolve().parent / "results" / f"{socket.gethostname().split('.')[0]}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    g, p = result["generation"], result["prompt_processing"]
    print(f"\n## {result['date']}  {result['host']}  {args.label}".rstrip())
    print(f"Chat model: `{chat_model}` (build {result['build_info']}, {slots} slots)")
    print("\n| Test | Result |\n|---|---|")
    acc = f", draft acceptance {g['draft_acceptance']} %" if "draft_acceptance" in g else ""
    print(f"| Generation, 1 request, {g['generated_tokens']} tokens | **{g['generation_tps']} tokens/s**{acc} |")
    print(f"| Prompt processing, {p['prompt_tokens']} tokens | **{p['prompt_tps']} tokens/s** |")
    for c in result["concurrency"]:
        mixed = c.get("concurrent_embedding")
        extra = f" + embeddings at {mixed['tokens_per_s']} tokens/s" if mixed else ""
        print(f"| {c['concurrency']} concurrent x ({c['prompt_tokens_each']} prompt + {c['generated_tokens_each']} generated)"
              f"{' with concurrent embeddings' if mixed else ''} "
              f"| {c['aggregate_generation_tps']} tokens/s generated in total, "
              f"{min(c['per_request_generation_tps'])}-{max(c['per_request_generation_tps'])} per request, "
              f"{c['wall_s']} s{extra} |")
    if embedding_model:
        e = result["embeddings"]
        print(f"| Embeddings `{embedding_model}`, {e['texts']} x {e['tokens_each']} tokens | **{e['tokens_per_s']} tokens/s**, "
              f"{e['dimension']} dims, single text {e['single_text_s']} s |")
    if gpu.available:
        print(f"| Peak GPU memory | {', '.join(f'{n}: {m} MiB' for n, m in zip(gpu.names, gpu.peak, strict=False))} |")
    print(f"\nAppended to `{output}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
