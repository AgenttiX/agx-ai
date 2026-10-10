#!/usr/bin/env python3
"""Benchmark a running llama.cpp server (llama-server) over its HTTP API.

Measures, for the serving configuration as it actually runs (router presets, slots, KV cache, speculative decoding):
- generation speed (tokens/s) for a single request,
- prompt processing speed (tokens/s) for a long prompt,
- throughput with concurrent requests (aggregate tokens/s and per-request speed),
- embedding throughput (tokens/s) for the embedding model, if one is served,
- optionally, generation speed for realistic chat prompts (``chat``) and prompt processing and generation with
  real code as a long context (``depth``); these use the chat template and the server's own sampling settings, so
  they give representative draft acceptance for speculative decoding, unlike the repeated-text prompts at
  temperature 0 of the other tests,
- peak GPU memory during the run (nvidia-smi or amd-smi, if available on this machine).

Speeds for the LLM come from the ``timings`` object that llama-server attaches to every completion (server-side
prompt and generation timings, plus draft-token acceptance when speculative decoding is on). Embedding timings are
wall-clock, with token counts from ``/tokenize``. Nothing but the Python standard library is needed.

Examples:
    ./llama_cpp_bench_http.py --env-file ../llama-cpp-big-machine/llama-cpp.env --label "Qwen3.8-27B ctx 65536"
    ./llama_cpp_bench_http.py --url http://agx-z2e:9932 --api-key ... --model gemma --no-embedding
    ./llama_cpp_bench_http.py --concurrency 1 2 4 8 --n-predict 512 --prompt-tokens 8192 --mixed
    ./llama_cpp_bench_http.py --tests chat depth --depth 6400 16000 36500   # speculative decoding, long contexts

For the raw model speed independent of the server configuration, see ``llama_cpp_bench_container.py``.

Results are printed as a Markdown table and appended as one JSON line per run to ``results/<hostname>.jsonl``,
so that runs with different models, settings or hardware can be compared afterwards.
The hostname comes from ``../hostname.py`` by default.
``--hostname NAME`` (or the environment variable ``LLAMA_BENCH_HOSTNAME``) stores NAME instead of the hostname, and
``--hash-hostname`` replaces the hostname by its SHA-256 hash, in the result and the file name.
If ``../hostname.py`` censors the hostname, one of these is required.

Note: LiteLLM on the personal agx-ai server (``../litellm/config.yaml``, ``health_check_interval: 60``) sends a health
check request to every model once a minute over the network, which adds noise to the measurements and reloads
unloaded models. Turn it off there (``background_health_checks: false``) or stop that LiteLLM while benchmarking.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any, Self
import urllib.error
import urllib.parse
import urllib.request
import zlib

# hostname.py is in the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hostname import repo_hostname

PARAGRAPH = (
    "A first-order phase transition in the early universe proceeds through the nucleation, expansion and collision "
    "of bubbles of the new phase. The expanding bubbles set the surrounding plasma in motion, and the resulting sound "
    "waves source a stochastic background of gravitational waves whose spectrum depends on the transition strength, "
    "the wall speed, the mean bubble separation and the equation of state of the plasma. "
)

# Varied, realistic requests for the chat test (code, explanation, translation, maths, scripting, essay).
CHAT_PROMPTS = [
    "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H5M' into a timedelta, "
    "with docstring and tests.",
    "Explain how a transformer's attention mechanism works to an undergraduate physics student, with an analogy.",
    "Translate into Finnish and then explain any tricky grammar: "
    "'The committee postponed its decision until the budget had been reviewed.'",
    "A train leaves at 14:05 going 80 km/h; another leaves the same station at 14:35 going 110 km/h on a parallel "
    "track. When and where does the second catch up? Show the steps.",
    "Write a bash script that finds the ten largest files under a directory, excluding .git, "
    "and prints human-readable sizes.",
    "Summarise the pros and cons of ROCm versus Vulkan for running local LLMs on older AMD GPUs, as a short essay.",
]

# Questions asked about a long code context in the depth test; each uses a differently ordered context.
DEPTH_QUESTIONS = [
    "Review this code: list the five most serious potential bugs with explanations and fixes.",
    "Write a detailed architecture overview of these modules for a new contributor.",
    "Propose a refactoring plan for the largest module above, with example code.",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class Server:
    """A llama-server HTTP client with the endpoints used by the benchmark."""

    def __init__(self, url: str, api_key: str | None, timeout: float) -> None:
        self.url: str = url.rstrip("/")
        self.api_key: str | None = api_key
        self.timeout: float = timeout

    def request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        """Send a GET (or a POST with a JSON payload) request and return the decoded JSON response."""
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
        """Return the models listed by the server."""
        return self.request("/v1/models").get("data", [])

    def props(self, model: str) -> dict[str, Any]:
        """Return the server properties of the model."""
        return self.request(f"/props?model={urllib.parse.quote(model, safe='')}")

    def tokenize(self, model: str, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self.request("/tokenize", {"model": model, "content": text})["tokens"]

    def detokenize(self, model: str, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
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

    def chat(self, model: str, content: str, max_tokens: int) -> dict[str, Any]:
        """Chat completion with the chat template and the server's sampling settings; returns the server's timings."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "seed": 42,
            "cache_prompt": False,
        }
        return self.request("/v1/chat/completions", payload)

    def chat_image(self, model: str, png: bytes, text: str, max_tokens: int) -> dict[str, Any]:
        """Chat completion with one PNG image (for models with a vision encoder, mmproj)."""
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": text},
            ]}],
            "max_tokens": max_tokens,
            "seed": 42,
            "cache_prompt": False,
        }
        return self.request("/v1/chat/completions", payload)

    def embeddings(self, model: str, texts: list[str]) -> dict[str, Any]:
        """Embed a batch of texts."""
        return self.request("/v1/embeddings", {"model": model, "input": texts, "encoding_format": "float"})


class VllmServer(Server):
    """The same interface for a vLLM server, which has no ``timings``, ``/props`` or ``cache_prompt``.

    Requests are streamed, and the timings are measured on the client: prompt processing is the prompt length over the
    time to the first streamed token, and generation is the remaining tokens over the time from the first to the
    last one (both include the HTTP overhead, which is negligible on localhost). Every request gets its own
    ``cache_salt``, so that the prefix cache is not hit, like ``cache_prompt: false`` on llama-server. The
    draft acceptance of speculative decoding comes from the difference of the counters in ``/metrics`` before and
    after the request, so it is only exact for one request at a time.
    """

    def text(self, path: str) -> str:
        """Fetch a plain-text (non-JSON) response."""
        req = urllib.request.Request(self.url + path)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode()

    def props(self, model: str) -> dict[str, Any]:
        """Return llama-server-like properties assembled from the vLLM version and model list."""
        info = next((m for m in self.models() if m["id"] == model), {})
        return {
            "build_info": "vLLM " + json.loads(self.text("/version")).get("version", "?"),
            "model_path": info.get("root"),
            "max_model_len": info.get("max_model_len"),
            "total_slots": 1,
        }

    def tokenize(self, model: str, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self.request("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]

    def detokenize(self, model: str, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.request("/detokenize", {"model": model, "tokens": tokens})["prompt"]

    def spec_counters(self) -> tuple[float, float]:
        """Return the cumulative speculative decoding draft and accepted token counts from ``/metrics``."""
        draft = accepted = 0.0
        for line in self.text("/metrics").splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                draft += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                accepted += float(line.rsplit(" ", 1)[1])
        return draft, accepted

    def stream(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a streamed request and return client-measured ``timings`` and the ``usage``."""
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True},
                   "cache_salt": os.urandom(8).hex()}
        draft0, accepted0 = self.spec_counters()
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        start = time.perf_counter()
        first = last = None
        usage: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if (choice.get("text") or delta.get("content")
                                or delta.get("reasoning_content") or delta.get("reasoning")):
                            last = time.perf_counter()
                            first = first or last
        except urllib.error.HTTPError as e:
            raise SystemExit(f"HTTP {e.code} for {path}: {e.read().decode(errors='replace')[:300]}") from e
        prompt_n = usage.get("prompt_tokens", 0)
        predicted_n = usage.get("completion_tokens", 0)
        first = first or time.perf_counter()
        last = last or first
        draft1, accepted1 = self.spec_counters()
        timings = {
            "prompt_n": prompt_n,
            "prompt_ms": 1000 * (first - start),
            "prompt_per_second": prompt_n / (first - start),
            "predicted_n": predicted_n,
            "predicted_per_second": (predicted_n - 1) / (last - first) if predicted_n > 1 and last > first else 0.0,
        }
        if draft1 > draft0:
            timings["draft_n"] = draft1 - draft0
            timings["draft_n_accepted"] = accepted1 - accepted0
        return {"timings": timings, "usage": usage}

    def completion(self, model: str, prompt: str, n_predict: int) -> dict[str, Any]:
        """Raw completion of exactly ``n_predict`` tokens; returns the client-measured timings."""
        payload = {"model": model, "prompt": prompt, "max_tokens": n_predict, "temperature": 0.0, "ignore_eos": True}
        return self.stream("/v1/completions", payload)

    def chat(self, model: str, content: str, max_tokens: int) -> dict[str, Any]:
        """Chat completion with the chat template; returns the client-measured timings."""
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                   "seed": 42}
        return self.stream("/v1/chat/completions", payload)

    def chat_image(self, model: str, png: bytes, text: str, max_tokens: int) -> dict[str, Any]:
        """Chat completion with one PNG image; returns the client-measured timings."""
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": text},
            ]}],
            "max_tokens": max_tokens,
            "seed": 42,
        }
        return self.stream("/v1/chat/completions", payload)


def is_vllm(url: str, timeout: float) -> bool:
    """VLLM answers ``/version`` with its version; llama-server does not have that endpoint."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=timeout) as resp:
            return "version" in json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return False


def build_prompt(server: Server, model: str, n_tokens: int, salt: str = "") -> tuple[str, int]:
    """A prompt of about n_tokens tokens (measured with the server's tokenizer), unique per salt."""
    text = salt + PARAGRAPH * (n_tokens // 20 + 1)
    tokens = server.tokenize(model, text)
    if len(tokens) < n_tokens:
        raise SystemExit(f"Could not build a {n_tokens}-token prompt (got {len(tokens)})")
    prompt = server.detokenize(model, tokens[:n_tokens])
    return prompt, len(server.tokenize(model, prompt))


def build_code_context(server: Server, model: str, n_tokens: int, rotate: int = 0) -> str:
    """About n_tokens tokens of real Python source (the asyncio and email packages of the client's Python)."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted((stdlib / "asyncio").glob("*.py")) + sorted((stdlib / "email").glob("*.py"))
    files = files[rotate:] + files[:rotate]
    text = "".join(f"\n### {f}\n" + f.read_text(encoding="utf-8") for f in files)
    tokens = server.tokenize(model, text)
    if len(tokens) < n_tokens:
        raise SystemExit(f"Could not build a {n_tokens}-token code context (got {len(tokens)})")
    return server.detokenize(model, tokens[:n_tokens])


# ---------------------------------------------------------------------------
# GPU memory sampling
# ---------------------------------------------------------------------------

class GpuSampler:
    """Samples GPU memory use with nvidia-smi and amd-smi in a background thread; records the peak per GPU.

    GPUs of both vendors are listed if both tools are installed. amd-smi takes about a second per query, so its
    peaks are coarser. With nvidia-smi, the peak of the memory used by llama-server processes alone is recorded too
    (``process_peak``), which excludes other programs on the GPU, such as a desktop session.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.tools: list[str] = [t for t in ("nvidia-smi", "amd-smi") if shutil.which(t)]
        self.available: bool = bool(self.tools)
        self.interval: float = interval
        self.peak: list[int] = []
        self.process_peak: int | None = None
        self.names: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _run(cmd: list[str]) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout

    def _query_tool(self, tool: str) -> list[int]:
        if tool == "amd-smi":
            try:
                metrics = json.loads(self._run(["amd-smi", "metric", "-m", "--json"]))
                return [int(g["mem_usage"]["used_vram"]["value"]) for g in metrics]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return []
        out = self._run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return [int(x) for x in out.split()]

    def _names_tool(self, tool: str) -> list[str]:
        if tool == "amd-smi":
            try:
                gpus = json.loads(self._run(["amd-smi", "static", "-a", "-v", "--json"]))
                return [f"{g['asic']['market_name']}, {g['vram']['size']['value']} MiB" for g in gpus]
            except (json.JSONDecodeError, KeyError, TypeError):
                return []
        out = self._run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        return [line.strip() for line in out.splitlines() if line.strip()]

    def _query(self) -> list[int]:
        return [m for tool in self.tools for m in self._query_tool(tool)]

    def _query_processes(self) -> int | None:
        """MiB used by llama-server processes on all NVIDIA GPUs (None without nvidia-smi or if it fails)."""
        if "nvidia-smi" not in self.tools:
            return None
        out = self._run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"]
        )
        try:
            return sum(int(line.rsplit(",", 1)[1]) for line in out.splitlines() if "llama" in line)
        except (IndexError, ValueError):
            return None

    def __enter__(self) -> Self:
        if not self.available:
            return self
        for tool in self.tools:
            n = len(self._query_tool(tool))
            names = self._names_tool(tool)
            self.names += names if len(names) == n else [f"{tool} GPU {i}" for i in range(n)]
        self.peak = self._query()
        self.process_peak = self._query_processes()

        def loop() -> None:
            while not self._stop.is_set():
                now = self._query()
                if len(now) == len(self.peak):  # skip samples where a tool failed
                    self.peak = [max(a, b) for a, b in zip(self.peak, now, strict=True)]
                proc = self._query_processes()
                if proc is not None:
                    self.process_peak = max(self.process_peak or 0, proc)
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
    """Reduce the server timings to rounded throughput figures and the draft acceptance rate."""
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
    """Measure generation speed with a short prompt; reports the best of ``repeat`` runs."""
    prompt, _ = build_prompt(server, model, 64)
    runs = [timings_summary(server.completion(model, prompt, n_predict)["timings"]) for _ in range(repeat)]
    best = max(runs, key=lambda r: r["generation_tps"])
    return {**best, "runs": len(runs), "generation_tps_all": [r["generation_tps"] for r in runs]}


def test_prompt_processing(server: Server, model: str, prompt_tokens: int, repeat: int) -> dict[str, Any]:
    """Measure prompt processing speed with ``prompt_tokens`` tokens; reports the best of ``repeat`` runs."""
    prompt, n = build_prompt(server, model, prompt_tokens)
    runs = [timings_summary(server.completion(model, prompt, 16)["timings"]) for _ in range(repeat)]
    best = max(runs, key=lambda r: r["prompt_tps"])
    return {**best, "prompt_tokens": n, "runs": len(runs), "prompt_tps_all": [r["prompt_tps"] for r in runs]}


def test_chat(server: Server, model: str, max_tokens: int, repeat: int) -> dict[str, Any]:
    """Generation speed over the chat prompts, each run ``repeat`` times; mean, median and range over all requests."""
    runs = [timings_summary(server.chat(model, p, max_tokens)["timings"]) for p in CHAT_PROMPTS * repeat]
    tps = [r["generation_tps"] for r in runs]
    out = {
        "requests": len(runs),
        "max_tokens": max_tokens,
        "generation_tps_mean": round(statistics.mean(tps), 1),
        "generation_tps_median": round(statistics.median(tps), 1),
        "generation_tps_min": min(tps),
        "generation_tps_max": max(tps),
    }
    acc = [r["draft_acceptance"] for r in runs if "draft_acceptance" in r]
    if acc:
        out["draft_acceptance_mean"] = round(statistics.mean(acc), 1)
    return out


def test_depth(server: Server, model: str, depth: int, max_tokens: int) -> dict[str, Any]:
    """A long code context (about ``depth`` tokens) and a question: prompt processing and generation at that depth."""
    runs = [
        timings_summary(server.chat(
            model, build_code_context(server, model, depth, rotate=7 * i) + "\n\n" + q, max_tokens
        )["timings"])
        for i, q in enumerate(DEPTH_QUESTIONS)
    ]
    tps = [r["generation_tps"] for r in runs]
    out = {
        "depth": depth,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "max_tokens": max_tokens,
        "prompt_tps_mean": round(statistics.mean(r["prompt_tps"] for r in runs), 1),
        "generation_tps_mean": round(statistics.mean(tps), 1),
        "generation_tps_min": min(tps),
        "generation_tps_max": max(tps),
    }
    acc = [r["draft_acceptance"] for r in runs if "draft_acceptance" in r]
    if acc:
        out["draft_acceptance_mean"] = round(statistics.mean(acc), 1)
    return out


def synthetic_png(width: int, height: int, seed: int) -> bytes:
    """A deterministic RGB test image (colour gradients and a grid of shapes), encoded as PNG with the stdlib."""
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter type 0 for each scanline
        for x in range(width):
            cell = ((x // 64) + (y // 64) + seed) % 7
            inside = (x % 64 - 32) ** 2 + (y % 64 - 32) ** 2 < (8 + 3 * cell) ** 2
            r = (x * 255 // width + 40 * cell) % 256
            g = (y * 255 // height) if not inside else 255 - 30 * cell
            b = (128 + 60 * cell) % 256 if inside else (x ^ y) & 0xFF
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"".join(rows), 6)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_image(server: Server, model: str, size: tuple[int, int], repeat: int) -> dict[str, Any]:
    """Time to process a prompt with one image (vision encoder + prompt) and the wall time of the whole request.

    Shows the cost of running the vision encoder on the CPU (``no-mmproj-offload``). A different image per request,
    so that no cache is hit.
    """
    runs = []
    for i in range(repeat):
        png = synthetic_png(*size, seed=i)
        start = time.perf_counter()
        resp = server.chat_image(model, png, "Describe this image in one sentence.", 32)
        wall = time.perf_counter() - start
        t = resp["timings"]
        runs.append({"prompt_tokens": t.get("prompt_n"), "prompt_ms": round(t.get("prompt_ms") or 0),
                     "wall_s": round(wall, 2)})
    return {
        "image_size": f"{size[0]}x{size[1]}",
        "prompt_tokens": runs[0]["prompt_tokens"],
        "prompt_ms_mean": round(statistics.mean(r["prompt_ms"] for r in runs)),
        "wall_s_mean": round(statistics.mean(r["wall_s"] for r in runs), 2),
        "runs": runs,
    }


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
    out: dict[str, Any] = {
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
            "tokens_per_s": (
                round(embed_stats["tokens"] / embed_stats["seconds"], 1) if embed_stats["seconds"] else None
            ),
        }
    return out


def test_embeddings(server: Server, model: str, n_texts: int, tokens_each: int) -> dict[str, Any]:
    """Measure embedding throughput for a batch of ``n_texts`` texts and for a single text."""
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
        "reported_prompt_tokens": (
            (batch.get("usage") or {}).get("prompt_tokens") or (single.get("usage") or {}).get("prompt_tokens")
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_env_key(path: Path, name: str = "LLAMA_API_KEY") -> str | None:
    """Read the value of ``name`` from a ``NAME=value`` env file (None if it is not set)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("'\"")
    return None


def host_name(hashed: bool = False, name: str | None = None) -> str:
    """The hostname (``name`` if given, e.g. from ``--hostname``, else from ``hostname.py``), or its SHA-256 hex digest.

    Both keep the real name of e.g. a work computer out of a public repository.
    If ``hostname.py`` censors the hostname, ``name`` or ``hashed`` is required.
    """
    if not name and not hashed:
        name = repo_hostname()
        if name is None:
            raise SystemExit(
                "The hostname of this computer is censored by hostname.py. "
                "Give another name with --hostname NAME or LLAMA_BENCH_HOSTNAME=NAME, or use --hash-hostname."
            )
    name = name or socket.gethostname()
    return hashlib.sha256(name.encode()).hexdigest() if hashed else name


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


ALL_TESTS = ["generation", "prompt", "concurrency", "embedding", "chat", "depth", "image"]
DEFAULT_TESTS = ["generation", "prompt", "concurrency", "embedding"]


def main() -> int:
    """Parse the arguments, run the selected tests and store and print the result."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9931", help="llama-server base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="chat model id (default: first loaded non-embedding model)")
    parser.add_argument("--embedding-model",
                        help="embedding model id (default: first loaded model with 'embed' in its id)")
    parser.add_argument("--no-embedding", action="store_true", help="skip the embedding test")
    parser.add_argument("--tests", nargs="+", choices=ALL_TESTS, default=DEFAULT_TESTS,
                        help=f"tests to run (default: {' '.join(DEFAULT_TESTS)})")
    parser.add_argument("--n-predict", type=int, default=256,
                        help="tokens to generate per request (default %(default)s)")
    parser.add_argument("--prompt-tokens", type=int, default=4096,
                        help="prompt length for the prompt-processing test")
    parser.add_argument("--concurrency", type=int, nargs="*",
                        help="concurrent request counts (default: 1 and the slot count)")
    parser.add_argument("--concurrent-prompt-tokens", type=int, default=1024,
                        help="prompt length per concurrent request")
    parser.add_argument("--embed-texts", type=int, default=16,
                        help="texts in the embedding batch (default %(default)s)")
    parser.add_argument("--embed-tokens", type=int, default=1024,
                        help="tokens per embedded text (default %(default)s)")
    parser.add_argument("--mixed", action="store_true",
                        help="run embedding batches concurrently with the highest concurrency test "
                             "(PaperQA2-like load)")
    parser.add_argument("--chat-n-predict", type=int, default=512,
                        help="maximum tokens per request in the chat and depth tests (default %(default)s)")
    parser.add_argument("--depth", type=int, nargs="+", default=[16000],
                        help="context lengths in tokens for the depth test (default %(default)s)")
    parser.add_argument("--image-size", default="1280x960",
                        help="width x height of the test image of the image test (default %(default)s)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="repetitions of the single-request tests (the best is reported) and of the chat prompts")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout per request in seconds")
    parser.add_argument("--label", default="", help="free-text label stored with the result, e.g. what was changed")
    parser.add_argument("--server-config", type=Path,
                        help="configuration file of the server to store with the result, as vLLM does not report its "
                             "arguments (e.g. the YAML file given to vllm serve --config)")
    parser.add_argument("--output", type=Path, help="JSON lines file to append to (default results/<hostname>.jsonl)")
    parser.add_argument("--hash-hostname", action="store_true",
                        help="store the hostname as its SHA-256 hash in the result and the default output file name")
    parser.add_argument("--hostname", default=os.environ.get("LLAMA_BENCH_HOSTNAME"),
                        help="hostname to store instead of this computer's (default: $LLAMA_BENCH_HOSTNAME, if set)")
    args = parser.parse_args()

    api_key = (
        args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get("LLAMA_API_KEY")
    )
    server = (VllmServer if is_vllm(args.url, args.timeout) else Server)(args.url, api_key, args.timeout)
    models = server.models()
    chat_model, embedding_model = pick_models(models, args.model, args.embedding_model)
    if chat_model is None:
        raise SystemExit("No chat model found; pass --model")
    if args.no_embedding or "embedding" not in args.tests:
        embedding_model = None

    props = server.props(chat_model)
    slots = int(props.get("total_slots") or 1)
    concurrency = args.concurrency if args.concurrency is not None else sorted({1, slots})
    print(f"server {server.url}  build {props.get('build_info')}  chat model {chat_model} ({slots} slots)"
          f"  embedding model {embedding_model or '-'}", file=sys.stderr)

    result: dict[str, Any] = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": host_name(args.hash_hostname, args.hostname),
        "client_platform": platform.platform(),
        "label": args.label,
        "url": server.url,
        "build_info": props.get("build_info"),
        "chat_model": chat_model,
        "chat_model_path": props.get("model_path"),
        "chat_server_args": server_args(models, chat_model),
        "chat_server_config": args.server_config.read_text(encoding="utf-8") if args.server_config else None,
        "max_model_len": props.get("max_model_len"),
        "embedding_model": embedding_model,
        "embedding_server_args": server_args(models, embedding_model) if embedding_model else None,
    }

    with GpuSampler() as gpu:
        print("warm-up...", file=sys.stderr)
        server.completion(chat_model, "Hello", 8)
        if embedding_model:
            server.embeddings(embedding_model, ["hello"])
        if "generation" in args.tests:
            print(f"generation: {args.n_predict} tokens x{args.repeat}...", file=sys.stderr)
            result["generation"] = test_generation(server, chat_model, args.n_predict, args.repeat)
        if "prompt" in args.tests:
            print(f"prompt processing: {args.prompt_tokens} tokens x{args.repeat}...", file=sys.stderr)
            result["prompt_processing"] = test_prompt_processing(server, chat_model, args.prompt_tokens, args.repeat)
        if "chat" in args.tests:
            print(f"chat: {len(CHAT_PROMPTS)} prompts x{args.repeat}, up to {args.chat_n_predict} tokens...",
                  file=sys.stderr)
            result["chat"] = test_chat(server, chat_model, args.chat_n_predict, args.repeat)
        if "depth" in args.tests:
            result["depth"] = []
            for d in args.depth:
                print(f"depth: {len(DEPTH_QUESTIONS)} x {d}-token code context, up to {args.chat_n_predict} tokens...",
                      file=sys.stderr)
                result["depth"].append(test_depth(server, chat_model, d, args.chat_n_predict))
        if "image" in args.tests:
            size = tuple(int(v) for v in args.image_size.lower().split("x"))
            print(f"image: {args.image_size} image x{args.repeat}...", file=sys.stderr)
            result["image"] = test_image(server, chat_model, size, args.repeat)  # type: ignore[arg-type]
        result["concurrency"] = []
        mixed_embedding = None
        if args.mixed and embedding_model:
            texts = [
                build_prompt(server, embedding_model, args.embed_tokens, salt=f"Chunk {i}. ")[0]
                for i in range(args.embed_texts)
            ]
            mixed_embedding = (embedding_model, texts, sum(len(server.tokenize(embedding_model, t)) for t in texts))
        for c in concurrency if "concurrency" in args.tests else []:
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
        gpu_result: dict[str, Any] = {"names": gpu.names, "peak_memory_used_mib": gpu.peak}
        if gpu.process_peak is not None:
            gpu_result["peak_llama_server_memory_mib"] = gpu.process_peak
        result["gpu"] = gpu_result

    output = args.output or Path(__file__).resolve().parent / "results" / f"{result['host'].split('.')[0]}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n## {result['date']}  {result['host']}  {args.label}".rstrip())
    print(f"Chat model: `{chat_model}` (build {result['build_info']}, {slots} slots)")
    print("\n| Test | Result |\n|---|---|")
    if g := result.get("generation"):
        acc = f", draft acceptance {g['draft_acceptance']} %" if "draft_acceptance" in g else ""
        print(f"| Generation, 1 request, {g['generated_tokens']} tokens | **{g['generation_tps']} tokens/s**{acc} |")
    if p := result.get("prompt_processing"):
        print(f"| Prompt processing, {p['prompt_tokens']} tokens | **{p['prompt_tps']} tokens/s** |")
    if ch := result.get("chat"):
        acc = f", draft acceptance {ch['draft_acceptance_mean']} %" if "draft_acceptance_mean" in ch else ""
        print(f"| Chat, {ch['requests']} requests, up to {ch['max_tokens']} tokens "
              f"| **{ch['generation_tps_mean']} tokens/s** "
              f"(median {ch['generation_tps_median']}, {ch['generation_tps_min']}-{ch['generation_tps_max']}){acc} |")
    for d in result.get("depth", []):
        acc = f", draft acceptance {d['draft_acceptance_mean']} %" if "draft_acceptance_mean" in d else ""
        print(f"| Code context of {d['prompt_tokens']} tokens, up to {d['max_tokens']} generated "
              f"| prompt **{d['prompt_tps_mean']} tokens/s**, "
              f"generation **{d['generation_tps_mean']} tokens/s** "
              f"({d['generation_tps_min']}-{d['generation_tps_max']}){acc} |")
    for c in result["concurrency"]:
        mixed = c.get("concurrent_embedding")
        extra = f" + embeddings at {mixed['tokens_per_s']} tokens/s" if mixed else ""
        print(f"| {c['concurrency']} concurrent x "
              f"({c['prompt_tokens_each']} prompt + {c['generated_tokens_each']} generated)"
              f"{' with concurrent embeddings' if mixed else ''} "
              f"| {c['aggregate_generation_tps']} tokens/s generated in total, "
              f"{min(c['per_request_generation_tps'])}-{max(c['per_request_generation_tps'])} per request, "
              f"{c['wall_s']} s{extra} |")
    if embedding_model:
        e = result["embeddings"]
        print(f"| Embeddings `{embedding_model}`, {e['texts']} x {e['tokens_each']} tokens "
              f"| **{e['tokens_per_s']} tokens/s**, "
              f"{e['dimension']} dims, single text {e['single_text_s']} s |")
    if im := result.get("image"):
        print(f"| Image {im['image_size']} ({im['prompt_tokens']} prompt tokens), {len(im['runs'])} requests | "
              f"prompt **{im['prompt_ms_mean']} ms**, whole request {im['wall_s_mean']} s |")
    if gpu.available:
        print(f"| Peak GPU memory | {', '.join(f'{n}: {m} MiB' for n, m in zip(gpu.names, gpu.peak, strict=False))} |")
        if gpu.process_peak is not None:
            print(f"| Peak GPU memory of llama-server (NVIDIA) | {gpu.process_peak} MiB |")
    print(f"\nAppended to `{output}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
