r"""Benchmark of a chat server over its OpenAI-compatible API: time to the first token, generation, images, answers.

Works with any server with a streamed ``/v1/chat/completions`` endpoint: llama.cpp, vLLM and the Gemma 4 NPU server
of ``openvino-l14-npu-e4b``, for which it was written. The requests are streamed and timed on the client, so the
numbers include the HTTP overhead. The NPU server reuses the KV cache of the previous conversation, so each prompt
starts with a random nonce. The generation speed uses the completion tokens reported in ``usage`` (else one token
per streamed chunk). The answers to short questions with an unambiguous answer are checked as well; use
``--no-thinking`` for thinking models, as the answers have only 32 tokens.

Examples:
    uv run -m agx_ai.benchmark.chat_stream --url http://localhost:9934 --json results.json
    uv run -m agx_ai.benchmark.chat_stream --url http://localhost:9932 --env-file llama-cpp/llama-cpp.env \\
        --no-thinking
"""

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import time
from typing import Any
import urllib.error
import urllib.request
import uuid
import zlib

from agx_ai.benchmark.bench_http import pick_models, read_env_key

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the river flows quietly past the old mill, "
    "and the farmers gather their harvest before the autumn rains arrive in the valley. "
)

# Short questions with an unambiguous answer, which is checked as a substring of the reply.
QUESTIONS = [
    ("What is 17 * 23? Answer with the number only.", "391"),
    ("What is the capital of Australia? Answer with the city name only.", "Canberra"),
    ("If a train travels 120 km in 1.5 hours, what is its average speed in km/h? Answer with the number only.", "80"),
    ("Which planet is the largest in the Solar System? Answer with the name only.", "Jupiter"),
    ("How many days are there in a leap year? Answer with the number only.", "366"),
    ("What is the chemical symbol of gold? Answer with the symbol only.", "Au"),
    ("Alice is older than Bob, and Bob is older than Carol. Who is the youngest? Answer with the name only.", "Carol"),
    ("What is the square root of 144? Answer with the number only.", "12"),
]


class ChatClient:
    """A minimal client for streamed chat completions with greedy decoding."""

    def __init__(self, url: str, api_key: str | None = None, model: str | None = None, thinking: bool = True,
                 timeout: float = 600) -> None:
        self.url: str = url.rstrip("/")
        self.api_key: str | None = api_key
        self.model: str | None = model
        self.thinking: bool = thinking
        self.timeout: float = timeout

    def _request(self, path: str, body: dict[str, Any] | None = None) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.url + path, data, headers)

    def default_model(self) -> str | None:
        """The first loaded chat model of ``/v1/models``, or None if the server does not list them."""
        try:
            with urllib.request.urlopen(self._request("/v1/models"), timeout=self.timeout) as resp:
                models = json.loads(resp.read()).get("data", [])
        except (urllib.error.URLError, OSError, json.JSONDecodeError, AttributeError):
            return None
        return pick_models(models, None, None)[0]

    def stream(self, content: str | list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        """Send a streamed chat request with greedy decoding and time it."""
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream_options": {"include_usage": True},
        }
        if self.model:
            body["model"] = self.model
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        t_start = time.perf_counter()
        t_first = None
        chunks = 0
        text = []
        usage = None
        with urllib.request.urlopen(self._request("/v1/chat/completions", body), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                if "error" in data:
                    raise RuntimeError(data["error"])
                usage = data.get("usage") or usage
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        t_first = t_first or time.perf_counter()
                        chunks += 1
                    if delta.get("content"):
                        text.append(delta["content"])
        t_end = time.perf_counter()
        if t_first is None:
            raise RuntimeError("No tokens were generated")
        tokens = (usage or {}).get("completion_tokens") or chunks
        tokens_per_s = (tokens - 1) / (t_end - t_first) if tokens > 1 and t_end > t_first else float("nan")
        return {"ttft_s": t_first - t_start, "tokens_per_s": tokens_per_s, "usage": usage, "text": "".join(text)}


def nonce() -> str:
    """Return a random prefix that prevents the reuse of the KV cache of the previous request."""
    return f"[{uuid.uuid4().hex[:8]}] "


def gradient_png(width: int = 640, height: int = 480) -> str:
    """Return a synthetic gradient PNG image as a data URL."""
    rows = (
        b"\0" + b"".join(bytes((x * 255 // width, y * 255 // height, 128)) for x in range(width))
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def summary(name: str, runs: list[dict[str, Any]]) -> None:
    """Print the timings of the runs of one test."""
    ttft = ", ".join(f"{r['ttft_s']:.2f}" for r in runs)
    speed = ", ".join(f"{r['tokens_per_s']:.2f}" for r in runs)
    prompt = runs[0]["usage"]["prompt_tokens"] if runs[0]["usage"] else "?"
    print(f"{name}: {prompt} prompt tokens, time to first token {ttft} s, generation {speed} t/s")


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9934", help="server base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="model id (default: the first loaded chat model of /v1/models, if listed)")
    parser.add_argument("--no-thinking", action="store_true", help="disable thinking with the chat template")
    parser.add_argument("--repeat", type=int, default=3, help="requests per speed test (default %(default)s)")
    parser.add_argument("--no-image", action="store_true", help="skip the image test (for models without vision)")
    parser.add_argument("--json", type=Path, help="write the full results, including the replies, to this file")
    args = parser.parse_args(argv)
    api_key = (
        args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get("LLAMA_API_KEY")
    )
    client = ChatClient(args.url, api_key, thinking=not args.no_thinking)
    client.model = args.model or client.default_model()

    client.stream(nonce() + "Hello!", 8)  # Warm-up
    short_runs = max(args.repeat, 1)
    long_runs = max(args.repeat - 1, 1)
    results = {
        "short": [
            client.stream(nonce() + "Write a short paragraph about the history of Finland.", 128)
            for _ in range(short_runs)
        ],
        "long": [
            client.stream(nonce() + PARAGRAPH * 22 + "\nSummarize the text above in one sentence.", 64)
            for _ in range(long_runs)
        ],
    }
    if not args.no_image:
        image = gradient_png()
        results["image"] = [
            client.stream(
                [
                    {"type": "text", "text": nonce() + "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
                64,
            )
            for _ in range(long_runs)
        ]
    for name, runs in results.items():
        summary(name, runs)

    answers: list[dict[str, Any]] = []
    for question, expected in QUESTIONS:
        reply = client.stream(question, 32)["text"].strip()
        answers.append({"question": question, "reply": reply, "correct": expected.lower() in reply.lower()})
    print(f"Correct answers: {sum(bool(a['correct']) for a in answers)}/{len(answers)}")
    for answer in answers:
        print(f"  {'OK ' if answer['correct'] else 'BAD'} {answer['reply']!r}")

    if args.json:
        with args.json.open("w", encoding="utf-8") as file:
            json.dump({**results, "answers": answers}, file, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
