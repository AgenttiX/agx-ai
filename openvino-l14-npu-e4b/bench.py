#!/usr/bin/env python3
"""Benchmark of the Gemma 4 NPU server: time to the first token, generation speed, image input and answers.

The server has no timings in its responses, so the requests are streamed and timed on the client.
It reuses the KV cache of the previous conversation, so each prompt starts with a random nonce.
Usage: ./bench.py [--url http://localhost:9934] [--json results.json]
"""

import argparse
import base64
import json
from pathlib import Path
import struct
import time
import urllib.request
import uuid
import zlib

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


def stream(url: str, content: str | list[dict], max_tokens: int) -> dict:
    """Send a streamed chat request with greedy decoding and time it."""
    body = {
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t_start = time.perf_counter()
    t_first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            usage = data.get("usage") or usage
            delta = data["choices"][0]["delta"].get("content")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                text.append(delta)
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("No tokens were generated")
    # The server sends one chunk per token.
    tokens_per_s = (len(text) - 1) / (t_end - t_first) if len(text) > 1 else float("nan")
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


def summary(name: str, runs: list[dict]) -> None:
    """Print the timings of the runs of one test."""
    ttft = ", ".join(f"{r['ttft_s']:.2f}" for r in runs)
    speed = ", ".join(f"{r['tokens_per_s']:.2f}" for r in runs)
    prompt = runs[0]["usage"]["prompt_tokens"] if runs[0]["usage"] else "?"
    print(f"{name}: {prompt} prompt tokens, time to first token {ttft} s, generation {speed} t/s")


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9934")
    parser.add_argument("--json", help="Write the full results, including the replies, to this file")
    args = parser.parse_args()
    url = args.url

    stream(url, nonce() + "Hello!", 8)  # Warm-up
    results = {
        "short": [
            stream(url, nonce() + "Write a short paragraph about the history of Finland.", 128) for _ in range(3)
        ],
        "long": [
            stream(url, nonce() + PARAGRAPH * 22 + "\nSummarize the text above in one sentence.", 64)
            for _ in range(2)
        ],
    }
    image = gradient_png()
    results["image"] = [
        stream(
            url,
            [
                {"type": "text", "text": nonce() + "Describe this image briefly."},
                {"type": "image_url", "image_url": {"url": image}},
            ],
            64,
        )
        for _ in range(2)
    ]
    for name, runs in results.items():
        summary(name, runs)

    answers = []
    for question, expected in QUESTIONS:
        reply = stream(url, question, 32)["text"].strip()
        answers.append({"question": question, "reply": reply, "correct": expected.lower() in reply.lower()})
    print(f"Correct answers: {sum(a['correct'] for a in answers)}/{len(answers)}")
    for answer in answers:
        print(f"  {'OK ' if answer['correct'] else 'BAD'} {answer['reply']!r}")

    if args.json:
        with Path(args.json).open("w", encoding="utf-8") as file:
            json.dump({**results, "answers": answers}, file, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
