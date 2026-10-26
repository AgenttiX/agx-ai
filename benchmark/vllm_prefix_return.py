#!/usr/bin/env python3
r"""Time to the first token of a conversation after other long prompts have evicted it from the GPU.

The case of a coding agent that is resumed after other agents have worked on the same server: with vLLM's CPU KV
offloading (``kv-offloading-size``), the evicted prefix is loaded from RAM instead of being computed again.

1. A conversation of ``--tokens`` tokens of code and a question (time to the first token: a cold prompt),
2. the same request again (a hit in the GPU's prefix cache),
3. ``--evict`` distinct prompts of ``--evict-tokens`` tokens each, which push the conversation out of the GPU,
4. the conversation again: a hit in the CPU KV cache, or a cold prompt without one.

The answers are greedy and compared with the first one, which checks that the restored KV cache is correct. Only
for vLLM (``cache_salt`` keeps the requests of each step apart). Needs only the Python standard library.

Example:
    ./vllm_prefix_return.py --url http://localhost:9933 --env-file ../llama-cpp/llama-cpp.env \\
        --tokens 25000 --evict 2 --evict-tokens 100000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

from llama_cpp_bench_http import VllmServer, build_code_context, connect, read_env_key


def ask(server: VllmServer, model: str, content: str, salt: str, max_tokens: int) -> tuple[float, int, str]:
    """A greedy chat request without thinking; returns the time to the first token, the prompt tokens and the text."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": salt,
    }
    start = time.perf_counter()
    out = server.stream("/v1/chat/completions", payload)
    ttft = out["timings"]["prompt_ms"] / 1000
    print(f"  {time.perf_counter() - start:6.1f} s in total", file=sys.stderr)
    return ttft, out["timings"]["prompt_n"], out["text"]


def main() -> int:
    """Run the test and print the times as a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9933", help="vLLM base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="model id (default: the first one of /v1/models)")
    parser.add_argument("--tokens", type=int, default=25000, help="conversation length (default %(default)s)")
    parser.add_argument("--evict", type=int, default=2, help="evicting prompts (default %(default)s)")
    parser.add_argument("--evict-tokens", type=int, default=100000,
                        help="tokens per evicting prompt (default %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=64, help="tokens per answer (default %(default)s)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout per request in seconds")
    args = parser.parse_args()

    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get(
        "LLAMA_API_KEY")
    server = connect(args.url, api_key, args.timeout)
    if not isinstance(server, VllmServer):
        raise SystemExit("This test needs a vLLM server")
    model = args.model or server.models()[0]["id"]
    question = "\n\nWhich module above is the largest, and what does it do? Answer in two sentences."
    conversation = build_code_context(server, model, args.tokens) + question
    salt = os.urandom(8).hex()

    print("cold conversation...", file=sys.stderr)
    cold, n, reference = ask(server, model, conversation, salt, args.max_tokens)
    print("the same again (GPU prefix cache)...", file=sys.stderr)
    gpu, _, text_gpu = ask(server, model, conversation, salt, args.max_tokens)
    for i in range(args.evict):
        print(f"evicting prompt {i + 1}/{args.evict}...", file=sys.stderr)
        ask(server, model, build_code_context(server, model, args.evict_tokens, rotate=11 * (i + 1)) + question,
            os.urandom(8).hex(), 8)
    print("the conversation again...", file=sys.stderr)
    back, _, text_back = ask(server, model, conversation, salt, args.max_tokens)

    print(f"\n| Conversation of {n} tokens | Time to the first token | Same answer as cold |\n|---|---|---|")
    print(f"| cold | {cold:.2f} s | |")
    print(f"| again, from the GPU prefix cache | {gpu:.2f} s | {text_gpu == reference} |")
    print(f"| after {args.evict} x {args.evict_tokens} other tokens | {back:.2f} s | {text_back == reference} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
