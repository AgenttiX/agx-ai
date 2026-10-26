#!/usr/bin/env python3
r"""Generation speed of several coding agents at once, each with its own long context already in the prefix cache.

Each agent has a distinct code context of ``--depth`` tokens (real Python source, as in the depth test of
``llama_cpp_bench_http.py``) and a question about it. The contexts are first sent one at a time with one generated
token, which puts them into the prefix cache, as for an agent's next turn. Then all agents ask at once, and the script
reports the generation speed per agent, the generated tokens per second in total and each agent's time to the first
token (long if its context had to be computed again). Only for vLLM (``cache_salt`` keeps the agents apart). Needs
only the Python standard library.

Example:
    ./vllm_concurrent_agents.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \\
        --depth 16000 --agents 1 2 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any

from llama_cpp_bench_http import (
    DEPTH_QUESTIONS,
    VllmServer,
    build_code_context,
    connect,
    read_env_key,
    timings_summary,
)


def run_agents(server: VllmServer, model: str, depth: int, n: int, max_tokens: int) -> dict[str, Any]:
    """Warm the prefix cache with ``n`` contexts, then send the ``n`` requests at once."""
    prompts = [
        build_code_context(server, model, depth, rotate=5 * i + 3 * n)
        + "\n\n" + DEPTH_QUESTIONS[i % len(DEPTH_QUESTIONS)]
        for i in range(n)
    ]
    salts = [os.urandom(8).hex() for _ in range(n)]

    def payload(i: int, tokens: int) -> dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": prompts[i]}], "max_tokens": tokens,
                "seed": 42, "cache_salt": salts[i]}

    for i in range(n):
        server.stream("/v1/chat/completions", payload(i, 1))
    results: list[dict[str, Any]] = [{} for _ in range(n)]

    def run(i: int) -> None:
        results[i] = timings_summary(server.stream("/v1/chat/completions", payload(i, max_tokens))["timings"])

    start = time.perf_counter()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    gen = [r["generation_tps"] for r in results]
    return {
        "agents": n,
        "depth": depth,
        "generation_tps_mean": round(statistics.mean(gen), 1),
        "generation_tps_min": min(gen),
        "generation_tps_max": max(gen),
        "total_tps": round(sum(r["generated_tokens"] or 0 for r in results) / wall, 1),
        "ttft_s": [round(r["prompt_tokens"] / r["prompt_tps"], 1) if r["prompt_tps"] else None for r in results],
    }


def main() -> int:
    """Run the test for each number of agents and print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9933", help="vLLM base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="model id (default: the first one of /v1/models)")
    parser.add_argument("--depth", type=int, default=16000, help="context tokens per agent (default %(default)s)")
    parser.add_argument("--agents", type=int, nargs="+", default=[1, 2, 4],
                        help="numbers of agents at once (default %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=512, help="tokens per answer (default %(default)s)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout per request in seconds")
    args = parser.parse_args()

    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get(
        "LLAMA_API_KEY")
    server = connect(args.url, api_key, args.timeout)
    if not isinstance(server, VllmServer):
        raise SystemExit("This test needs a vLLM server")
    model = args.model or server.models()[0]["id"]
    print("| Agents x context | Generation per agent | In total | Time to the first token |\n|---|---|---|---|")
    for n in args.agents:
        r = run_agents(server, model, args.depth, n, args.max_tokens)
        print(f"| {n} x {args.depth} | {r['generation_tps_mean']} t/s ({r['generation_tps_min']}-"
              f"{r['generation_tps_max']}) | {r['total_tps']} t/s | {', '.join(f'{t} s' for t in r['ttft_s'])} |",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
