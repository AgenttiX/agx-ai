"""Send one chat request with a prompt of a given length and report the peak GPU memory of the server processes.

The prompt is Python standard library source of about --tokens tokens (the code corpus of bench_http ends at ~212k
tokens); the peak comes from nvidia-smi every 0.5 s. Standard library only.

  python peak_probe.py --url http://localhost:9935 --env-file ../../llama-cpp/llama-cpp.env --tokens 258000
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
import urllib.request


def api_key(path: str) -> str | None:
    """LLAMA_API_KEY from an env file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("LLAMA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def post(url: str, body: dict[str, Any], key: str | None, timeout: float = 3600) -> Any:
    """POST JSON and return the decoded response."""
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                   "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build_text(tokens: int) -> str:
    """About 1.1 * tokens tokens of Python source (~3.3 characters per token)."""
    parts, est = [], 0.0
    for f in sorted(Path(os.__file__).parent.rglob("*.py")):
        try:
            t = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        parts.append(f"# file: {f.name}\n{t}")
        est += len(t) / 3.3
        if est > tokens * 1.1:
            break
    return "\n".join(parts)


def gpu_memory() -> int:
    """MiB used by the compute processes on the NVIDIA GPUs."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout.split()
    return sum(int(x) for x in out if x.isdigit())


def main() -> None:
    """Send the request and print the token counts, the time, the peak memory and the answer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9935")
    ap.add_argument("--env-file")
    ap.add_argument("--tokens", type=int, default=258000)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    key = api_key(a.env_file) if a.env_file else None
    models_req = urllib.request.Request(a.url + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(models_req) as r:
        model = json.load(r)["data"][0]["id"]
    text = build_text(a.tokens)
    q = "\n\nWhich of the files above defines the most classes? Answer in one sentence."
    n = post(a.url + "/tokenize", {"model": model, "prompt": text}, key)["count"]
    text = text[: int(len(text) * (a.tokens - 100) / n)]
    peak = [0]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_memory())
            stop.wait(0.5)
    th = threading.Thread(target=poll)
    th.start()
    t0 = time.time()
    r = post(a.url + "/v1/chat/completions", {"model": model, "max_tokens": a.max_tokens, "temperature": 0,
                                              "messages": [{"role": "user", "content": text + q}],
                                              "chat_template_kwargs": {"enable_thinking": False}}, key)
    dt = time.time() - t0
    stop.set()
    th.join()
    u = r["usage"]
    print(f"prompt {u['prompt_tokens']} tokens, completion {u['completion_tokens']} tokens, {dt:.1f} s; "
          f"peak GPU memory of the compute processes {peak[0]} MiB")
    print("answer:", r["choices"][0]["message"]["content"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
