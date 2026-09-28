#!/usr/bin/env python3
"""Measure how settings that should not change the output (KV cache quantization etc.) change it: KL divergence.

Runs llama.cpp's ``llama-perplexity`` in a new container from ``--image``: first a base run that saves the full
logits of every scored token (``--kl-divergence-base``), then one run per ``--variant`` that compares its logits with
the saved ones (``--kl-divergence``). The result per variant is llama.cpp's statistics: the mean, median and high
percentiles of the KL divergence (KLD) between the two token distributions, how often the most likely token is the
same ("same top p"), the RMS change of the probability of the correct token, and the perplexity ratio.

To tell real degradation from noise, include a variant that should be lossless but changes the order of the
floating-point operations, such as a different ``-ub``: its KLD is the noise floor.

The text is scored with a context of ``--ctx`` tokens; llama-perplexity scores only the second half of each chunk,
so every scored token has at least ctx/2 tokens of context, which is where KV cache quantization matters most. The
default text is real Python source (from the client's standard library) followed by English prose (the bash man
page), as a mix of agentic-coding-like content; ``--text`` uses your own file instead.

The base logits take ``n_vocab * 2`` bytes per scored token (Gemma 4: 262144 tokens, 512 KiB per token, ~17 GB per
65536-token chunk). They are written to ``--work-dir``, which must be on a disk with enough space (not a RAM-backed
/tmp), and deleted at the end unless ``--keep-base``.

The GPU memory must be free: stop the llama-server container first. Only the Python standard library is needed.

Example (the Radeon VII, V cache q8_0 vs f16 with ub 256 as the noise floor):
    docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml stop
    ./llama_cpp_kld.py --image mixa3607/llama.cpp-gfx906:v0.5.0-rocm-7.14 \\
        --model /root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/.../gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \\
        --ctx 65536 --work-dir ~/.cache/llama-kld --label "Radeon VII" --base-args "-ngl 999 -fa on -ub 512" \\
        --variant "noise: ub 256=-ub 256" --variant "V q8_0=-ctv q8_0" --variant "K+V q8_0=-ctk q8_0 -ctv q8_0"
    docker compose -f ../llama-cpp-radeon-vii/docker-compose.yml start

Results are printed as a Markdown table and appended to ``results/<hostname>-kld.jsonl``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from llama_cpp_bench_http import host_name

# Statistics printed by llama-perplexity --kl-divergence, "Name : value ± error" (the error is optional).
STAT_RE = re.compile(r"^(Mean PPL\(Q\)|Mean PPL\(base\)|Mean ln\(PPL\(Q\)/PPL\(base\)\)|Mean PPL\(Q\)/PPL\(base\)"
                     r"|Mean\s+KLD|Maximum KLD|99\.9%\s+KLD|99\.0%\s+KLD|90\.0%\s+KLD|Median\s+KLD|RMS Δp|Same top p)"
                     r"\s*:\s*([-0-9.e+]+)(?:\s*±\s*([-0-9.e+]+))?")


def build_text(path: Path, code_chars: int, prose_chars: int) -> None:
    """Python source from the client's standard library, then the bash man page as prose."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    code = []
    for package in ("asyncio", "email", "http", "logging", "json", "concurrent", "importlib", "xml", "unittest"):
        code += [f"\n### {f.relative_to(stdlib)}\n" + f.read_text(encoding="utf-8") for f in sorted((stdlib / package).rglob("*.py"))]
    code_text = "".join(code)[:code_chars]
    man = subprocess.run("man -P cat bash | col -bx", shell=True, capture_output=True, text=True, check=False,
                         env={"MANWIDTH": "100", "PATH": "/usr/bin:/bin"}).stdout
    if len(man) < 100000:
        raise SystemExit("Could not render the bash man page for the prose part; pass --text")
    path.write_text(code_text + "\n\n" + man[:prose_chars], encoding="utf-8")


def perplexity(args: argparse.Namespace, extra: list[str]) -> str:
    """Run llama-perplexity in a new container; returns its combined output."""
    cmd = ["docker", "run", "--rm", "--device", "/dev/dri", "--group-add", "video",
           "-v", f"{args.hf_cache}:/root/.cache/huggingface/hub:ro", "-v", f"{args.work_dir}:/work"]
    if Path("/dev/kfd").exists():  # ROCm
        cmd += ["--device", "/dev/kfd"]
    cmd += [*args.docker_arg, "--entrypoint", "/app/llama-perplexity", args.image,
            "-m", args.model, "-f", "/work/" + args.text_name, "-c", str(args.ctx), *extra]
    if args.chunks:
        cmd += ["--chunks", str(args.chunks)]
    print("Running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(f"llama-perplexity failed ({proc.returncode}):\n{tail}")
    return proc.stdout + proc.stderr


def parse_stats(output: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for line in output.splitlines():
        if m := STAT_RE.match(line.strip()):
            key = re.sub(r"\s+", " ", m.group(1))
            stats[key] = float(m.group(2))
            if m.group(3):
                stats[key + " ±"] = float(m.group(3))
    if n := re.search(r"perplexity: calculating perplexity over (\d+) chunks", output):
        stats["chunks"] = int(n.group(1))
    if n := re.search(r"n_ctx=(\d+)", output):
        stats["n_ctx"] = int(n.group(1))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="llama.cpp image with /app/llama-perplexity")
    parser.add_argument("--model", required=True, help="GGUF path inside the container")
    parser.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub",
                        help="Hugging Face cache mounted at /root/.cache/huggingface/hub (default %(default)s)")
    parser.add_argument("--work-dir", type=Path, required=True, help="directory for the text and the base logits")
    parser.add_argument("--text", type=Path, help="text file to score (default: Python source + bash man page)")
    parser.add_argument("--ctx", type=int, default=65536, help="context per chunk (default %(default)s)")
    parser.add_argument("--chunks", type=int, default=2, help="number of chunks to score (default %(default)s)")
    parser.add_argument("--base-args", default="-ngl 999 -fa on", help="llama-perplexity arguments for all runs")
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL=ARGS",
                        help="a variant: label and the arguments added to --base-args (repeatable)")
    parser.add_argument("--docker-arg", action="append", default=[], help="extra argument for docker run (repeatable)")
    parser.add_argument("--keep-base", action="store_true", help="keep the base logits file")
    parser.add_argument("--reuse-base", action="store_true", help="use an existing base logits file in --work-dir")
    parser.add_argument("--label", default="", help="label stored with the results")
    parser.add_argument("--output", type=Path, help="JSON lines file (default results/<hostname>-kld.jsonl)")
    parser.add_argument("--hash-hostname", action="store_true",
                        help="store the hostname as its SHA-256 hash in the result and the default output file name")
    args = parser.parse_args()

    variants = [v.split("=", 1) for v in args.variant]
    if not variants or any(len(v) != 2 for v in variants):
        raise SystemExit("Give at least one --variant LABEL=ARGS")
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.text:
        args.text_name = "text.txt"
        shutil.copyfile(args.text, args.work_dir / args.text_name)
    else:
        args.text_name = "default-text.txt"
        # ~3.5 characters per token for code and ~4.2 for prose; generous so that the chunks are full.
        half = args.ctx * args.chunks // 2
        build_text(args.work_dir / args.text_name, code_chars=int(half * 4.2), prose_chars=int(half * 5.0))
    base = shlex.split(args.base_args)
    base_file = args.work_dir / "base.kld"

    output_file = args.output or Path(__file__).resolve().parent / "results" / f"{host_name(args.hash_hostname).split('.')[0]}-kld.jsonl"
    common = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": host_name(args.hash_hostname),
        "label": args.label,
        "image": args.image,
        "model": args.model,
        "text": str(args.text) if args.text else "Python stdlib source + bash man page",
        "ctx": args.ctx,
        "chunks": args.chunks,
        "base_args": args.base_args,
    }
    try:
        if not (args.reuse_base and base_file.exists()):
            print("base run (saving logits)...", file=sys.stderr)
            out = perplexity(args, [*base, "--kl-divergence-base", "/work/base.kld"])
            if m := re.search(r"Final estimate: PPL = ([0-9.]+)", out):
                print(f"base PPL {m.group(1)}", file=sys.stderr)
        results = []
        for label, extra in variants:
            print(f"variant {label}...", file=sys.stderr)
            out = perplexity(args, [*base, *shlex.split(extra), "--kl-divergence-base", "/work/base.kld", "--kl-divergence"])
            results.append({**common, "variant": label, "variant_args": extra, "stats": parse_stats(out)})
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
    finally:
        if not args.keep_base:
            base_file.unlink(missing_ok=True)

    print(f"\n## {common['date']}  {common['host']}  {args.label}".rstrip())
    print(f"`{args.image}`, ctx {args.ctx} x {args.chunks} chunks, base `{args.base_args}`\n")
    print("| Variant | Mean KLD | Median KLD | 99 % KLD | 99.9 % KLD | Max KLD | Same top token | RMS Δp | PPL ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r["stats"]
        print(f"| {r['variant']} | {s.get('Mean KLD', float('nan')):.6f} ± {s.get('Mean KLD ±', float('nan')):.6f} "
              f"| {s.get('Median KLD', float('nan')):.6f} | {s.get('99.0% KLD', float('nan')):.5f} "
              f"| {s.get('99.9% KLD', float('nan')):.4f} | {s.get('Maximum KLD', float('nan')):.3f} "
              f"| {s.get('Same top p', float('nan')):.3f} % | {s.get('RMS Δp', float('nan')):.3f} % "
              f"| {s.get('Mean PPL(Q)/PPL(base)', float('nan')):.5f} |")
    print(f"\nAppended to `{output_file}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
