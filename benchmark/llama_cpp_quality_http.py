#!/usr/bin/env python3
"""Check whether a serving setting (e.g. KV cache quantization) degrades the output quality of a running llama-server.

Two tests, both with greedy decoding (temperature 0) so that runs are repeatable:

- Retrieval at long contexts (``retrieval``): real Python source of about ``--depth`` tokens with facts hidden in it
  as code comments ("needles"), followed by questions about them; exact-match scoring. There are two kinds of
  questions: a single lookup (the rollout token of one of many similarly named services) and a two-step lookup
  (service -> database -> region, with the two facts far apart). This is where a lossy KV cache would show up first:
  the answer depends on attending precisely to a few tokens far back in the context. Thinking is disabled so that
  the answer is only the value. The haystack is the same for all questions of a depth and is reused from the
  server's prompt cache, so only the first question per depth pays for the long prompt.
- Greedy agreement (``agreement``): the chat prompts and the long-context code questions of
  ``llama_cpp_bench_http.py`` (thinking as the server's defaults), generated greedily. With ``--reference``, the
  outputs are compared with those of an earlier run: how many are identical and how long the common prefix is.
  Greedy outputs diverge after any change in the floating-point operations, so compare with a noise floor: a run
  with a setting that is lossless in principle, e.g. another ``ubatch-size``.

The retrieval answers also carry the probabilities of the top ``--logprobs`` tokens at every answer position. With
``--reference``, these give the most sensitive comparison: the KL divergence between the two runs' token
distributions at each position where the answers still agree, how often the most likely token differs, and how much
the probability of the reference's token changed. This is measured in the model's normal chat regime at long
contexts, i.e. exactly where a lossy KV cache matters. Tokens accepted from speculative decoding (MTP, draft models)
have no probabilities, so disable speculative decoding on the test server for this (it does not change the main
model's distributions); the coverage is reported.

The needles and questions are generated from ``--seed``, so different runs with the same arguments are comparable.
llama-perplexity's KL divergence (``llama_cpp_kld.py``) is the more precise method, but it is not usable for all
models: with Gemma 4 26B-A4B on raw text, even a different ubatch size flips ~30 % of the top tokens.

Examples:
    ./llama_cpp_quality_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \\
        --label "f16 KV (reference)"
    ./llama_cpp_quality_http.py --url http://localhost:9933 --env-file ../llama-cpp-radeon-vii/llama-cpp.env \\
        --label "V q8_0" --reference "f16 KV (reference)"

Each run is appended to two files: ``results/full/<hostname>-quality.jsonl`` with everything, including the
questions, the generated outputs and the token probabilities (several MB per run, not in Git), and
``results/<hostname>-quality.jsonl`` with the summaries, the comparison and the answers only. ``--reference`` is the
label of an earlier run in the full file (the latest with that label is used).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

from llama_cpp_bench_http import (
    CHAT_PROMPTS, DEPTH_QUESTIONS, Server, VllmServer, build_code_context, host_name, is_vllm, pick_models, read_env_key,
    server_args,
)

NAMES = ["orion", "vega", "lyra", "altair", "rigel", "sirius", "deneb", "mira", "castor", "pollux", "antares", "spica"]
REGIONS = ["eu-north-1", "eu-north-2", "eu-west-1", "eu-west-3", "eu-central-1", "us-east-1", "us-east-2", "us-west-1",
           "us-west-2", "ap-south-1", "ap-northeast-1", "ap-southeast-2", "sa-east-1", "ca-central-1", "me-south-1"]


def build_code(server: Server, model: str, n_tokens: int) -> str:
    """About n_tokens tokens of real Python source (more packages than the depth test, for contexts beyond 70k)."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = [f for pkg in ("asyncio", "email", "http", "logging", "json", "concurrent", "importlib", "xml")
             for f in sorted((stdlib / pkg).rglob("*.py"))]
    text = "".join(f"\n### {f.relative_to(stdlib)}\n" + f.read_text(encoding="utf-8") for f in files)
    tokens = server.tokenize(model, text)
    if len(tokens) < n_tokens:
        raise SystemExit(f"Could not build a {n_tokens}-token code context (got {len(tokens)})")
    return server.detokenize(model, tokens[:n_tokens])


def make_needles(rng: random.Random, n_lookup: int, n_pairs: int
                 ) -> tuple[list[str], list[dict[str, str]], list[tuple[str, str]]]:
    """Fact lines to hide and the questions about them; similar names on purpose, to require precise attention."""
    names = rng.sample([f"{n}-{i}" for n in NAMES for i in range(1, 100)], n_lookup + n_pairs)
    databases = rng.sample([f"db-{n}-{i}" for n in NAMES for i in range(1, 100)], n_pairs)
    lines, questions = [], []
    for name in names[:n_lookup]:
        token = str(rng.randrange(100000, 1000000))
        lines.append(f"# Deployment note: the rollout token of service {name} is {token}.")
        questions.append({"kind": "lookup", "expected": token,
                          "question": f"The code above contains deployment notes as comments. What is the rollout token of "
                                      f"service {name}? Reply with only the number."})
    pair_lines = []
    for name, db in zip(names[n_lookup:], databases, strict=True):
        region = rng.choice(REGIONS)
        pair_lines.append((f"# Deployment note: service {name} reads its configuration from database {db}.",
                           f"# Infrastructure note: database {db} is hosted in region {region}."))
        questions.append({"kind": "two-step", "expected": region,
                          "question": f"The code above contains deployment and infrastructure notes as comments. In which "
                                      f"region is the database hosted that service {name} reads its configuration from? "
                                      f"Reply with only the region name."})
    return lines, questions, pair_lines


def build_haystack(server: Server, model: str, depth: int, rng: random.Random, n_lookup: int, n_pairs: int
                   ) -> tuple[str, list[dict[str, str]]]:
    """About ``depth`` tokens of code with the needle lines inserted at random line boundaries."""
    lines, questions, pair_lines = make_needles(rng, n_lookup, n_pairs)
    code = build_code(server, model, depth).split("\n")
    # Positions in the code: lookups anywhere; the two facts of a pair far apart (first half, second half).
    inserts: list[tuple[int, str]] = [(rng.randrange(1, len(code)), line) for line in lines]
    for first, second in pair_lines:
        a, b = sorted(rng.sample(range(1, len(code)), 2))
        if b - a < len(code) // 4:  # keep the two facts at least a quarter of the context apart
            a, b = rng.randrange(1, len(code) // 2), rng.randrange(len(code) // 2 + len(code) // 4, len(code))
        inserts += [(a, first), (b, second)]
    for pos, line in sorted(inserts, key=lambda x: x[0], reverse=True):
        code.insert(pos, line)
    rng.shuffle(questions)
    return "\n".join(code), questions


def chat(server: Server, model: str, content: str, max_tokens: int, thinking: bool | None, cache: bool,
         logprobs: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}],
                               "max_tokens": max_tokens, "temperature": 0.0, "seed": 42, "cache_prompt": cache}
    if isinstance(server, VllmServer):
        del payload["cache_prompt"]
        if not cache:  # vLLM has no cache_prompt; a unique salt keeps the request from hitting the prefix cache
            payload["cache_salt"] = os.urandom(8).hex()
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    if logprobs:
        payload |= {"logprobs": True, "top_logprobs": logprobs}
    r = server.request("/v1/chat/completions", payload)
    choice = r["choices"][0]
    msg = choice["message"]
    timings = r.get("timings") or {"prompt_n": (r.get("usage") or {}).get("prompt_tokens", 0)}  # vLLM: no timings
    out = {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
           "timings": timings}
    if logprobs:
        # Per generated token: the token and the top alternatives' log-probabilities (empty for drafted tokens).
        # Several special tokens render as "", so only the first (most likely) entry of a text is kept.
        out["tokens"] = []
        for t in (choice.get("logprobs") or {}).get("content") or []:
            top: dict[str, float] = {}
            for x in t.get("top_logprobs") or []:
                top.setdefault(x["token"], round(x["logprob"], 5))
            out["tokens"].append({"t": t["token"], "top": top})
    return out


def test_retrieval(server: Server, model: str, depths: list[int], seed: int, n_lookup: int, n_pairs: int,
                   logprobs: int) -> list[dict[str, Any]]:
    out = []
    for depth in depths:
        rng = random.Random(f"{seed}-{depth}")
        haystack, questions = build_haystack(server, model, depth, rng, n_lookup, n_pairs)
        print(f"retrieval: {depth}-token context, {len(questions)} questions...", file=sys.stderr)
        answers = []
        t0 = time.perf_counter()
        for q in questions:
            r = chat(server, model, haystack + "\n\n" + q["question"], 24, thinking=False, cache=True, logprobs=logprobs)
            answer = r["content"].strip()
            answers.append({**q, "answer": answer, "correct": q["expected"] in re.findall(r"[A-Za-z0-9-]+", answer),
                            "prompt_tokens": r["timings"].get("prompt_n", 0) + r["timings"].get("cache_n", 0),
                            **({"tokens": r["tokens"]} if logprobs else {})})
        out.append({"depth": depth, "prompt_tokens": max(a["prompt_tokens"] for a in answers),
                    "seconds": round(time.perf_counter() - t0, 1), "answers": answers,
                    **{f"{kind}_correct": sum(a["correct"] for a in answers if a["kind"] == kind) for kind in ("lookup", "two-step")},
                    **{f"{kind}_total": sum(a["kind"] == kind for a in answers) for kind in ("lookup", "two-step")}})
    return out


def test_agreement(server: Server, model: str, depths: list[int], max_tokens: int) -> list[dict[str, Any]]:
    items = [{"id": f"chat-{i}", "content": p} for i, p in enumerate(CHAT_PROMPTS)]
    for d in depths:
        items += [{"id": f"depth-{d}-{i}", "content": build_code_context(server, model, d, rotate=7 * i) + "\n\n" + q}
                  for i, q in enumerate(DEPTH_QUESTIONS)]
    out = []
    for it in items:
        print(f"agreement: {it['id']}...", file=sys.stderr)
        # No prompt cache: a cache hit changes how the prompt is batched, and with it the greedy output.
        r = chat(server, model, it["content"], max_tokens, thinking=None, cache=False)
        out.append({"id": it["id"], "reasoning": r["reasoning"], "content": r["content"]})
    return out


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def token_kld(ref_top: dict[str, float], top: dict[str, float]) -> float:
    """KL(ref || this) over the reference's top tokens (renormalized); a token missing from this run's top list
    gets the lowest log-probability of that list, so the value is a slight underestimate for large divergences."""
    floor = min(top.values())
    z = sum(math.exp(v) for v in ref_top.values())
    return max(sum(math.exp(lp) / z * (lp - top.get(t, floor)) for t, lp in ref_top.items()), 0.0)


def compare_logprobs(result: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    """Token-distribution statistics over the retrieval answers, at the positions where both answers still agree."""
    ref_answers = {(d["depth"], a["question"]): a for d in ref.get("retrieval", []) for a in d["answers"]}
    klds, dlp, flips, first_klds, covered, total = [], [], 0, [], 0, 0
    per_depth: dict[int, list[float]] = {}
    for d in result.get("retrieval", []):
        for a in d["answers"]:
            b = ref_answers.get((d["depth"], a["question"]))
            if not b or "tokens" not in a or "tokens" not in b:
                continue
            for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"], strict=False)):
                if x["t"] == "" and y["t"] == "":
                    break  # end of the answer; special tokens render as "" and cannot be told apart
                total += 1
                if not x["top"] or not y["top"]:
                    continue
                covered += 1
                k = token_kld(y["top"], x["top"])
                klds.append(k)
                per_depth.setdefault(d["depth"], []).append(k)
                if i == 0:
                    first_klds.append(k)
                if max(x["top"], key=x["top"].get) != max(y["top"], key=y["top"].get):
                    flips += 1
                if y["t"] in x["top"] and y["t"] in y["top"]:
                    dlp.append(abs(x["top"][y["t"]] - y["top"][y["t"]]))
                if x["t"] != y["t"]:
                    break  # the answers diverge; later positions have different contexts
    if not klds:
        return None
    klds_sorted = sorted(klds)
    return {
        "positions": len(klds),
        "coverage": f"{covered}/{total}",
        "mean_kld": round(sum(klds) / len(klds), 6),
        "median_kld": round(klds_sorted[len(klds) // 2], 6),
        "p99_kld": round(klds_sorted[min(int(len(klds) * 0.99), len(klds) - 1)], 5),
        "max_kld": round(klds_sorted[-1], 4),
        "first_token_mean_kld": round(sum(first_klds) / len(first_klds), 6) if first_klds else None,
        "top_token_flips": flips,
        "mean_abs_delta_logprob": round(sum(dlp) / len(dlp), 5) if dlp else None,
        "mean_kld_per_depth": {str(k): round(sum(v) / len(v), 6) for k, v in sorted(per_depth.items())},
    }


def compare(result: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    cmp: dict[str, Any] = {"reference_label": ref["label"], "reference_date": ref["date"]}
    ra = {(d["depth"], a["question"]): a["answer"] for d in ref.get("retrieval", []) for a in d["answers"]}
    same = [a["answer"] == ra[(d["depth"], a["question"])] for d in result.get("retrieval", []) for a in d["answers"]
            if (d["depth"], a["question"]) in ra]
    if same:
        cmp["retrieval_same_answer"] = f"{sum(same)}/{len(same)}"
    if lp := compare_logprobs(result, ref):
        cmp["retrieval_logprobs"] = lp
    rg = {g["id"]: g["reasoning"] + "\n" + g["content"] for g in ref.get("agreement", [])}
    rows = []
    for g in result.get("agreement", []):
        if g["id"] in rg:
            a, b = g["reasoning"] + "\n" + g["content"], rg[g["id"]]
            rows.append({"id": g["id"], "identical": a == b, "common_prefix_chars": common_prefix(a, b), "length": len(b)})
    if rows:
        cmp["agreement_identical"] = f"{sum(r['identical'] for r in rows)}/{len(rows)}"
        cmp["agreement_mean_common_prefix_fraction"] = round(sum(min(r["common_prefix_chars"] / max(r["length"], 1), 1) for r in rows) / len(rows), 3)
        cmp["agreement"] = rows
    return cmp


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """The result without the bulky parts (questions, token probabilities and generated texts), for the file in Git."""
    out = dict(result)
    if "retrieval" in out:
        out["retrieval"] = [
            {**d, "answers": [{k: a[k] for k in ("kind", "expected", "answer", "correct")} for a in d["answers"]]}
            for d in out["retrieval"]
        ]
    if "agreement" in out:
        out["agreement"] = [{"id": g["id"], "length": len(g["reasoning"] + "\n" + g["content"])} for g in out["agreement"]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9931", help="llama-server base URL (default %(default)s)")
    parser.add_argument("--api-key", help="API key, if the server was started with one")
    parser.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
    parser.add_argument("--model", help="chat model id (default: first loaded non-embedding model)")
    parser.add_argument("--tests", nargs="+", choices=["retrieval", "agreement"], default=["retrieval", "agreement"])
    parser.add_argument("--depth", type=int, nargs="+", default=[8000, 32000, 60000, 75000],
                        help="context lengths in tokens for the retrieval test (default %(default)s)")
    parser.add_argument("--lookups", type=int, default=32, help="single-lookup questions per depth (default %(default)s)")
    parser.add_argument("--pairs", type=int, default=12, help="two-step questions per depth (default %(default)s)")
    parser.add_argument("--agreement-depth", type=int, nargs="+", default=[16000, 60000],
                        help="context lengths for the long-context prompts of the agreement test (default %(default)s)")
    parser.add_argument("--agreement-n-predict", type=int, default=512, help="tokens per agreement output (default %(default)s)")
    parser.add_argument("--logprobs", type=int, default=20,
                        help="top log-probabilities recorded per retrieval answer token, 0 = none (default %(default)s)")
    parser.add_argument("--seed", type=int, default=1, help="seed for the needles and questions (default %(default)s)")
    parser.add_argument("--reference", help="label of an earlier run in the output file to compare with")
    parser.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout per request in seconds")
    parser.add_argument("--label", required=True, help="label of this run (used by --reference)")
    parser.add_argument("--output", type=Path,
                        help="JSON lines file for the compact results (default results/<hostname>-quality.jsonl); "
                             "the full results go to full/ in the same directory")
    parser.add_argument("--hash-hostname", action="store_true",
                        help="store the hostname as its SHA-256 hash in the result and the default output file name")
    parser.add_argument("--hostname", default=os.environ.get("LLAMA_BENCH_HOSTNAME"),
                        help="hostname to store instead of this computer's (default: $LLAMA_BENCH_HOSTNAME, if set)")
    args = parser.parse_args()

    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None) or os.environ.get("LLAMA_API_KEY")
    server = (VllmServer if is_vllm(args.url, args.timeout) else Server)(args.url, api_key, args.timeout)
    models = server.models()
    model, _ = pick_models(models, args.model, None)
    if model is None:
        raise SystemExit("No chat model found; pass --model")
    props = server.props(model)
    output = args.output or Path(__file__).resolve().parent / "results" / f"{host_name(args.hash_hostname, args.hostname).split('.')[0]}-quality.jsonl"
    full_output = output.parent / "full" / output.name
    ref = None
    if args.reference:
        runs = [json.loads(line) for line in full_output.read_text(encoding="utf-8").splitlines()] if full_output.exists() else []
        ref = next((r for r in reversed(runs) if r["label"] == args.reference), None)
        if ref is None:
            raise SystemExit(f"No run labelled {args.reference!r} in {full_output}")

    result: dict[str, Any] = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": host_name(args.hash_hostname, args.hostname),
        "label": args.label,
        "url": server.url,
        "build_info": props.get("build_info"),
        "model": model,
        "model_path": props.get("model_path"),
        "server_args": server_args(models, model),
        "seed": args.seed,
    }
    if "retrieval" in args.tests:
        result["retrieval"] = test_retrieval(server, model, args.depth, args.seed, args.lookups, args.pairs, args.logprobs)
    if "agreement" in args.tests:
        result["agreement"] = test_agreement(server, model, args.agreement_depth, args.agreement_n_predict)
    if ref:
        result["comparison"] = compare(result, ref)
    full_output.parent.mkdir(parents=True, exist_ok=True)
    with full_output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(compact(result), ensure_ascii=False) + "\n")

    print(f"\n## {result['date']}  {result['host']}  {args.label}")
    print(f"Model: `{model}` (build {result['build_info']})\n")
    if result.get("retrieval"):
        print("| Context | Single lookup | Two-step lookup | Time |\n|---|---|---|---|")
        for d in result["retrieval"]:
            print(f"| {d['prompt_tokens']} tokens | {d['lookup_correct']}/{d['lookup_total']} "
                  f"| {d['two-step_correct']}/{d['two-step_total']} | {d['seconds']} s |")
        total = sum(d["lookup_correct"] + d["two-step_correct"] for d in result["retrieval"])
        n = sum(d["lookup_total"] + d["two-step_total"] for d in result["retrieval"])
        print(f"| **All** | | **{total}/{n}** | |")
    if c := result.get("comparison"):
        print(f"\nCompared with `{c['reference_label']}` ({c['reference_date']}): retrieval answers identical "
              f"{c.get('retrieval_same_answer', '-')}, greedy outputs identical {c.get('agreement_identical', '-')}, "
              f"mean common prefix {c.get('agreement_mean_common_prefix_fraction', '-')} of the reference length.")
        if lp := c.get("retrieval_logprobs"):
            print(f"Answer token distributions ({lp['positions']} positions, coverage {lp['coverage']}): "
                  f"mean KLD {lp['mean_kld']}, median {lp['median_kld']}, 99 % {lp['p99_kld']}, max {lp['max_kld']}, "
                  f"first token {lp['first_token_mean_kld']}; top-token flips {lp['top_token_flips']}; "
                  f"mean |Δ log p| of the reference token {lp['mean_abs_delta_logprob']}; "
                  f"mean KLD per context {lp['mean_kld_per_depth']}")
    print(f"\nAppended to `{output}` and `{full_output}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
