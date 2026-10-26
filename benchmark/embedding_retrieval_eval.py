#!/usr/bin/env python3
r"""Compare the retrieval quality of embedding models on a document corpus, e.g. the chunks of a PaperQA2 index.

PaperQA2 embeds the chunks of the papers and the question, and gives the ``evidence_k`` (default 10) chunks that are
most similar to the question to the LLM for summarizing. If the right chunk is not among them, the LLM never sees it.
This script measures how often that happens for a given embedding model, on the actual corpus:

1. ``questions``: a random sample of the chunks is given to an LLM, which writes one specific question that the chunk
   answers (or skips the chunk, e.g. a reference list or a title page). This is done once; the questions are stored
   in a JSON lines file and reused for every embedding model.
2. ``eval``: all chunks and all questions are embedded with the model of an embedding server, and the chunks are
   ranked by cosine similarity to each question. The result is the recall at 1, 5 and 10 (how often the chunk that
   the question was written from is among the top k) and the mean reciprocal rank (MRR). The questions are embedded
   as they are, without an instruction prefix, as PaperQA2 does.

The input chunks are a JSON lines file with a ``text`` field per line. For a PaperQA2 index, they can be exported from
the pickled documents in ``<index>/docs/`` (see ``../vllm-rtx-a4000/README.md``).

Examples:
    ./embedding_retrieval_eval.py questions --chunks chunks.jsonl --out questions.jsonl --n 200 \
        --url http://localhost:9932 --env-file ../llama-cpp-rtx-a4000/llama-cpp.env
    ./embedding_retrieval_eval.py eval --chunks chunks.jsonl --questions questions.jsonl \
        --url http://localhost:9934 --env-file ../llama-cpp-rtx-a4000/llama-cpp.env --label "Qwen3-Embedding-8B FP8"

The results are appended to ``results/<hostname>-embedding.jsonl``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
import urllib.request

from llama_cpp_bench_http import host_name, read_env_key
import numpy as np

QUESTION_PROMPT = """Below is an excerpt of a physics paper. Write one specific question that this excerpt answers, \
as a researcher would ask it when searching a library of papers on the topic. The question must be answerable from \
this excerpt alone, must not refer to "the excerpt", "the paper" or "the authors", and should name the concepts \
involved instead of quoting long phrases. If the excerpt is only a reference list, a title page, author \
affiliations, acknowledgements or otherwise has no substantive content, answer only SKIP.

Excerpt:
{text}

Answer with the question only."""


def post(url: str, path: str, payload: dict[str, Any], api_key: str | None, timeout: float = 600) -> dict[str, Any]:
    """POST a JSON request and return the JSON response."""
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_model(url: str, api_key: str | None) -> str:
    """The id of the first model of a server."""
    req = urllib.request.Request(url.rstrip("/") + "/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["data"][0]["id"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_questions(args: argparse.Namespace, api_key: str | None) -> None:
    """Write one question per sampled chunk with an LLM."""
    chunks = load_jsonl(args.chunks)
    model = args.model or get_model(args.url, api_key)
    indices = random.Random(args.seed).sample(range(len(chunks)), min(args.n, len(chunks)))

    def ask(i: int) -> dict[str, Any] | None:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": QUESTION_PROMPT.format(text=chunks[i]["text"])}],
            "max_tokens": 200,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        text = post(args.url, "/v1/chat/completions", payload, api_key)["choices"][0]["message"]["content"] or ""
        text = text.strip().strip('"')
        if not text or "SKIP" in text or "?" not in text:
            return None
        return {"chunk": i, "question": text}

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = [r for r in pool.map(ask, indices) if r]
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(results)} questions from {len(indices)} chunks ({model}) -> {args.out}")


def embed(url: str, model: str, texts: list[str], api_key: str | None, batch: int, concurrency: int) -> np.ndarray:
    """Embed texts in batches and return the L2-normalized embeddings as rows."""
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]

    def run(b: list[str]) -> list[list[float]]:
        data = post(url, "/v1/embeddings", {"model": model, "input": b, "encoding_format": "float"}, api_key)
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    with ThreadPoolExecutor(concurrency) as pool:
        vectors = [v for vs in pool.map(run, batches) for v in vs]
    out = np.asarray(vectors, dtype=np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def evaluate(args: argparse.Namespace, api_key: str | None) -> None:
    """Embed the chunks and the questions and measure how well the source chunks are retrieved."""
    chunks = load_jsonl(args.chunks)
    questions = load_jsonl(args.questions)
    model = args.model or get_model(args.url, api_key)
    start = time.perf_counter()
    chunk_vecs = embed(args.url, model, [c["text"] for c in chunks], api_key, args.batch, args.concurrency)
    chunk_s = time.perf_counter() - start
    query_vecs = embed(args.url, model, [q["question"] for q in questions], api_key, args.batch, args.concurrency)
    if args.dims:
        # Matryoshka truncation, re-normalized
        chunk_vecs = chunk_vecs[:, : args.dims] / np.linalg.norm(chunk_vecs[:, : args.dims], axis=1, keepdims=True)
        query_vecs = query_vecs[:, : args.dims] / np.linalg.norm(query_vecs[:, : args.dims], axis=1, keepdims=True)
    sims = query_vecs @ chunk_vecs.T
    target = np.array([q["chunk"] for q in questions])
    # Rank of the source chunk: the number of chunks that are more similar than it (0 = first)
    ranks = (sims > sims[np.arange(len(questions)), target][:, None]).sum(axis=1)
    result = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": host_name(name=args.hostname),
        "label": args.label,
        "url": args.url,
        "model": model,
        "dimension": int(chunk_vecs.shape[1]),
        "chunks": len(chunks),
        "questions": len(questions),
        "recall_at_1": round(float((ranks < 1).mean()), 3),
        "recall_at_5": round(float((ranks < 5).mean()), 3),
        "recall_at_10": round(float((ranks < 10).mean()), 3),
        "mrr": round(float((1 / (ranks + 1)).mean()), 3),
        "chunk_embedding_s": round(chunk_s, 1),
    }
    output = args.output or Path(__file__).resolve().parent / "results" / f"{result['host']}-embedding.jsonl"
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"| {args.label or model} | {result['recall_at_1']} | {result['recall_at_5']} | {result['recall_at_10']} "
          f"| {result['mrr']} | {result['dimension']} | {chunk_s:.0f} s |")


def main() -> None:
    """Parse the arguments and run the command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("questions", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--chunks", type=Path, required=True, help="JSON lines file with a 'text' field per chunk")
        p.add_argument("--url", required=True, help="server base URL (LLM for questions, embedding model for eval)")
        p.add_argument("--api-key", help="API key, if the server was started with one")
        p.add_argument("--env-file", type=Path, help="read the key from LLAMA_API_KEY=... in this file")
        p.add_argument("--model", help="model id (default: the first model of the server)")
        p.add_argument("--concurrency", type=int, default=4, help="concurrent requests (default %(default)s)")
    q = sub.choices["questions"]
    q.add_argument("--out", type=Path, required=True, help="JSON lines file for the questions")
    q.add_argument("--n", type=int, default=200, help="chunks to sample (default %(default)s)")
    q.add_argument("--seed", type=int, default=1, help="random seed of the sample (default %(default)s)")
    e = sub.choices["eval"]
    e.add_argument("--questions", type=Path, required=True, help="JSON lines file written by 'questions'")
    e.add_argument("--batch", type=int, default=8, help="texts per embedding request (default %(default)s)")
    e.add_argument("--dims", type=int, help="truncate the embeddings to this dimension (Matryoshka models)")
    e.add_argument("--label", default="", help="free-text label stored with the result")
    e.add_argument("--hostname", help="hostname for the results file (default from ../hostname.py)")
    e.add_argument("--output", type=Path, help="JSON lines file to append to (default results/<host>-embedding.jsonl)")
    args = parser.parse_args()
    api_key = args.api_key or (read_env_key(args.env_file) if args.env_file else None)
    if args.command == "questions":
        make_questions(args, api_key)
    else:
        evaluate(args, api_key)


if __name__ == "__main__":
    sys.exit(main())
