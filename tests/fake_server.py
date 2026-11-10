"""A fake LLM server that emulates the endpoints of llama-server, vLLM and OpenVINO Model Server used by the benchmarks.

The tokenizer splits the text at spaces, so that the token counts are predictable. The generated text is always the
same few tokens, and streamed responses send one chunk per token, like the real servers.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Literal, Self
import zlib

Kind = Literal["llama", "vllm", "ovms"]
CHAT_MODEL = "fake-chat-model"
EMBEDDING_MODEL = "fake-embed-model"
#: The generated tokens of every completion.
REPLY = ["The", " answer", " is", " 42."]
REASONING = ["Let", " me", " think."]
EMBEDDING_DIM = 64


def embed_text(text: str) -> list[float]:
    """A deterministic bag-of-words embedding: texts with the same words are similar."""
    vec = [0.0] * EMBEDDING_DIM
    for word in text.lower().split():
        vec[zlib.crc32(word.strip(".,?!").encode()) % EMBEDDING_DIM] += 1.0
    return vec


class FakeServer:
    """A fake server in a background thread. Use as a context manager; ``url`` is its base URL."""

    def __init__(self, kind: Kind = "llama", api_key: str | None = None, models: list[dict[str, Any]] | None = None,
                 reasoning: bool = False) -> None:
        self.kind: Kind = kind
        self.api_key: str | None = api_key
        self.reasoning: bool = reasoning
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []
        self.requests: list[tuple[str, str, Any]] = []
        self.draft_tokens: int = 0
        self.accepted_tokens: int = 0
        self.lock: threading.Lock = threading.Lock()
        if models is None:
            models = [{"id": CHAT_MODEL}, {"id": EMBEDDING_MODEL}]
            if kind == "llama":
                models = [
                    {"id": EMBEDDING_MODEL, "status": {"value": "loaded", "args": ["llama-server", "--embeddings"]}},
                    {"id": CHAT_MODEL, "status": {"value": "loaded", "args": ["llama-server", "--ctx-size", "4096"]}},
                    {"id": "unloaded-model", "status": {"value": "unloaded"}},
                ]
        self.models: list[dict[str, Any]] = models
        self.httpd: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url: str = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread: threading.Thread = threading.Thread(target=self.httpd.serve_forever, args=(0.01,), daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def paths(self) -> list[str]:
        """The paths of the received requests."""
        return [path for _, path, _ in self.requests]

    # Tokenizer

    def tokenize(self, text: str) -> list[int]:
        """Split the text at spaces (keeping the spaces) and map the pieces to ids."""
        pieces = [p for p in text.replace(" ", "\0 ").split("\0") if p]
        with self.lock:
            for p in pieces:
                if p not in self.vocab:
                    self.vocab[p] = len(self.words)
                    self.words.append(p)
        return [self.vocab[p] for p in pieces]

    def detokenize(self, tokens: list[int]) -> str:
        """Join the pieces of the token ids."""
        return "".join(self.words[t] for t in tokens)

    # Responses

    def timings(self, prompt_n: int, predicted_n: int) -> dict[str, Any]:
        """llama-server's timings object with fixed speeds."""
        return {"prompt_n": prompt_n, "prompt_ms": prompt_n, "prompt_per_second": 1000.0,
                "predicted_n": predicted_n, "predicted_ms": 10.0 * predicted_n, "predicted_per_second": 100.0,
                "draft_n": 4, "draft_n_accepted": 3}

    def stream_chunks(self, chat: bool, prompt_n: int, max_tokens: int) -> list[dict[str, Any]]:
        """The chunks of a streamed response: reasoning (optional), the reply and the usage."""
        reasoning = REASONING if self.reasoning and chat else []
        reply = REPLY[: max(max_tokens - len(reasoning), 0)]
        chunks: list[dict[str, Any]] = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": piece}}]} for piece in reasoning
        ]
        for piece in reply:
            choice = {"index": 0, "delta": {"content": piece}} if chat else {"index": 0, "text": piece}
            chunks.append({"choices": [choice]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": prompt_n,
                                                "completion_tokens": len(reasoning) + len(reply)}})
        with self.lock:
            self.draft_tokens += 4
            self.accepted_tokens += 2
        return chunks

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the signature of the base class)
                pass

            def send(self, code: int, body: Any, content_type: str = "application/json") -> None:
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self) -> bool:
                if server.api_key and self.headers.get("Authorization") != f"Bearer {server.api_key}":
                    self.send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self) -> None:
                server.requests.append(("GET", self.path, None))
                path = self.path.split("?")[0]
                if path == "/health":
                    self.send(200, {})
                elif not self.authorized():
                    return
                elif path == "/version" and server.kind == "vllm":
                    self.send(200, {"version": "0.0.0-fake"})
                elif path == "/v1/config" and server.kind == "ovms":
                    self.send(200, {EMBEDDING_MODEL: {"model_version_status": []}})
                elif path == "/v1/models":
                    self.send(200, {"object": "list", "data": server.models})
                elif path == "/props" and server.kind == "llama":
                    self.send(200, {"build_info": "b0-fake", "total_slots": 2, "model_path": "/models/fake.gguf"})
                elif path == "/metrics" and server.kind == "vllm":
                    self.send(200, f"vllm:spec_decode_num_draft_tokens_total{{}} {server.draft_tokens}.0\n"
                                   f"vllm:spec_decode_num_accepted_tokens_total{{}} {server.accepted_tokens}.0\n",
                              "text/plain")
                else:
                    self.send(404, {"error": "not found"})

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                server.requests.append(("POST", self.path, payload))
                if not self.authorized():
                    return
                path = self.path
                kind = server.kind
                if path == "/tokenize" and kind != "ovms":
                    text = payload["content"] if kind == "llama" else payload["prompt"]
                    self.send(200, {"tokens": server.tokenize(text)})
                elif path == "/v3/tokenize" and kind == "ovms":
                    self.send(200, {"tokens": server.tokenize(payload["text"])})
                elif path == "/detokenize" and kind != "ovms":
                    key = "content" if kind == "llama" else "prompt"
                    self.send(200, {key: server.detokenize(payload["tokens"])})
                elif path == "/v1/embeddings":
                    texts = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
                    self.send(200, {
                        "data": [{"index": i, "embedding": embed_text(t)} for i, t in enumerate(texts)],
                        "usage": {"prompt_tokens": sum(len(server.tokenize(t)) for t in texts)},
                    })
                elif path == "/completion" and kind == "llama":
                    n = len(server.tokenize(payload["prompt"]))
                    self.send(200, {"content": "".join(REPLY), "timings": server.timings(n, payload["n_predict"])})
                elif path in ("/v1/chat/completions", "/v1/completions"):
                    self.completions(payload, chat=path == "/v1/chat/completions")
                else:
                    self.send(404, {"error": "not found"})

            def completions(self, payload: dict[str, Any], chat: bool) -> None:
                if chat:
                    content = payload["messages"][-1]["content"]
                    text = content if isinstance(content, str) else " ".join(
                        c.get("text", "<image>") for c in content)
                else:
                    text = payload["prompt"]
                prompt_n = len(server.tokenize(text))
                max_tokens = payload.get("max_tokens", 16)
                if not payload.get("stream"):
                    reply = "".join(REPLY)
                    self.send(200, {
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}],
                        "usage": {"prompt_tokens": prompt_n, "completion_tokens": len(REPLY)},
                        "timings": server.timings(prompt_n, len(REPLY)),
                    })
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in server.stream_chunks(chat, prompt_n, max_tokens):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
