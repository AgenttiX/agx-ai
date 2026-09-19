"""Obtaining the full text of referenced notes.

A chat export does not reliably contain the complete text of an attached
note: Open WebUI's note picker stores the note object returned by the notes
list API, which truncates the Markdown to 1000 characters, and the ``sources``
of a reply only carry the retrieved chunks. Full notes can be supplied from

* JSON files (``--notes``): a single note as returned by
  ``GET /api/v1/notes/<id>``, a list of notes, or ``{"items": [...]}``; or
  Markdown files named ``<note id>.md``; or directories of these, or
* the server (``--notes-url``): every referenced note is fetched from
  ``<url>/api/v1/notes/<id>`` with an API key.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .sources import SourceRegistry, _text_content


def _note_record(obj: dict) -> dict | None:
    if not isinstance(obj, dict) or not obj.get("id"):
        return None
    content = _text_content(obj)
    if content is None and isinstance(obj.get("content"), str):
        content = obj["content"]
    if content is None:
        return None
    return {
        "id": str(obj["id"]),
        "title": obj.get("title") or obj.get("name") or "",
        "content": content,
        "created_at": obj.get("created_at"),
        "updated_at": obj.get("updated_at"),
    }


def _records_from_json(data) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [r for r in (_note_record(o) for o in data) if r]


def load_note_library(paths: list[Path]) -> dict[str, dict]:
    """Read notes from files/directories; returns {note id: record}."""
    library: dict[str, dict] = {}
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in (".json", ".md"))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"error: --notes path not found: {p}")
    for f in files:
        try:
            if f.suffix.lower() == ".md":
                text = f.read_text(encoding="utf-8")
                library[f.stem] = {"id": f.stem, "title": f.stem, "content": text,
                                   "created_at": None, "updated_at": None}
                continue
            for rec in _records_from_json(json.loads(f.read_text(encoding="utf-8"))):
                library[rec["id"]] = rec
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"warning: ignoring {f}: {e}", file=sys.stderr)
    return library


def fetch_note(base_url: str, token: str, note_id: str) -> dict | None:
    """GET <base_url>/api/v1/notes/<id>; returns a record or None."""
    url = base_url.rstrip("/") + "/api/v1/notes/" + urllib.parse.quote(note_id, safe="")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _note_record(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        print(f"warning: could not fetch note {note_id} from {base_url}: HTTP {e.code}", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"warning: could not fetch note {note_id} from {base_url}: {e}", file=sys.stderr)
    return None


def resolve_notes(registry: SourceRegistry, opts) -> None:
    """Fill in the full text of note sources from --notes files and/or --notes-url."""
    library: dict[str, dict] = getattr(opts, "note_library", None) or {}
    base_url = getattr(opts, "notes_url", None)
    token = getattr(opts, "notes_token", None)
    cache: dict[str, dict | None] = getattr(opts, "_note_cache", None)
    if cache is None:
        cache = {}
        try:
            opts._note_cache = cache
        except AttributeError:
            pass
    for src in registry.sources.values():
        if src.kind != "note" or src.has_full_content:
            continue
        rec = library.get(src.ident)
        if rec is None and base_url:
            if src.ident not in cache:
                cache[src.ident] = fetch_note(base_url, token or "", src.ident)
            rec = cache[src.ident]
        if rec is None:
            # Last resort for file libraries: match on the title.
            rec = next((r for r in library.values() if r["title"] and r["title"] == src.title), None)
        if rec is not None:
            src.set_full_content(rec["content"], rec.get("created_at"), rec.get("updated_at"))
