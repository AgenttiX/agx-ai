# Open WebUI chat export → PDF

`agx_ai.owui2pdf` converts a chat exported from Open WebUI as JSON
(*Chat menu → Download → Export chat (.json)*, or *Settings → Chats → Export all chats*)
into a PDF. `example.json` is a sample export that exercises headings, emphasis,
links, inline and display LaTeX, tables, highlighted code, nested lists, special
characters, emoji, attachments/sources and long text.

Run it with `uv run owui2pdf` (or `uv run -m agx_ai.owui2pdf`) in any directory of the repository, or with
`python3 -m agx_ai.owui2pdf` from the repository root (only the standard library is needed):

```bash
cd agx_ai/owui2pdf
uv run owui2pdf example.json                 # writes example.pdf next to the input
uv run owui2pdf example.json -o out.pdf
uv run owui2pdf all-chats.json               # one PDF per chat: all-chats-001-<title>.pdf, ...
uv run owui2pdf exports/                     # every *.json below exports/, PDF next to each JSON
uv run owui2pdf exports/ --outdir pdfs       # same, mirroring the folder layout under pdfs/
uv run owui2pdf exports/ --skip-existing     # only JSON files without an up-to-date PDF
uv run owui2pdf --help
```

Inputs may be JSON files or directories, which are searched recursively for
`*.json`. A file containing a single chat becomes `<name>.pdf`; a file with
several chats becomes `<name>-001-<title>.pdf`, `<name>-002-<title>.pdf`, …
JSON files that are not chat exports are reported and skipped.

## How it works

1. The messages of the currently selected conversation branch are collected
   (`chat.history.currentId` followed back to the root).
2. Each message becomes a Markdown section with a coloured header
   (role, model, timestamp, token usage). Open WebUI specifics are normalised:
   `<details type="reasoning">` blocks are dropped (use `--include-reasoning` to
   keep them), tool calls are summarised, `\( … \)` / `\[ … \]` math is turned
   into `$ … $` / `$$ … $$`, base64 images are extracted, attachments are listed,
   and unclosed code fences are closed.
   Source markers such as `[1]` become biblatex citations; every referenced
   note, file or web page gets one numbered entry in a *References* section
   at the end (biber backend, numeric style, ordered by first citation).
   Notes and files are listed with their creation and update time, character
   count and word count only. With `--include-notes` the full text of each
   referenced note is added as an appendix that the bibliography entry links to.

   A chat export does not necessarily contain the whole note: Open WebUI's
   note picker stores the note as returned by the notes *list* API, which cuts
   the text at 1000 characters, and a reply's `sources` only hold the
   retrieved chunks. The script detects this and reports "at least N
   characters … (text truncated in the export)" or "… in K retrieved
   excerpts" instead of a misleading exact count. To get exact counts and the
   complete appendix, supply the notes:

   ```bash
   # from the server (API key: Settings → Account → API keys)
   OWUI_API_KEY=sk-... uv run owui2pdf chat.json --include-notes --notes-url https://ai.example.com
   # or from files: GET /api/v1/notes/<id> saved as JSON, or <note id>.md files
   uv run owui2pdf chat.json --include-notes --notes notes/
   ```

   `--notes-url` fetches every referenced note from `<url>/api/v1/notes/<id>`
   and sends the API key only to that host.
3. pandoc converts the Markdown (`commonmark_x`, close to the GFM dialect Open WebUI
   uses) into LaTeX. A Lua filter handles plain code blocks, task lists,
   table column widths, unreachable images, very long words and emoji sequences.
4. latexmk + LuaLaTeX build the PDF. Emoji are set in a colour emoji font
   (Noto Color Emoji or Twemoji Mozilla) through HarfBuzz so ZWJ sequences,
   flags and skin tones render as one glyph. Missing glyphs fall back to
   Noto Sans Symbols / DejaVu Sans / Noto Sans CJK when those are installed.

## Layout

| Path | Contents |
| --- | --- |
| `__main__.py` | Entry point of `python -m agx_ai.owui2pdf` |
| `cli.py` | Argument parsing, output file naming |
| `chat.py` | Reading the export, walking the conversation branch, attachments, sources, usage |
| `markdown.py` | Normalising message content and assembling the Markdown document |
| `sources.py` | Registry of referenced notes/files/web pages: citations, `.bib` generation, appendix |
| `notes.py` | Loading full note texts from files or the Open WebUI API |
| `fonts.py` | Font discovery (incl. the emoji font) and filling in the LaTeX header |
| `build.py` | Running pandoc and latexmk |
| `util.py` | Escaping, timestamps, subprocess helpers |
| `resources/header.tex` | LaTeX preamble: fonts, code blocks, lists, headings, message headers |
| `resources/filter.lua` | pandoc Lua filter: code blocks, task lists, tables, images, long words, emoji |

## Requirements

* Python 3.10+ (standard library only)
* pandoc ≥ 3
* TeX Live with `latexmk`, LuaLaTeX and `biber`, including `fontspec`,
  `unicode-math`, `biblatex`, `fvextra`, `tcolorbox`, `enumitem`, `titlesec`,
  `needspace`, `mathtools`, `cancel`, `braket` and optionally `mhchem`
* Fonts: Noto Serif / Noto Sans / Noto Sans Mono (falls back to TeX Gyre and
  Latin Modern) and a colour emoji font. The emoji font is located with
  `fc-list`, in `~/.local/share/fonts`, in a `fonts/` directory in this
  directory or inside Flatpak runtimes; pass `--emoji-font PATH` to override.
  Only CBDT (Noto Color Emoji), COLRv0 (Twemoji Mozilla) and sbix fonts work
  with LuaLaTeX; COLRv1 fonts are skipped.

## Options

| Option | Effect |
| --- | --- |
| `-o FILE`, `--outdir DIR` | Output location (default: next to each JSON file) |
| `--skip-existing` | Skip JSON files whose PDF already exists and is newer |
| `--keep-tex` | Keep the LaTeX sources in `<name>_tex/` next to the PDF |
| `--hard-breaks` | Treat single newlines as line breaks (all messages) |
| `--user-paragraphs` | In user messages, treat single newlines between plain text lines as paragraph breaks; lists, tables, quotes, headings and code are left alone |
| `--include-reasoning` | Render model reasoning blocks as quotes |
| `--include-notes` | Append the full text of referenced notes as an appendix |
| `--notes PATH`, `--notes-url URL`, `--notes-token KEY` | Supply the complete text of referenced notes from files or from the server (see above) |
| `--no-sources`, `--no-usage` | Omit citations and bibliography / token usage |
| `--main-font`, `--sans-font`, `--mono-font`, `--math-font` | Font overrides |
| `--paper`, `--font-size`, `--highlight-style` | Layout tweaks |
| `--strict` | Stop at the first LaTeX error instead of producing a best-effort PDF |

LaTeX errors caused by unusual content (for example KaTeX-only macros) are
reported as warnings and the PDF is still produced unless `--strict` is given.

## Temporary files and privacy

Every conversion writes its intermediate files (the Markdown document, the
LaTeX sources, extracted images, `refs.bib`, and the TeX/biber auxiliary
files such as `.aux`, `.log`, `.bcf`, `.bbl`) into one build directory.
The chat text is contained in these files, so they are kept private:

* **Default:** the build directory is `owui2pdf-<random>` under the system
  temporary directory (`$TMPDIR`, usually `/tmp`). It is created with mode
  `0700` (owner only) and deleted when the conversion of that chat finishes,
  also when pandoc or LaTeX fail. Set `TMPDIR` to move it elsewhere.
* **`--keep-tex`:** the build directory is `<pdf name>_tex/` next to the PDF.
  It is not deleted and, like the PDF itself, is created with the user's
  normal umask; adjust the umask if these should be private as well.

Nothing else is written outside the build directory by the script. Two
side effects of the tools it calls are worth knowing:

* LuaLaTeX keeps a font-name database and font caches in
  `TEXMFVAR/luatex-cache/` (typically `~/.texlive<year>/texmf-var/`). They
  contain only font data, never chat content, and are updated on first use.
* TeX Live's upstream `biber` binary (not the Debian/Ubuntu package, which is
  a plain Perl script) unpacks itself into `$TMPDIR/par-<user>/` on first run.
  That directory contains biber's own code only.

## Tests

The unit tests are in [`../../tests/test_owui2pdf.py`](../../tests/test_owui2pdf.py) (`uv run pytest tests/test_owui2pdf.py`).
They run pandoc but replace latexmk with a stub, as a LuaLaTeX build takes tens of seconds.
The real PDF build of `example.json` (about 30 s) runs only with `uv run pytest --run-slow`,
and is skipped if pandoc, latexmk, LuaLaTeX or biber is missing.
