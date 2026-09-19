# Open WebUI chat export → PDF

`owui2pdf.py` converts a chat exported from Open WebUI as JSON
(*Chat menu → Download → Export chat (.json)*, or *Settings → Chats → Export all chats*)
into a PDF. `example.json` is a sample export that exercises headings, emphasis,
links, inline and display LaTeX, tables, highlighted code, nested lists, special
characters, emoji, attachments/sources and long text.

```bash
./owui2pdf.py example.json                 # writes example.pdf next to the input
./owui2pdf.py example.json -o out.pdf
./owui2pdf.py all-chats.json               # one PDF per chat: all-chats-001-<title>.pdf, ...
./owui2pdf.py exports/                     # every *.json below exports/, PDF next to each JSON
./owui2pdf.py exports/ --outdir pdfs       # same, mirroring the folder layout under pdfs/
./owui2pdf.py exports/ --skip-existing     # only JSON files without an up-to-date PDF
./owui2pdf.py --help
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
| `owui2pdf.py` | Launcher script (`python -m owui2pdf` works too) |
| `owui2pdf/cli.py` | Argument parsing, output file naming |
| `owui2pdf/chat.py` | Reading the export, walking the conversation branch, attachments, sources, usage |
| `owui2pdf/markdown.py` | Normalising message content and assembling the Markdown document |
| `owui2pdf/sources.py` | Registry of referenced notes/files/web pages: citations, `.bib` generation, appendix |
| `owui2pdf/fonts.py` | Font discovery (incl. the emoji font) and filling in the LaTeX header |
| `owui2pdf/build.py` | Running pandoc and latexmk |
| `owui2pdf/util.py` | Escaping, timestamps, subprocess helpers |
| `owui2pdf/resources/header.tex` | LaTeX preamble: fonts, code blocks, lists, headings, message headers |
| `owui2pdf/resources/filter.lua` | pandoc Lua filter: code blocks, task lists, tables, images, long words, emoji |

## Requirements

* Python 3.10+ (standard library only)
* pandoc ≥ 3
* TeX Live with `latexmk`, LuaLaTeX and `biber`, including `fontspec`,
  `unicode-math`, `biblatex`, `fvextra`, `tcolorbox`, `enumitem`, `titlesec`,
  `needspace`, `mathtools`, `cancel`, `braket` and optionally `mhchem`
* Fonts: Noto Serif / Noto Sans / Noto Sans Mono (falls back to TeX Gyre and
  Latin Modern) and a colour emoji font. The emoji font is located with
  `fc-list`, in `~/.local/share/fonts`, in a `fonts/` directory next to the
  script or inside Flatpak runtimes; pass `--emoji-font PATH` to override.
  Only CBDT (Noto Color Emoji), COLRv0 (Twemoji Mozilla) and sbix fonts work
  with LuaLaTeX; COLRv1 fonts are skipped.

## Options

| Option | Effect |
| --- | --- |
| `-o FILE`, `--outdir DIR` | Output location (default: next to each JSON file) |
| `--skip-existing` | Skip JSON files whose PDF already exists and is newer |
| `--keep-tex` | Keep the LaTeX sources in `<name>_tex/` next to the PDF |
| `--hard-breaks` | Treat single newlines as line breaks |
| `--include-reasoning` | Render model reasoning blocks as quotes |
| `--include-notes` | Append the full text of referenced notes as an appendix |
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
