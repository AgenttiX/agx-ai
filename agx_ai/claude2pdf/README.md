# Claude Code session → PDF

`agx_ai.claude2pdf` converts a Claude Code session, stored by Claude Code as
`~/.claude/projects/<project>/<session id>.jsonl`, into a PDF.
The rendering (pandoc, LuaLaTeX, fonts, emoji) is shared with
[`owui2pdf`](../owui2pdf/README.md) in [`agx_ai/convert/`](../convert/),
so the requirements are the same as there.

Run it with `uv run claude2pdf` (or `uv run -m agx_ai.claude2pdf`):

```bash
uv run claude2pdf ../phd-research                 # list the sessions of a repository
uv run claude2pdf ../phd-research -s 12 -s 3923b  # convert sessions by number in the list or by ID prefix
uv run claude2pdf ../phd-research -s all          # convert every session of the repository
uv run claude2pdf ~/.claude/projects/-home-user-repo/<id>.jsonl     # convert one session
uv run claude2pdf a.jsonl b.jsonl c.jsonl -j 4    # convert several sessions, at most 4 in parallel
uv run claude2pdf session.jsonl -l chat           # only the messages between the user and Claude
uv run claude2pdf session.jsonl -l full -o out.pdf
uv run claude2pdf --help
```

Given a repository, the path is turned into the name of its directory in
`~/.claude/projects/` (the absolute path with every character other than a
letter or a digit replaced by `-`), and its sessions are listed with their
start and last message times, the number of prompts, the file size, the ID
and the name (the title set with `/rename`, or else the start of the first
prompt). A directory of `~/.claude/projects/` itself works as well.

Given `.jsonl` files, they are converted in parallel (one job per CPU by default, `-j`).
The PDFs are written to `dist/` in the working directory (`--outdir`), named
`<start date>-<name>-<ID prefix>[-<level>].pdf`. The intermediate Markdown document is saved next to
each PDF as `<name>.md`, with the images in `<name>_media/` (`--no-markdown` to skip). It is pandoc
Markdown with raw LaTeX blocks for the message headers.

## Verbosity

| Level (`-l`) | Contents |
| --- | --- |
| `chat` | The prompts of the user (including slash commands and the prompts typed while Claude was working) and the text replies of Claude |
| `tools` (default) | Also the tool calls: the command of Bash, the file of Read, the diff of Edit, the content of Write, the prompt of Agent, … |
| `results` | Also the results of the tool calls, including images |
| `full` | Also the thinking and the text injected by Claude Code: skill instructions, subagent reports, task notifications and compaction summaries; nothing is truncated |

Interruptions, API errors and compaction notices are shown at every level.
Below `full`, tool calls and results are truncated to 40 lines (`--max-lines N`, 0 for no limit).
`--thinking` / `--no-thinking` and `--meta` / `--no-meta` override the level for the thinking and the injected text.
Claude Code stores the thinking of most responses only as a signature, so usually little of it is shown.

`--no-dates` leaves out the dates and `--no-times` the times of day (HH:MM), both below the title and in the
message headers; with both, no timestamps are shown.
The other options (fonts, paper, `--keep-tex`, `--strict`, …) are the same as in `owui2pdf`.

## Link to the session

If the session was connected to claude.ai (Remote Control, the desktop or mobile app), the session ID at the
top of the PDF is a link to the session at `https://claude.ai/code/<session ID on claude.ai>`.
The transcript does not contain that ID. While a session is running, Claude Code has it as `bridgeSessionId`
in the process file `~/.claude/sessions/<pid>.json`, but it removes it when the session ends.
Therefore the hook [`agx_ai/claude/session_urls.py`](../claude/session_urls.py), which runs at the start of every session and
on every prompt (`hooks` in [`claude/settings.json`](../../claude/settings.json)), copies it to
`~/.claude/session-urls.json`. claude2pdf reads both. Sessions that ended before the hook was set up, and sessions
not connected to claude.ai, get no link.

## How it works

1. The records of the transcript form a tree through `parentUuid`. Rewinding a
   conversation starts a new branch; only the active branch, which ends at
   the last message, is converted. Parallel tool calls also branch the tree,
   so the sibling records of the kept responses and tool calls are included.
2. The records become events: prompts, text, thinking, tool calls and results,
   and injected text, which is recognised by `isMeta`, `origin` and its tags.
   The context that Claude Code attaches (system reminders, environment,
   tool lists) is left out.
3. Each prompt gets a user header and each turn of Claude one header, with the
   tool calls and results under it as labelled code blocks.
   `<tags>` outside code are escaped so that they are shown as in Claude Code
   instead of being dropped as HTML. Terminal escape sequences are removed from tool output.
4. The Markdown document is built into a PDF with pandoc and LuaLaTeX by
   `agx_ai.convert.build`, with the extra LaTeX commands of `markdown.HEADER_EXTRA`.

Subagent transcripts (`<session id>/subagents/`) are not included; their prompts
and final reports are part of the main session as tool calls and results.

## Layout

| Path | Contents |
| --- | --- |
| `__main__.py` | Entry point of `python -m agx_ai.claude2pdf` |
| `cli.py` | Argument parsing, listing and selecting sessions, parallel conversion |
| `session.py` | Reading the transcript, the active branch, events, session metadata |
| `markdown.py` | Assembling the Markdown document at the chosen level |

## Tests

The unit tests are in [`../../tests/test_claude2pdf.py`](../../tests/test_claude2pdf.py)
(`uv run pytest tests/test_claude2pdf.py`). They use a synthetic transcript and run pandoc
with a stub in place of latexmk.
