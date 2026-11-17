# usage-band

A Claude Code mod (a plugin with a hooks module) that shows the context window fill and the Claude usage limits
in a band above the prompt and in the status line, e.g.

```
ctx 42% (84k/200k) · 5h 23% ↻14:00 · 7d 61% ↻Fri
```

It is mainly kept as a template for developing further mods.

## Usage
Load it for a single session from the repository root:
```bash
claude --plugin-dir claude/mods/usage-band
```
To load it in every session, add its absolute path to `CLAUDE_CODE_PLUGIN_DIRS`
in the `env` block of `~/.claude/settings.json`.

The figures update when the session starts and whenever the engine measures the session
(after each turn, and when a usage limit window moves a whole percentage point).
The context fill is shown as `–` until the first reply,
and the usage limits are shown only when using a Claude subscription.

## Limitations
- The band above the prompt is drawn only in the terminal and in the Claude Desktop app, not on mobile or in VS Code.
- Nothing is shown in Remote Control (`claude rc`) sessions viewed in Claude Desktop.
  The Desktop app does not attach to such a session as a surface that mods can draw on,
  so the session's `$.session.surfaces()` is empty and the band is never rendered.
  Running the mod on the client computer does not help either,
  since mods run in the Claude Code process of the session, which is on the Remote Control server.
  (Tested with Claude Code 2.1.309.)

## Files
- `.claude-plugin/plugin.json`: the plugin manifest
- `hooks/hooks.json`: lists the hooks module
- `hooks/register.tsx`: the hooks module
- `types/index.d.ts`: the type contract of the values the mod keeps in `$.state`
- `tsconfig.json`: extends `.claude-plugin/types/tsconfig.json`

Claude Code writes the TypeScript declarations of the plugin API, the built-in tools and the connected MCP tools
to `.claude-plugin/types/` each time it loads the mod.
That directory has its own `.gitignore`, so it is not committed.
On a fresh clone, load the mod once (e.g. with `--plugin-dir`) before type-checking it.

## Development
```bash
claude plugin validate claude/mods/usage-band
tsc -p claude/mods/usage-band
```
Claude Code reloads the mod when its files change in an interactive session started with `--plugin-dir`.
Use `claude --debug` to see the hooks that failed or were skipped.
