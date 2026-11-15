---
name: check-usage
description: Check the Claude usage limit status (rate limit window, whether limited, overage, reset time) of the current session. Use when the user asks about their Claude usage, usage limits, rate limits or when the limit resets, or before starting long work that may hit the limit.
---

# Check the Claude usage limits

## In a local session of the Claude desktop app

If the tool `mcp__ccd_session_mgmt__get_usage` exists (load it with ToolSearch `select:mcp__ccd_session_mgmt__get_usage`),
call it and report its output. It gives more detail than the method below, including the percentage used.

## In a remote session (`claude rc` or claude.ai/code)

1. Call `mcp__claude-code-remote__get_session` without a `session_id`, so that it describes the current session.
2. Pass its output unmodified to the parser in this skill's directory
   (`~/.claude/skills/check-usage/check_usage.py`, which is `agents/skills/check-usage/` in the agx-ai repository).
   Use a quoted heredoc so that the shell does not expand anything:
   ```bash
   python3 ~/.claude/skills/check-usage/check_usage.py <<'EOF'
   <the get_session output>
   EOF
   ```
3. Report the result to the user. The output is `key: value` lines, for example:
   ```
   status: allowed
   limited: no
   limit_window: five_hour
   using_overage: no
   resets_at_utc: 2026-11-15 15:50 UTC
   resets_at_local: 2026-11-15 17:50 EET (UTC+0200)
   resets_in: 2h 51m
   session_cost_usd: 0.37
   session_context: 88102/1000000 tokens (8.8 %)
   ```
   - `other.*` lines are fields of `rate_limit_info` that the parser does not know yet.
   - `session_cost_usd` and `session_context` are for this session only, and may be missing early in a session.
   - An `error:` line with exit code 1 means that the input had no `rate_limit_info` or was not valid JSON.

## Limitations

`rate_limit_info` does not include the percentage of the limit used or the weekly and per-model limits.
For those, tell the user to use `/usage` in the CLI, claude.ai → Settings → Usage,
or a local desktop app session with `mcp__ccd_session_mgmt__get_usage`.
