---
name: check-usage
description: Check the Claude usage limit status (5-hour and weekly usage percentages, whether limited, overage, reset times). Use when the user asks about their Claude usage, usage limits, rate limits or when the limit resets, or before starting long work that may hit the limit.
---

# Check the Claude usage limits

The parser `check_usage.py` is in this skill's directory: `~/.claude/skills/check-usage/check_usage.py`
(`agents/skills/check-usage/` in the agx-ai repository). It prints `key: value` lines and exits with 1 on an `error:` line.

## 1. The usage file saved by the status line (preferred)

The status line (`agx_ai/claude/statusline.py` in the agx-ai repository) saves the rate limits it receives
to `~/.claude/usage.json`, with the time they were obtained. This includes the percentages used,
and does not need any data in your context. Run:
```bash
python3 ~/.claude/skills/check-usage/check_usage.py
```
Example output:
```
source: status line usage file
obtained_at_local: 2026-11-15 15:22:19 EET (UTC+0200)
data_age: 0h 0m
five_hour.used: 31 %
five_hour.resets_at_utc: 2026-11-15 15:50 UTC
five_hour.resets_at_local: 2026-11-15 17:50 EET (UTC+0200)
five_hour.resets_in: 2h 27m
seven_day.used: 12 %
seven_day.resets_at_utc: 2026-11-20 18:40 UTC
seven_day.resets_at_local: 2026-11-20 20:40 EET (UTC+0200)
seven_day.resets_in: 5d 5h 17m
```
- The data is only as fresh as the last status line update of any session on this computer: report `data_age`.
  A `<window>.note` line means that the window has reset since then, so its percentage is outdated.
- The status line does not run in headless sessions (e.g. `claude rc`), so the file may be missing or old.
  Then also use one of the methods below.

## 2. In a local session of the Claude desktop app

If the tool `mcp__ccd_session_mgmt__get_usage` exists (load it with ToolSearch `select:mcp__ccd_session_mgmt__get_usage`),
call it and report its output.

## 3. In a remote session (`claude rc` or claude.ai/code)

1. Call `mcp__claude-code-remote__get_session` without a `session_id`, so that it describes the current session.
2. Pass its output unmodified to the parser via stdin with a quoted heredoc, so that the shell does not expand anything:
   ```bash
   python3 ~/.claude/skills/check-usage/check_usage.py - <<'EOF'
   <the get_session output>
   EOF
   ```
3. The output has `status`, `limited`, `limit_window`, `using_overage` and the reset time of the current limit window,
   and `session_cost_usd` and `session_context` for this session only (these may be missing early in a session).
   `other.*` lines are fields of `rate_limit_info` that the parser does not know yet.
   This method does not give the percentages used.

## Limitations

If none of the above give the percentages, tell the user to use `/usage` in the CLI or claude.ai → Settings → Usage.
