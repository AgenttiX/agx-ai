import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, SessionMeasureInput, SessionUsage } from 'claude-code'

import type { Usage } from '../types'

const usage = atom({ plugin: 'usage-band', key: 'usage' } as const, null)

const LIMIT_NAMES: Record<string, string> = { five_hour: '5h', seven_day: '7d', spend_limit: 'spend' }

function formatTokens(n: number): string {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

function formatReset(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  const hours = (d.getTime() - Date.now()) / 3_600_000
  if (hours < 24) {
    return ` ↻${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  return ` ↻${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()]}`
}

function format(u: Usage): string {
  const parts: string[] = []
  if (u.contextTokens !== undefined && u.contextPercent !== undefined) {
    parts.push(`ctx ${u.contextPercent}% (${formatTokens(u.contextTokens)}/${formatTokens(u.contextWindow)})`)
  } else {
    parts.push(`ctx –/${formatTokens(u.contextWindow)}`)
  }
  for (const l of u.limits) {
    parts.push(`${LIMIT_NAMES[l.kind] ?? l.kind} ${l.percentUsed}%${formatReset(l.resetsAt)}`)
  }
  return parts.join(' · ')
}

async function store($: EngineInterface, m: SessionMeasureInput | SessionUsage): Promise<void> {
  const u: Usage = {
    contextTokens: m.context.tokens,
    contextWindow: m.context.window,
    contextPercent: m.context.percent,
    limits: m.rateLimits.map(l => ({ kind: l.kind, percentUsed: l.percentUsed, resetsAt: l.resetsAt })),
  }
  await update($, usage, () => u)
  $.ui.status(format(u))
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const result = await next(e)
    await store($, await $.session.usage())
    return result
  })

  on('session.measure', async ($, e, next) => {
    await store($, e)
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const u = await read($, usage)
    if (e.props.hasSurvey) {
      return next(e)
    }
    const { Box, Text } = $.ui.resolve(e)
    return (
      <Box>
        <Text dimColor wrap="truncate-end">{u === null ? 'usage: waiting for the first reply' : format(u)}</Text>
      </Box>
    )
  })
}
