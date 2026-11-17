export type Usage = {
  contextTokens?: number
  contextWindow: number
  contextPercent?: number
  limits: { kind: string; percentUsed: number; resetsAt?: string }[]
}

declare module 'claude-code' {
  interface PluginState {
    'usage-band': { usage: Usage | null }
  }
}
