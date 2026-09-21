import { useCallback, useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import KnowledgePanel from './components/KnowledgePanel'
import TracePanel from './components/TracePanel'
import { getHealth } from './api/client'
import type { Health } from './api/types'

export type Panel = 'chat' | 'knowledge' | 'trace'

export default function App() {
  const [panel, setPanel] = useState<Panel>('chat')
  const [health, setHealth] = useState<Health | null>(null)
  const [pendingTrace, setPendingTrace] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = () =>
      getHealth()
        .then((h) => {
          if (alive) setHealth(h)
        })
        .catch(() => {})
    load()
    const timer = setInterval(load, 30000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  const openTrace = useCallback((traceId: string) => {
    setPendingTrace(traceId)
    setPanel('trace')
  }, [])

  return (
    <div className="app">
      <Sidebar panel={panel} onSelect={setPanel} health={health} />
      <main className="main">
        {panel === 'chat' && <ChatPanel health={health} onOpenTrace={openTrace} />}
        {panel === 'knowledge' && <KnowledgePanel />}
        {panel === 'trace' && (
          <TracePanel pendingTrace={pendingTrace} onConsume={() => setPendingTrace(null)} />
        )}
      </main>
    </div>
  )
}
