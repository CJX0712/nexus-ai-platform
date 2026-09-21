import { useEffect, useRef, useState } from 'react'
import { Loader2, MessageSquare, Send } from 'lucide-react'
import type { Citation, Health, ToolCall, Usage } from '../api/types'
import { streamChat } from '../api/client'
import MessageItem from './MessageItem'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  toolCalls: ToolCall[]
  traceId?: string
  usage?: Usage
  streaming: boolean
  error?: string
}

let counter = 0
const newId = () => `m${++counter}`

export default function ChatPanel({
  health,
  onOpenTrace,
}: {
  health: Health | null
  onOpenTrace: (id: string) => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionId] = useState<string>(() => crypto.randomUUID())
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTo({ top: el.scrollHeight })
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || busy) return
    setInput('')

    const aiId = newId()
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: 'user', content: q, citations: [], toolCalls: [], streaming: false },
      { id: aiId, role: 'assistant', content: '', citations: [], toolCalls: [], streaming: true },
    ])
    setBusy(true)

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === aiId ? fn(m) : m)))

    try {
      await streamChat(
        { query: q, session_id: sessionId, use_rag: true, top_k: 5, provider: health?.provider },
        {
          onToken: (delta) => patch((m) => ({ ...m, content: m.content + delta })),
          onCitation: (c) => patch((m) => ({ ...m, citations: [...m.citations, c] })),
          onTool: (t) => patch((m) => ({ ...m, toolCalls: [...m.toolCalls, t] })),
          onDone: (d) =>
            patch((m) => ({ ...m, streaming: false, traceId: d.trace_id, usage: d.usage })),
          onError: (msg) => patch((m) => ({ ...m, streaming: false, error: msg })),
        },
      )
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      patch((m) => ({ ...m, streaming: false, error: msg }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel chat">
      <header className="panel-head">
        <div>
          <h1>对话</h1>
          <p className="panel-sub">基于知识库的检索增强对话</p>
        </div>
        <div className="head-right">
          <span className="provider">{health?.provider ?? '—'}</span>
          <span className={'pill ' + (health?.status === 'ok' ? 'ok' : 'bad')}>
            <span className="dot" />
            {health?.status === 'ok' ? '在线' : health ? '异常' : '未连接'}
          </span>
        </div>
      </header>

      <div className="chat-body" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <MessageSquare size={28} color="var(--accent)" />
            <h2>开始一次检索增强对话</h2>
            <p>输入问题后，系统会先检索知识库再生成回答，并附上引用来源与可追踪的链路 ID。</p>
          </div>
        ) : (
          messages.map((m) => <MessageItem key={m.id} msg={m} onOpenTrace={onOpenTrace} />)
        )}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <textarea
          className="input textarea composer-input"
          value={input}
          placeholder="输入问题，回车发送（Shift+Enter 换行）"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
          rows={2}
        />
        <button className="btn primary" type="submit" disabled={busy || !input.trim()}>
          {busy ? <Loader2 size={20} className="spin" /> : <Send size={20} />}
          <span>发送</span>
        </button>
      </form>
    </div>
  )
}
