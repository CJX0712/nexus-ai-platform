import { AlertTriangle, ArrowRight } from 'lucide-react'
import type { ChatMessage } from './ChatPanel'
import CitationList from './CitationList'
import ToolCalls from './ToolCalls'
import { shortId } from '../lib/format'

export default function MessageItem({
  msg,
  onOpenTrace,
}: {
  msg: ChatMessage
  onOpenTrace: (id: string) => void
}) {
  if (msg.role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble user">{msg.content}</div>
      </div>
    )
  }

  return (
    <div className="msg ai">
      <div className="bubble ai">
        {msg.content || (msg.streaming ? '…' : '')}
        {msg.streaming && <span className="caret" />}
      </div>

      {msg.toolCalls.length > 0 && <ToolCalls tools={msg.toolCalls} />}
      {msg.citations.length > 0 && <CitationList citations={msg.citations} />}

      {msg.error && (
        <div className="error-box">
          <AlertTriangle size={16} /> {msg.error}
        </div>
      )}

      {msg.traceId && msg.usage && (
        <div className="msg-foot">
          <button className="link" onClick={() => onOpenTrace(msg.traceId!)}>
            <ArrowRight size={16} /> trace {shortId(msg.traceId)}
          </button>
          <span className="meta">
            首字 {msg.usage.ttft_ms}ms · 总 {msg.usage.total_ms}ms
          </span>
          <span className="meta">
            {msg.usage.prompt_tokens}+{msg.usage.completion_tokens} tokens
          </span>
        </div>
      )}
    </div>
  )
}
