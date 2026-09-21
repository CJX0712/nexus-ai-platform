// 原生 fetch 封装；SSE 用 ReadableStream 逐块解析
import type {
  Health,
  DocumentList,
  IngestRequest,
  IngestResult,
  SearchRequest,
  SearchResult,
  ChatRequest,
  TraceList,
  TraceDetail,
  Citation,
  ToolCall,
  DoneEvent,
  ErrorEvent,
  Metrics,
} from './types'

const BASE = '/api'

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`请求失败 ${res.status}: ${body || res.statusText}`)
  }
  return (await res.json()) as T
}

export function getHealth() {
  return jsonFetch<Health>('/health')
}

export function ingest(payload: IngestRequest) {
  return jsonFetch<IngestResult>('/v1/ingest', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function listDocuments() {
  return jsonFetch<DocumentList>('/v1/documents')
}

export function deleteDocument(docId: string) {
  return jsonFetch<{ deleted: true; doc_id: string }>(
    `/v1/documents/${encodeURIComponent(docId)}`,
    { method: 'DELETE' },
  )
}

export function search(payload: SearchRequest) {
  return jsonFetch<SearchResult>('/v1/search', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function chat(payload: ChatRequest) {
  return jsonFetch<{
    session_id: string
    answer: string
    citations: Citation[]
    tool_calls: ToolCall[]
    trace_id: string
    usage: import('./types').Usage
  }>('/v1/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function listTraces() {
  return jsonFetch<TraceList>('/v1/traces')
}

export function getTrace(traceId: string) {
  return jsonFetch<TraceDetail>(`/v1/traces/${encodeURIComponent(traceId)}`)
}

export function getMetrics() {
  return jsonFetch<Metrics>('/v1/metrics')
}

export interface StreamHandlers {
  onToken: (delta: string) => void
  onCitation: (c: Citation) => void
  onTool: (t: ToolCall) => void
  onDone: (d: DoneEvent) => void
  onError: (msg: string) => void
}

// 解析一段 SSE 事件块（event: / data: 行，空行分隔）
function handleRawEvent(raw: string, h: StreamHandlers): void {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return
  let data: unknown
  try {
    data = JSON.parse(dataLines.join('\n'))
  } catch {
    return
  }
  switch (event) {
    case 'token':
      h.onToken((data as { delta: string }).delta)
      break
    case 'citation':
      h.onCitation(data as Citation)
      break
    case 'tool':
      h.onTool(data as ToolCall)
      break
    case 'done':
      h.onDone(data as DoneEvent)
      break
    case 'error':
      h.onError((data as ErrorEvent).message)
      break
  }
}

export async function streamChat(payload: ChatRequest, handlers: StreamHandlers): Promise<void> {
  const res = await fetch(BASE + '/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok || !res.body) {
    handlers.onError(`请求失败 ${res.status}`)
    return
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const pump = async (): Promise<void> => {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let sep: number
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        handleRawEvent(rawEvent, handlers)
      }
    }
    const tail = buffer.trim()
    if (tail) handleRawEvent(tail, handlers)
  }
  await pump()
}
