// 后端 API 契约对应的类型定义（后端会精确实现这些接口）

export interface Health {
  status: string
  provider: string
  version: string
  uptime_s: number
}

export interface IngestRequest {
  title: string
  text: string
  doc_id?: string
  metadata?: Record<string, string>
}

export interface IngestResult {
  doc_id: string
  n_chunks: number
  n_chars: number
}

export interface DocumentItem {
  doc_id: string
  title: string
  n_chunks: number
  created_at: string
}

export interface DocumentList {
  items: DocumentItem[]
  total: number
}

export type SearchMode = 'hybrid' | 'vector' | 'bm25'

export interface SearchRequest {
  query: string
  top_k?: number
  mode?: SearchMode
}

export interface SearchHit {
  chunk_id: string
  doc_id: string
  text: string
  score: number
  rank: number
  bm25_score: number | null
  vector_score: number | null
}

export interface SearchResult {
  hits: SearchHit[]
  latency_ms: number
}

export interface ChatRequest {
  query: string
  session_id?: string
  use_rag?: boolean
  top_k?: number
  provider?: string
}

export interface Citation {
  chunk_id: string
  doc_id: string
  text: string
  score: number
}

export interface ToolCall {
  name: string
  args: Record<string, unknown>
  result: string
  ok: boolean
}

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_ms: number
  ttft_ms: number
}

export interface ChatResult {
  session_id: string
  answer: string
  citations: Citation[]
  tool_calls: ToolCall[]
  trace_id: string
  usage: Usage
}

export interface DoneEvent {
  answer: string
  trace_id: string
  usage: Usage
}

export interface ErrorEvent {
  message: string
}

export interface TraceSummary {
  trace_id: string
  query: string
  total_ms: number
  n_spans: number
  ts: string
}

export interface TraceList {
  items: TraceSummary[]
  total: number
}

export interface Span {
  span_id: string
  parent_id: string | null
  name: string
  start_ms: number
  dur_ms: number
  attrs: Record<string, unknown>
}

export interface TraceDetail {
  trace_id: string
  spans: Span[]
}

export interface Metrics {
  requests: number
  errors: number
  p50_ms: number
  p95_ms: number
  tokens_total: number
  uptime_s: number
}
