import { useEffect, useState } from 'react'
import { Loader2, Search } from 'lucide-react'
import { search } from '../api/client'
import type { SearchHit, SearchMode } from '../api/types'
import { formatScore } from '../lib/format'

const MODES: SearchMode[] = ['hybrid', 'vector', 'bm25']

export default function SearchTest() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [topK, setTopK] = useState(5)
  const [hits, setHits] = useState<SearchHit[]>([])
  const [latency, setLatency] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const q = query.trim()
    if (!q) {
      setHits([])
      setLatency(null)
      setError(null)
      return
    }
    setLoading(true)
    const timer = setTimeout(() => {
      search({ query: q, mode, top_k: topK })
        .then((r) => {
          setHits(r.hits)
          setLatency(r.latency_ms)
          setError(null)
        })
        .catch((e) => setError(e instanceof Error ? e.message : '检索失败'))
        .finally(() => setLoading(false))
    }, 350)
    return () => clearTimeout(timer)
  }, [query, mode, topK])

  return (
    <div className="card search-test">
      <div className="card-head">
        <Search size={20} />
        <span>检索测试</span>
        <div className="seg">
          {MODES.map((m) => (
            <button
              key={m}
              className={'seg-btn' + (mode === m ? ' active' : '')}
              onClick={() => setMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="search-bar">
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入检索语句，实时查看混合检索命中与分数构成…"
        />
        <label className="k-field">
          top_k
          <input
            className="input num"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
          />
        </label>
      </div>
      <div className="search-meta">
        {loading && (
          <span className="meta">
            <Loader2 size={16} className="spin" /> 检索中…
          </span>
        )}
        {latency !== null && !loading && (
          <span className="meta">
            耗时 {latency}ms · {hits.length} 命中
          </span>
        )}
        {error && <span className="error-box inline">{error}</span>}
      </div>
      <table className="hit-table">
        <thead>
          <tr>
            <th>#</th>
            <th>文档</th>
            <th>原文片段</th>
            <th>融合分</th>
            <th>BM25</th>
            <th>向量</th>
          </tr>
        </thead>
        <tbody>
          {hits.map((h) => (
            <tr key={h.chunk_id}>
              <td>{h.rank}</td>
              <td className="mono">{h.doc_id.slice(0, 8)}</td>
              <td className="hit-text">{h.text}</td>
              <td className="score">{formatScore(h.score)}</td>
              <td className="score">{formatScore(h.bm25_score)}</td>
              <td className="score">{formatScore(h.vector_score)}</td>
            </tr>
          ))}
          {hits.length === 0 && !loading && query.trim() && (
            <tr>
              <td colSpan={6} className="empty">
                无命中
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
