import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { getTrace, listTraces } from '../api/client'
import type { TraceDetail, TraceSummary } from '../api/types'
import SpanWaterfall from './SpanWaterfall'
import { formatTime } from '../lib/format'

export default function TracePanel({
  pendingTrace,
  onConsume,
}: {
  pendingTrace: string | null
  onConsume: () => void
}) {
  const [items, setItems] = useState<TraceSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(pendingTrace)
  const [detail, setDetail] = useState<TraceDetail | null>(null)

  const loadList = () => {
    setLoading(true)
    listTraces()
      .then((d) => {
        setItems(d.items)
        setError(null)
      })
      .catch((e) => setError(e instanceof Error ? e.message : '加载失败'))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadList()
  }, [])

  useEffect(() => {
    if (pendingTrace) {
      setSelected(pendingTrace)
      onConsume()
    }
  }, [pendingTrace, onConsume])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }
    getTrace(selected)
      .then((d) => setDetail(d))
      .catch(() => setDetail(null))
  }, [selected])

  return (
    <div className="panel">
      <header className="panel-head">
        <div>
          <h1>Trace</h1>
          <p className="panel-sub">请求链路与耗时瀑布</p>
        </div>
      </header>
      <div className="trace-grid">
        <div className="trace-list card">
          <div className="card-head">
            <span>链路列表</span>
            <button className="btn ghost" onClick={loadList}>
              刷新
            </button>
          </div>
          {loading && (
            <div className="meta row">
              <Loader2 size={16} className="spin" /> 加载中…
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
          {!loading && !error && items.length === 0 && <div className="empty">暂无链路记录。</div>}
          <ul className="trace-items">
            {items.map((t) => (
              <li key={t.trace_id}>
                <button
                  className={'trace-row' + (selected === t.trace_id ? ' active' : '')}
                  onClick={() => setSelected(t.trace_id)}
                >
                  <div className="trace-query">{t.query}</div>
                  <div className="meta">
                    {formatTime(t.ts)} · {t.total_ms}ms · {t.n_spans} spans
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="trace-detail">
          {!detail && <div className="empty card">从左侧选择一个链路查看 span 瀑布图。</div>}
          {detail && <SpanWaterfall detail={detail} />}
        </div>
      </div>
    </div>
  )
}
