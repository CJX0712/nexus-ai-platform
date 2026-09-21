import { useMemo, useState } from 'react'
import type { Span, TraceDetail } from '../api/types'

interface Row extends Span {
  depth: number
}

export default function SpanWaterfall({ detail }: { detail: TraceDetail }) {
  const { rows, start, span } = useMemo(() => {
    const spans = detail.spans
    if (spans.length === 0) {
      return { rows: [] as Row[], start: 0, span: 1 }
    }
    const minStart = Math.min(...spans.map((s) => s.start_ms))
    const maxEnd = Math.max(...spans.map((s) => s.start_ms + s.dur_ms))
    const byId = new Map(spans.map((s) => [s.span_id, s]))

    const depthOf = (s: Span): number => {
      let d = 0
      let cur: Span | undefined = s
      let guard = 0
      while (cur && cur.parent_id && byId.has(cur.parent_id) && guard++ < 64) {
        d++
        cur = byId.get(cur.parent_id)
      }
      return d
    }

    const sorted = [...spans].sort((a, b) => a.start_ms - b.start_ms)
    const rows: Row[] = sorted.map((s) => ({ ...s, depth: depthOf(s) }))
    return { rows, start: minStart, span: Math.max(1, maxEnd - minStart) }
  }, [detail])

  const [hover, setHover] = useState<Span | null>(null)

  return (
    <div className="card waterfall">
      <div className="card-head">
        <span>Span 瀑布 · {detail.trace_id.slice(0, 12)}</span>
        <span className="meta">总跨度 {span}ms</span>
      </div>
      <div className="wf-rows">
        {rows.map((r) => {
          const left = ((r.start_ms - start) / span) * 100
          const width = Math.max(0.5, (r.dur_ms / span) * 100)
          return (
            <div key={r.span_id} className="wf-row" style={{ paddingLeft: r.depth * 16 }}>
              <div className="wf-label" title={r.name}>
                <span className="wf-name">{r.name}</span>
                <span className="meta">{r.dur_ms}ms</span>
              </div>
              <div className="wf-track">
                <div
                  className="wf-bar"
                  style={{ left: `${left}%`, width: `${width}%` }}
                  onMouseEnter={() => setHover(r)}
                  onMouseLeave={() => setHover(null)}
                />
              </div>
            </div>
          )
        })}
      </div>
      {hover && (
        <div className="wf-detail">
          <div className="wf-detail-head">{hover.name}</div>
          <div className="meta">span_id: {hover.span_id}</div>
          {hover.parent_id && <div className="meta">parent: {hover.parent_id}</div>}
          <div className="meta">
            start {hover.start_ms}ms · dur {hover.dur_ms}ms
          </div>
          <div className="wf-attrs">
            {Object.entries(hover.attrs).map(([k, v]) => (
              <div key={k} className="attr-row">
                <span className="attr-k">{k}</span>
                <span className="attr-v">{String(v)}</span>
              </div>
            ))}
            {Object.keys(hover.attrs).length === 0 && <div className="meta">无附加属性</div>}
          </div>
        </div>
      )}
    </div>
  )
}
