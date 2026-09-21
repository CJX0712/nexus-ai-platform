import { useState } from 'react'
import { ChevronRight, Quote } from 'lucide-react'
import type { Citation } from '../api/types'
import { formatScore, shortId } from '../lib/format'

export default function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState<Record<number, boolean>>({})

  return (
    <div className="citations">
      <div className="citations-head">
        <Quote size={16} /> 引用来源（{citations.length}）
      </div>
      <ul className="cite-list">
        {citations.map((c, i) => (
          <li key={i} className="cite-item">
            <button className="cite-row" onClick={() => setOpen((o) => ({ ...o, [i]: !o[i] }))}>
              <ChevronRight size={16} className={'chev' + (open[i] ? ' open' : '')} />
              <span className="cite-doc">{shortId(c.doc_id)}</span>
              <span className="cite-score">融合 {formatScore(c.score)}</span>
            </button>
            {open[i] && <pre className="cite-text">{c.text}</pre>}
          </li>
        ))}
      </ul>
    </div>
  )
}
