import { useEffect, useState } from 'react'
import { Loader2, Trash2 } from 'lucide-react'
import { deleteDocument, listDocuments } from '../api/client'
import type { DocumentItem } from '../api/types'
import { formatTime } from '../lib/format'

export default function DocumentList({ refreshKey }: { refreshKey: number }) {
  const [items, setItems] = useState<DocumentItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [delId, setDelId] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    listDocuments()
      .then((d) => {
        if (!alive) return
        setItems(d.items)
        setTotal(d.total)
        setError(null)
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : '加载失败')
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [refreshKey])

  const remove = async (docId: string) => {
    if (delId) return
    setDelId(docId)
    try {
      await deleteDocument(docId)
      setItems((prev) => prev.filter((x) => x.doc_id !== docId))
      setTotal((t) => t - 1)
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    } finally {
      setDelId(null)
    }
  }

  return (
    <div className="card">
      <div className="card-head">
        <span>文档列表</span>
        <span className="meta">{total} 篇</span>
      </div>
      {loading && (
        <div className="meta row">
          <Loader2 size={16} className="spin" /> 加载中…
        </div>
      )}
      {error && <div className="error-box">{error}</div>}
      {!loading && !error && items.length === 0 && <div className="empty">暂无文档，先在左侧上传。</div>}
      <ul className="doc-list">
        {items.map((d) => (
          <li key={d.doc_id} className="doc-item">
            <div className="doc-main">
              <div className="doc-title">{d.title}</div>
              <div className="meta">
                {formatTime(d.created_at)} · {d.n_chunks} 分片
              </div>
            </div>
            <button
              className="icon-btn danger"
              onClick={() => void remove(d.doc_id)}
              disabled={delId === d.doc_id}
              title="删除文档"
            >
              {delId === d.doc_id ? <Loader2 size={16} className="spin" /> : <Trash2 size={16} />}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
