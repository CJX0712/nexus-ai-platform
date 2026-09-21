import { useState } from 'react'
import type { FormEvent } from 'react'
import { Loader2, Upload } from 'lucide-react'
import { ingest } from '../api/client'
import type { IngestResult } from '../api/types'

export default function IngestForm({ onDone }: { onDone: () => void }) {
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !text.trim() || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await ingest({ title: title.trim(), text: text.trim() })
      setResult(r)
      setTitle('')
      setText('')
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : '入库失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="card" onSubmit={submit}>
      <div className="card-head">
        <Upload size={20} />
        <span>上传文档</span>
      </div>
      <label className="field">
        <span>标题</span>
        <input
          className="input"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="例如：2024 年产品白皮书"
        />
      </label>
      <label className="field">
        <span>正文</span>
        <textarea
          className="input textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="粘贴文档正文内容，系统会自动切片与向量化…"
          rows={8}
        />
      </label>
      <button className="btn primary" type="submit" disabled={busy || !title.trim() || !text.trim()}>
        {busy ? <Loader2 size={20} className="spin" /> : <Upload size={20} />}
        <span>入库</span>
      </button>
      {result && (
        <div className="ok-box">
          已入库 {result.doc_id.slice(0, 8)} · {result.n_chunks} 分片 · {result.n_chars} 字符
        </div>
      )}
      {error && <div className="error-box">{error}</div>}
    </form>
  )
}
