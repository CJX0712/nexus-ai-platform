import { useState } from 'react'
import { ChevronRight, Wrench } from 'lucide-react'
import type { ToolCall } from '../api/types'

export default function ToolCalls({ tools }: { tools: ToolCall[] }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="tools">
      <button className="tools-head" onClick={() => setOpen((o) => !o)}>
        <ChevronRight size={16} className={'chev' + (open ? ' open' : '')} />
        <Wrench size={16} /> 工具调用（{tools.length}）
      </button>
      {open && (
        <ul className="tool-list">
          {tools.map((t, i) => (
            <li key={i} className="tool-item">
              <div className={'tool-status' + (t.ok ? ' ok' : ' bad')}>{t.ok ? '成功' : '失败'}</div>
              <div className="tool-body">
                <div className="tool-name">{t.name}</div>
                <div className="tool-args">{JSON.stringify(t.args)}</div>
                <pre className="tool-result">{t.result}</pre>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
