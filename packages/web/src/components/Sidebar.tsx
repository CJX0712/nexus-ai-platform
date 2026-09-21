import { Activity, Database, MessageSquare } from 'lucide-react'
import type { Health } from '../api/types'
import type { Panel } from '../App'
import { formatUptime } from '../lib/format'

const NAV: { key: Panel; label: string; icon: typeof MessageSquare; desc: string }[] = [
  { key: 'chat', label: '对话', icon: MessageSquare, desc: '检索增强对话' },
  { key: 'knowledge', label: '知识库', icon: Database, desc: '文档入库与检索' },
  { key: 'trace', label: 'Trace', icon: Activity, desc: '请求链路追踪' },
]

export default function Sidebar({
  panel,
  onSelect,
  health,
}: {
  panel: Panel
  onSelect: (p: Panel) => void
  health: Health | null
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">N</span>
        <div className="brand-text">
          <strong>nexus</strong>
          <span>AI 控制台</span>
        </div>
      </div>
      <nav className="nav">
        {NAV.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.key}
              className={'nav-item' + (panel === item.key ? ' active' : '')}
              onClick={() => onSelect(item.key)}
            >
              <Icon size={24} />
              <span className="nav-label">{item.label}</span>
              <span className="nav-desc">{item.desc}</span>
            </button>
          )
        })}
      </nav>
      <div className="sidebar-foot">
        {health ? (
          <>
            <div className={'status' + (health.status === 'ok' ? ' ok' : ' bad')}>
              <span className="dot" />
              {health.status === 'ok' ? '服务正常' : '服务异常'}
            </div>
            <div className="meta">provider: {health.provider}</div>
            <div className="meta">
              v{health.version} · 运行 {formatUptime(health.uptime_s)}
            </div>
          </>
        ) : (
          <div className="meta">连接后端中…</div>
        )}
      </div>
    </aside>
  )
}
