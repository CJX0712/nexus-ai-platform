import { useState } from 'react'
import IngestForm from './IngestForm'
import DocumentList from './DocumentList'
import SearchTest from './SearchTest'

export default function KnowledgePanel() {
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="panel">
      <header className="panel-head">
        <div>
          <h1>知识库</h1>
          <p className="panel-sub">文档入库、管理与混合检索测试</p>
        </div>
      </header>

      <div className="kb-grid">
        <div className="kb-col">
          <IngestForm onDone={() => setRefreshKey((k) => k + 1)} />
        </div>
        <div className="kb-col">
          <DocumentList refreshKey={refreshKey} />
        </div>
      </div>

      <SearchTest />
    </div>
  )
}
