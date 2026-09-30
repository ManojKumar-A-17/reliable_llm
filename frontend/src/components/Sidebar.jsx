import { useState, useCallback } from 'react'

const FILE_ICONS = { pdf: 'PDF', md: 'MD', html: 'HTM', htm: 'HTM', txt: 'TXT' }

const STEP_ICONS = ['⬡','✎','⬡','↺','◎','◈']

function DocCard({ doc, onDelete }) {
  const ext = doc.file_type || doc.filename?.split('.').pop()?.toLowerCase() || 'txt'
  const badge = FILE_ICONS[ext] || ext.toUpperCase().slice(0, 3)
  return (
    <div className="doc-card">
      <div className={`doc-type-badge ${ext}`}>{badge}</div>
      <div className="doc-meta">
        <div className="doc-name" title={doc.filename}>{doc.filename}</div>
        <div className="doc-stats">{doc.chunk_count} chunks · {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}</div>
      </div>
      <button className="doc-del" onClick={() => onDelete(doc.doc_id, doc.filename)} title="Remove">✕</button>
    </div>
  )
}

function PipelineStep({ step }) {
  const idx = step.id - 1
  const icon = step.status === 'running' ? <span className="spinner-ring" />
             : step.status === 'done'    ? '✓'
             : step.status === 'skip'   ? '–'
             : String(step.id)

  return (
    <div className={`ps-row ${step.status}`}>
      <div className="ps-indicator">{icon}</div>
      <span className="ps-name">{step.name}</span>
      {step.detail && <span className="ps-detail" title={step.detail}>{step.detail}</span>}
    </div>
  )
}

export default function Sidebar({ apiBase, docs, setDocs, steps, running }) {
  const [dragOver,  setDragOver]  = useState(false)
  const [uploading, setUploading] = useState(false)
  const [toast,     setToast]     = useState(null)

  const showToast = (type, text) => {
    setToast({ type, text })
    setTimeout(() => setToast(null), 5000)
  }

  const fetchDocs = useCallback(async () => {
    try {
      const res  = await fetch(`${apiBase}/api/documents`)
      const data = await res.json()
      setDocs(data.documents || [])
    } catch {}
  }, [apiBase, setDocs])

  // Fetch on mount
  useState(() => { fetchDocs() }, [])

  const upload = async (file) => {
    if (!file) return
    setUploading(true)
    setToast(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res  = await fetch(`${apiBase}/api/documents/ingest`, { method: 'POST', body: form })
      const data = await res.json()
      if (res.ok) {
        showToast('success', `Indexed ${data.chunk_count} chunks from "${data.filename}"`)
        fetchDocs()
      } else {
        showToast('error', data.detail || 'Upload failed')
      }
    } catch (e) {
      showToast('error', e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    upload(e.dataTransfer.files[0])
  }

  const handleInput = (e) => {
    upload(e.target.files[0])
    e.target.value = ''
  }

  const handleDelete = async (docId, filename) => {
    if (!confirm(`Remove "${filename}"?`)) return
    try {
      await fetch(`${apiBase}/api/documents/${docId}`, { method: 'DELETE' })
      fetchDocs()
    } catch {}
  }

  const anyActive = steps.some(s => s.status !== 'idle')

  return (
    <div className="sidebar">
      {/* Brand */}
      <div className="brand">
        <div className="brand-mark">R</div>
        <div className="brand-text">
          <span className="brand-name">ReliableLLM</span>
          <span className="brand-tag">Evidence-first RAG</span>
        </div>
      </div>

      {/* Document library */}
      <div className="sb-section scrollable">
        <div className="sb-label">Knowledge Base</div>

        {/* Upload zone */}
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'busy' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('sb-file-input').click()}
        >
          <input id="sb-file-input" type="file" accept=".pdf,.md,.html,.htm,.txt" hidden onChange={handleInput} />
          <div className="upload-icon">
            {uploading ? <span className="spinner-ring" style={{width:16,height:16}} /> : '↑'}
          </div>
          <div className="upload-title">{uploading ? 'Ingesting…' : 'Drop a file or click'}</div>
          <div className="upload-sub">PDF · MD · HTML · TXT</div>
        </div>

        {toast && (
          <div className={`upload-toast ${toast.type}`}>
            <span>{toast.type === 'success' ? '✓' : '⚠'}</span>
            <span>{toast.text}</span>
          </div>
        )}

        {/* Doc list */}
        {docs.length === 0 ? (
          <div className="doc-empty">No documents yet.<br />Upload a file to build your knowledge base.</div>
        ) : (
          <div className="doc-list">
            {docs.map(doc => (
              <DocCard key={doc.doc_id} doc={doc} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </div>

      {/* Pipeline status */}
      {anyActive && (
        <>
          <div className="sb-divider" />
          <div className="pipeline-section">
            <div className="sb-label" style={{padding:'0 4px 8px'}}>Pipeline</div>
            <div className="pipeline-steps">
              {steps.map(s => <PipelineStep key={s.id} step={s} />)}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
