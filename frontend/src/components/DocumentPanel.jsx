/** DocumentPanel — upload files, list documents, delete */

import { useState, useEffect, useCallback } from 'react'

const ACCEPT = '.pdf,.md,.html,.htm,.txt'

export default function DocumentPanel({ apiBase }) {
  const [docs, setDocs]         = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState(null)
  const [dragOver, setDragOver]  = useState(false)

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/documents`)
      const data = await res.json()
      setDocs(data.documents || [])
    } catch {}
  }, [apiBase])

  useEffect(() => { fetchDocs() }, [fetchDocs])

  const uploadFile = async (file) => {
    if (!file) return
    setUploading(true)
    setUploadMsg(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${apiBase}/api/documents/ingest`, { method: 'POST', body: form })
      const data = await res.json()
      if (res.ok) {
        setUploadMsg({ type: 'ok', text: `'${data.filename}' — ${data.chunk_count} chunks indexed` })
        fetchDocs()
      } else {
        setUploadMsg({ type: 'err', text: data.detail || 'Upload failed' })
      }
    } catch (e) {
      setUploadMsg({ type: 'err', text: e.message })
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  const deleteDoc = async (docId, filename) => {
    if (!confirm(`Remove '${filename}' from the knowledge base?`)) return
    try {
      const res = await fetch(`${apiBase}/api/documents/${docId}`, { method: 'DELETE' })
      if (res.ok) fetchDocs()
    } catch {}
  }

  return (
    <div className="doc-panel">
      <h2 className="doc-title">Knowledge Base</h2>
      <p className="doc-sub">Upload PDF, Markdown, HTML, or TXT files. Answers will cite these sources.</p>

      {/* ── Drop zone ─────────────────────────────────────── */}
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept={ACCEPT}
          hidden
          onChange={handleFileInput}
        />
        {uploading ? (
          <><span className="spinner" /> Ingesting…</>
        ) : (
          <>
            <span className="drop-icon">⬆</span>
            <span className="drop-text">Drop a file here or <u>click to browse</u></span>
            <span className="drop-hint">PDF · MD · HTML · TXT</span>
          </>
        )}
      </div>

      {uploadMsg && (
        <div className={`upload-msg ${uploadMsg.type}`}>
          {uploadMsg.type === 'ok' ? '✓' : '⚠'} {uploadMsg.text}
        </div>
      )}

      {/* ── Document list ──────────────────────────────────── */}
      {docs.length === 0 ? (
        <div className="doc-empty">No documents yet — upload one to get started.</div>
      ) : (
        <div className="doc-list">
          {docs.map(doc => (
            <div key={doc.doc_id} className="doc-row">
              <div className="doc-info">
                <span className="doc-name">{doc.filename}</span>
                <span className="doc-meta">
                  {doc.file_type.toUpperCase()} · {doc.page_count} page{doc.page_count !== 1 ? 's' : ''} · {doc.chunk_count} chunks
                </span>
                <span className="doc-date">{new Date(doc.created_at).toLocaleString()}</span>
              </div>
              <button
                className="doc-delete"
                onClick={() => deleteDoc(doc.doc_id, doc.filename)}
                title="Remove from knowledge base"
              >✕</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
