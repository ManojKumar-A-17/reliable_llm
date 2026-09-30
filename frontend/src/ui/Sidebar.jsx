import { useRef, useState } from 'react'
import { Icon, LEVEL, Spinner, ago, bytes } from './bits.jsx'

const TYPE_LABEL = { pdf: 'PDF', docx: 'DOCX', md: 'MD', txt: 'TXT', html: 'HTML' }

export default function Sidebar({ health, docs, onUpload, uploading, onDeleteDoc, runs, activeRunId,
  onOpenRun, onDeleteRun, onNew, open, onClose }) {
  const [tab, setTab] = useState('library')
  const accept = (health?.supported_types || ['.pdf', '.docx', '.md', '.txt', '.html']).join(',')

  return (
    <aside className={`sidebar ${open ? 'is-open' : ''}`} aria-label="Library and history">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true"><Icon name="check" size={14} /></div>
        <div>
          <div className="brand-name">Reliable LLM</div>
          <div className="brand-sub">Evidence-first answers</div>
        </div>
        <button className="icon-btn only-mobile" onClick={onClose} aria-label="Close panel"><Icon name="x" /></button>
      </div>

      <button className="btn btn-quiet new-btn" onClick={onNew}>
        <Icon name="spark" /> New question
      </button>

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'library'} className={tab === 'library' ? 'on' : ''}
          onClick={() => setTab('library')}>Library <span className="count">{docs.length}</span></button>
        <button role="tab" aria-selected={tab === 'history'} className={tab === 'history' ? 'on' : ''}
          onClick={() => setTab('history')}>History <span className="count">{runs.length}</span></button>
      </div>

      {tab === 'library' ? (
        <Library docs={docs} accept={accept} maxMb={health?.max_upload_mb || 25}
          onUpload={onUpload} uploading={uploading} onDelete={onDeleteDoc} />
      ) : (
        <History runs={runs} activeId={activeRunId} onOpen={onOpenRun} onDelete={onDeleteRun} />
      )}

      <div className="sidebar-foot">
        {health?.library ? (
          <span>{health.library.chunks.toLocaleString()} passages · {health.library.words.toLocaleString()} words indexed</span>
        ) : <span>—</span>}
      </div>
    </aside>
  )
}

function Library({ docs, accept, maxMb, onUpload, uploading, onDelete }) {
  const input = useRef(null)
  const [drag, setDrag] = useState(false)
  const [confirmId, setConfirmId] = useState(null)

  const handleFiles = (list) => {
    const files = Array.from(list || [])
    if (files.length) onUpload(files)
  }

  return (
    <div className="panel-scroll">
      <div
        className={`dropzone ${drag ? 'drag' : ''} ${uploading ? 'busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files) }}
        onClick={() => !uploading && input.current?.click()}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current?.click() } }}
        role="button" tabIndex={0} aria-label="Upload documents"
      >
        {uploading ? <Spinner size={18} /> : <Icon name="upload" size={20} />}
        <div className="dz-title">{uploading ? 'Reading & indexing…' : 'Add documents'}</div>
        <div className="dz-sub">PDF, DOCX, Markdown, HTML, TXT · up to {maxMb} MB</div>
        <input ref={input} type="file" multiple accept={accept} hidden
          onChange={(e) => { handleFiles(e.target.files); e.target.value = '' }} />
      </div>

      {docs.length === 0 ? (
        <p className="muted small pad">Your library is empty. Answers are only drawn from documents you add here.</p>
      ) : (
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.id} className="doc-item">
              <span className={`doc-type t-${d.type}`}>{TYPE_LABEL[d.type] || d.type.toUpperCase()}</span>
              <div className="doc-main">
                <div className="doc-name" title={d.name}>{d.name}</div>
                <div className="doc-meta">
                  {d.pages ? `${d.pages} pages · ` : ''}{d.chunks} passages · {bytes(d.size_bytes)}
                </div>
              </div>
              {confirmId === d.id ? (
                <div className="confirm">
                  <button className="btn btn-danger btn-xs" onClick={() => { onDelete(d.id); setConfirmId(null) }}>Remove</button>
                  <button className="btn btn-quiet btn-xs" onClick={() => setConfirmId(null)}>Keep</button>
                </div>
              ) : (
                <button className="icon-btn hover-reveal" onClick={() => setConfirmId(d.id)}
                  aria-label={`Remove ${d.name}`} title="Remove from library"><Icon name="trash" /></button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function History({ runs, activeId, onOpen, onDelete }) {
  if (!runs.length) {
    return <div className="panel-scroll"><p className="muted small pad">Questions you ask are saved here with their evidence and scores.</p></div>
  }
  return (
    <div className="panel-scroll">
      <ul className="run-list">
        {runs.map((r) => {
          const lv = LEVEL[r.level] || { tone: 'neutral', label: r.level || '—' }
          return (
            <li key={r.id} className={`run-item ${r.id === activeId ? 'active' : ''}`}>
              <button className="run-open" onClick={() => onOpen(r.id)}>
                <span className={`dot tone-${lv.tone}`} aria-hidden="true" />
                <span className="run-q">{r.question}</span>
                <span className="run-meta">
                  {lv.label}{r.score != null && r.level !== 'ABSTAINED' ? ` · ${Math.round(r.score * 100)}%` : ''} · {r.mode} · {ago(r.created_at)}
                </span>
              </button>
              <button className="icon-btn hover-reveal" onClick={() => onDelete(r.id)} aria-label="Delete from history">
                <Icon name="trash" />
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
