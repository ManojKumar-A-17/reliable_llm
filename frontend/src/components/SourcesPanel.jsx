/**
 * SourcesPanel — expandable accordion source cards showing retrieved evidence.
 */

import { useState } from 'react'

const FILE_EXT_COLOR = { pdf: '#ef4444', md: '#3b82f6', html: '#f59e0b', htm: '#f59e0b', txt: '#6b7280' }

function VerdictChip({ verdict }) {
  const map = {
    SUPPORTED:   { label: 'Supported',   cls: 'vchip-sup' },
    PARTIAL:     { label: 'Partial',      cls: 'vchip-par' },
    UNSUPPORTED: { label: 'Unsupported',  cls: 'vchip-uns' },
    UNKNOWN:     { label: 'Unknown',      cls: 'vchip-unk' },
  }
  const v = map[verdict?.toUpperCase()] || map['UNKNOWN']
  return <span className={`verdict-chip ${v.cls}`}>{v.label}</span>
}

function ClaimRow({ claim }) {
  return (
    <div className="claim-row">
      <VerdictChip verdict={claim.verdict} />
      <span className="claim-text">{claim.claim}</span>
    </div>
  )
}

function SourceCard({ source, index, cited }) {
  const [expanded, setExpanded] = useState(false)
  const ext = source.source?.split('.').pop()?.toLowerCase() || 'txt'
  const color = FILE_EXT_COLOR[ext] || '#6b7280'

  return (
    <div className={`source-card ${cited ? 'cited' : ''} ${expanded ? 'expanded' : ''}`}>
      <button
        className="source-card-header"
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
      >
        <span className="src-num" style={{ borderColor: color, color }}>{index}</span>
        <span className="src-title">{source.source || `Source ${index}`}</span>
        <span className="src-chevron">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && (
        <div className="source-card-body">
          <p className="source-text">{source.text || '(no text)'}</p>
          {source.page_num != null && (
            <div className="source-meta">Page {source.page_num}</div>
          )}
        </div>
      )}
    </div>
  )
}

export default function SourcesPanel({ sources = [], verdicts = [], citedNums = [] }) {
  if (!sources.length) return null

  return (
    <div className="sources-panel">
      {/* Claims / Verdicts */}
      {verdicts.length > 0 && (
        <div className="claims-section">
          <div className="panel-label">Claim Verification</div>
          <div className="claims-list">
            {verdicts.map((v, i) => <ClaimRow key={i} claim={v} />)}
          </div>
        </div>
      )}

      {/* Retrieved sources */}
      <div className="panel-label" style={{ marginTop: verdicts.length ? 16 : 0 }}>
        Retrieved Evidence
        <span className="panel-count">{sources.length}</span>
      </div>
      <div className="sources-list">
        {sources.map((src, i) => {
          const srcObj = typeof src === 'string'
            ? { source: src.match(/\[Source:\s*(.+?)\]/)?.[1] || `Doc ${i+1}`, text: src }
            : src
          return (
            <SourceCard
              key={i}
              index={i + 1}
              source={srcObj}
              cited={citedNums.includes(i + 1)}
            />
          )
        })}
      </div>
    </div>
  )
}
