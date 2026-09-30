/**
 * Workspace — right panel: query bar + results area.
 * Contains the question input, run button, and all result cards.
 */

import { useRef } from 'react'
import AnswerDisplay from './AnswerDisplay'
import ConfidenceGauge from './ConfidenceGauge'
import SourcesPanel from './SourcesPanel'

function EmptyState({ docCount }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">⬡</div>
      <div className="empty-title">Ask anything from your knowledge base</div>
      <div className="empty-sub">
        {docCount === 0
          ? 'Upload documents in the sidebar to get started'
          : `${docCount} document${docCount !== 1 ? 's' : ''} ready — ask a question above`}
      </div>
    </div>
  )
}

function ErrorCard({ message }) {
  return (
    <div className="rcard error-card">
      <div className="rcard-header">
        <span className="rcard-icon" style={{ color: '#f25c5c' }}>⚠</span>
        <span className="rcard-title">Error</span>
      </div>
      <div className="rcard-body">
        <p style={{ color: '#f25c5c', fontSize: 13 }}>{message}</p>
      </div>
    </div>
  )
}

function AnswerCard({ result }) {
  return (
    <div className="rcard answer-card">
      <div className="rcard-header">
        <span className="rcard-icon teal">◎</span>
        <span className="rcard-title">Answer</span>
        <span className="rcard-meta">{result.total_time}s · {result.kb_size} chunks in KB</span>
      </div>
      <AnswerDisplay
        answer={result.final_answer}
        sources={result.sources}
      />
    </div>
  )
}

function ConfidenceCard({ result }) {
  if (!result.verdict) return null
  return (
    <div className="rcard confidence-card">
      <div className="rcard-header">
        <span className="rcard-icon purple">◈</span>
        <span className="rcard-title">Confidence</span>
      </div>
      <ConfidenceGauge verdict={result.verdict} breakdown={result.breakdown || {}} />
    </div>
  )
}

function EvidenceCard({ result }) {
  if (!result.sources?.length) return null
  return (
    <div className="rcard evidence-card">
      <div className="rcard-header">
        <span className="rcard-icon amber">⬡</span>
        <span className="rcard-title">Evidence</span>
      </div>
      <SourcesPanel
        sources={result.sources}
        verdicts={result.verdicts || []}
        citedNums={result.cited_nums || []}
      />
    </div>
  )
}

export default function Workspace({ question, setQuestion, onRun, running, result, error, docCount }) {
  const textareaRef = useRef(null)

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onRun()
    }
  }

  return (
    <div className="workspace">
      {/* Top query bar */}
      <div className="query-bar">
        <div className="query-wrap">
          <textarea
            ref={textareaRef}
            className="query-input"
            placeholder="Ask a question about your documents…"
            rows={1}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
            disabled={running}
          />
          <button
            className={`run-btn ${running ? 'running' : ''}`}
            onClick={onRun}
            disabled={running || !question.trim()}
          >
            {running
              ? <span className="spinner-ring" style={{width:14,height:14,borderWidth:2}} />
              : '→'}
          </button>
        </div>
        <div className="query-hint">Enter to run · Shift+Enter for newline</div>
      </div>

      {/* Results area */}
      <div className="results-area">
        {!result && !error && !running && (
          <EmptyState docCount={docCount} />
        )}

        {error && <ErrorCard message={error} />}

        {result && (
          <div className="result-grid">
            <AnswerCard result={result} />
            <ConfidenceCard result={result} />
            <EvidenceCard result={result} />
          </div>
        )}
      </div>
    </div>
  )
}
