/** AnswerPanel — shows cited answer, sources, claim verdicts, confidence */

const STATUS_COLOR = {
  SUPPORTED:   '#22c55e',
  PARTIAL:     '#f0a732',
  UNSUPPORTED: '#f25c5c',
  UNKNOWN:     '#5a718a',
}

const LEVEL_COLOR = {
  HIGH:   '#22c55e',
  MEDIUM: '#f0a732',
  LOW:    '#f25c5c',
}

function ConfidenceMeter({ score, level }) {
  const pct = Math.round((score || 0) * 100)
  const color = LEVEL_COLOR[level] || '#5a718a'
  return (
    <div className="conf-meter">
      <div className="conf-bar-bg">
        <div
          className="conf-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="conf-value" style={{ color }}>
        {pct}% <span className="conf-level">{level}</span>
      </span>
    </div>
  )
}

function SourceCard({ src }) {
  return (
    <div className="source-card">
      <div className="source-header">
        <span className="source-ref">[{src.ref}]</span>
        <span className="source-label">{src.label}</span>
        <span className="source-score">{Math.round(src.score * 100)}% match</span>
      </div>
      <p className="source-snippet">{src.snippet}</p>
    </div>
  )
}

function ClaimRow({ v }) {
  const color = STATUS_COLOR[v.status] || '#5a718a'
  return (
    <div className="claim-row">
      <span className="claim-status" style={{ color, borderColor: color }}>
        {v.status}
      </span>
      <div className="claim-text-wrap">
        <p className="claim-text">{v.claim}</p>
        {v.evidence && (
          <p className="claim-evidence">Evidence: "{v.evidence}"</p>
        )}
      </div>
    </div>
  )
}

export default function AnswerPanel({ result }) {
  const { final_answer, sources = [], verdicts = [], verdict = {}, breakdown = {}, total_time } = result

  return (
    <div className="answer-panel">

      {/* ── Answer ──────────────────────────────────────────── */}
      <section className="ans-section">
        <div className="section-label">Answer</div>
        <div className="answer-text">{final_answer}</div>
        {total_time && (
          <div className="answer-meta">{total_time}s total</div>
        )}
      </section>

      {/* ── Confidence ──────────────────────────────────────── */}
      {verdict.score !== undefined && (
        <section className="ans-section">
          <div className="section-label">Confidence</div>
          <ConfidenceMeter score={verdict.score} level={verdict.level} />
          {verdict.explanation && (
            <p className="conf-reason">{verdict.explanation}</p>
          )}
          {breakdown && Object.keys(breakdown).length > 0 && (
            <div className="breakdown-row">
              {breakdown.evidence_score !== undefined && (
                <span className="bd-chip">Evidence {Math.round(breakdown.evidence_score * 100)}%</span>
              )}
              {breakdown.consistency_score !== null && breakdown.consistency_score !== undefined && (
                <span className="bd-chip">Consistency {Math.round(breakdown.consistency_score * 100)}%</span>
              )}
              {breakdown.reflection_score !== null && breakdown.reflection_score !== undefined && (
                <span className="bd-chip">Reflection {Math.round(breakdown.reflection_score * 100)}%</span>
              )}
            </div>
          )}
        </section>
      )}

      {/* ── Sources ─────────────────────────────────────────── */}
      {sources.length > 0 && (
        <section className="ans-section">
          <div className="section-label">Sources ({sources.length})</div>
          <div className="sources-list">
            {sources.map(src => <SourceCard key={src.ref} src={src} />)}
          </div>
        </section>
      )}

      {/* ── Claim Verification ──────────────────────────────── */}
      {verdicts.length > 0 && (
        <section className="ans-section">
          <div className="section-label">Claim Verification</div>
          <div className="verdicts-list">
            {verdicts.map((v, i) => <ClaimRow key={i} v={v} />)}
          </div>
        </section>
      )}
    </div>
  )
}
