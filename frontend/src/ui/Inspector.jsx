import { useEffect, useRef, useState } from 'react'
import { STAGE_META, STAGE_ORDER } from '../state.js'
import { Icon, LEVEL, Spinner, pct, secs } from './bits.jsx'

export default function Inspector({ run, activeRef, setActiveRef, open, onClose }) {
  return (
    <aside className={`inspector ${open ? 'is-open' : ''}`} aria-label="Evidence and confidence">
      <div className="inspector-head only-narrow">
        <b>Evidence</b>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" /></button>
      </div>
      <ConfidenceCard run={run} />
      <PipelineCard run={run} />
      <SourcesCard run={run} activeRef={activeRef} setActiveRef={setActiveRef} />
      <CallsCard run={run} />
    </aside>
  )
}

// ── Confidence gauge ──────────────────────────────────────────────────────
function Gauge({ value, tone }) {
  const r = 52
  const c = Math.PI * r
  const v = value == null ? 0 : Math.max(0, Math.min(1, value))
  return (
    <svg className="gauge" viewBox="0 0 128 72" role="img" aria-label={value == null ? 'No score' : `Confidence ${Math.round(v * 100)} percent`}>
      <path d="M12 66 A52 52 0 0 1 116 66" className="gauge-track" />
      <path d="M12 66 A52 52 0 0 1 116 66" className={`gauge-fill tone-${tone}`}
        strokeDasharray={c} strokeDashoffset={c * (1 - v)} />
      {[0.6, 0.8].map((t) => {
        const a = Math.PI * (1 - t)
        return <line key={t} className="gauge-tick" x1={64 + 44 * Math.cos(a)} y1={66 - 44 * Math.sin(a)}
          x2={64 + 58 * Math.cos(a)} y2={66 - 58 * Math.sin(a)} />
      })}
    </svg>
  )
}

function ConfidenceCard({ run }) {
  const c = run.confidence
  const running = run.status === 'running'
  if (!c) {
    return (
      <section className="card conf-card">
        <div className="eyebrow">Confidence</div>
        <div className="conf-empty">
          <Gauge value={null} tone="neutral" />
          <p className="muted small">{running ? 'Calculated once the audit finishes…' : 'Ask a question to see how well the answer is supported.'}</p>
        </div>
      </section>
    )
  }
  const lv = LEVEL[c.level] || LEVEL.LOW
  const noScore = c.level === 'ABSTAINED' || c.level === 'NO_EVIDENCE'
  return (
    <section className="card conf-card">
      <div className="eyebrow">Confidence</div>
      <div className="conf-top">
        <div className="gauge-wrap">
          <Gauge value={noScore ? null : c.score} tone={lv.tone} />
          <div className={`gauge-num tone-${lv.tone}`}>{noScore ? '—' : Math.round(c.score * 100)}<small>{noScore ? '' : '%'}</small></div>
        </div>
        <div>
          <div className={`conf-level tone-${lv.tone}`}>{lv.label}</div>
          <div className="conf-headline">{c.headline}</div>
        </div>
      </div>
      <p className="conf-expl">{c.explanation}</p>
      <ul className="components">
        {c.components.map((k) => (
          <li key={k.key} className={k.value == null ? 'absent' : ''} title={k.description}>
            <span className="comp-label">{k.label}</span>
            <span className="bar"><span className={barTone(k.value)} style={{ width: pct(k.value ?? 0) }} /></span>
            <span className="comp-val mono">{k.value == null ? 'n/a' : pct(k.value)}</span>
            <span className="comp-w mono">{k.value == null ? '' : `×${k.weight.toFixed(2)}`}</span>
          </li>
        ))}
      </ul>
      {c.caps?.length > 0 && (
        <div className="caps">{c.caps.map((cap) => <div key={cap}><Icon name="alert" size={13} /> {cap}</div>)}</div>
      )}
    </section>
  )
}

const barTone = (v) => (v == null ? '' : v >= 0.8 ? 'tone-ok' : v >= 0.6 ? 'tone-warn' : 'tone-bad')

// ── Pipeline trace ────────────────────────────────────────────────────────
function Elapsed({ since }) {
  const [, t] = useState(0)
  useEffect(() => { const i = setInterval(() => t((n) => n + 1), 250); return () => clearInterval(i) }, [])
  return <span className="mono">{secs(Date.now() - since)}</span>
}

function PipelineCard({ run }) {
  if (run.status === 'idle') return null
  return (
    <section className="card">
      <header className="card-head"><h3>Pipeline</h3>
        {run.status === 'running' && run.startedAt && <span className="muted small"><Elapsed since={run.startedAt} /></span>}
        {run.status !== 'running' && run.durationMs != null && <span className="muted small mono">{secs(run.durationMs)}</span>}
      </header>
      <ol className="trace">
        {STAGE_ORDER.map((id) => {
          const s = run.stages[id]
          return (
            <li key={id} className={`trace-step st-${s.status}`}>
              <span className="trace-mark" aria-hidden="true">
                {s.status === 'running' ? <Spinner size={12} /> : s.status === 'done' ? <Icon name="check" size={12} />
                  : s.status === 'error' ? <Icon name="x" size={12} /> : s.status === 'skipped' ? <Icon name="dash" size={12} /> : null}
              </span>
              <div className="trace-body">
                <div className="trace-name">{STAGE_META[id].name}
                  <span className="trace-time mono">
                    {s.status === 'running' && s.startedAt ? <Elapsed since={s.startedAt} /> : s.duration_ms ? secs(s.duration_ms) : ''}
                  </span>
                </div>
                <div className="trace-sum">{s.summary || STAGE_META[id].blurb}</div>
              </div>
            </li>
          )
        })}
      </ol>
      {run.notices.length > 0 && (
        <details className="notices-log">
          <summary>{run.notices.length} provider notice{run.notices.length !== 1 ? 's' : ''}</summary>
          <ul>{run.notices.map((n, i) => <li key={i} className={`tone-${n.level === 'error' ? 'bad' : n.level === 'failover' ? 'accent' : 'warn'}`}>{n.message}</li>)}</ul>
        </details>
      )}
    </section>
  )
}

// ── Sources ───────────────────────────────────────────────────────────────
function SourcesCard({ run, activeRef, setActiveRef }) {
  const refs = useRef({})
  const [expanded, setExpanded] = useState(null)
  const cited = new Set((run.verification?.claims || []).flatMap((c) => c.cited || []))
  for (const m of (run.answer || '').matchAll(/\[(\d+)\]/g)) cited.add(parseInt(m[1], 10))

  useEffect(() => {
    if (activeRef?.scroll && refs.current[activeRef.n]) {
      refs.current[activeRef.n].scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      setExpanded(activeRef.n)
    }
  }, [activeRef])

  if (!run.sources.length) {
    if (run.stages.retrieve.status === 'running') return <section className="card"><div className="loading-row"><Spinner /> Searching your library…</div></section>
    return null
  }
  return (
    <section className="card">
      <header className="card-head"><h3>Sources</h3><span className="muted small">{run.sources.length} retrieved · {[...cited].filter((n) => n <= run.sources.length).length} cited</span></header>
      <ol className="sources">
        {run.sources.map((s) => {
          const on = activeRef?.n === s.ref
          const open = expanded === s.ref
          return (
            <li key={s.ref} ref={(el) => { refs.current[s.ref] = el }}
              className={`source ${on ? 'on' : ''} ${run.status === 'done' && !cited.has(s.ref) ? 'uncited' : ''}`}
              onMouseEnter={() => setActiveRef(s.ref)} onMouseLeave={() => setActiveRef(null)}>
              <button className="source-head" onClick={() => setExpanded(open ? null : s.ref)} aria-expanded={open}>
                <span className="src-num">{s.ref}</span>
                <span className="src-label" title={s.label}>{s.label}</span>
                <span className="rel" title="Relevance to the question">
                  <span className="bar mini"><span className={barTone(s.relevance)} style={{ width: pct(s.relevance) }} /></span>
                  <span className="mono">{pct(s.relevance)}</span>
                </span>
              </button>
              <p className={`src-text ${open ? 'open' : ''}`}>{s.text}</p>
              {run.status === 'done' && !cited.has(s.ref) && <div className="src-flag">not cited in the answer</div>}
            </li>
          )
        })}
      </ol>
    </section>
  )
}

// ── Model calls ───────────────────────────────────────────────────────────
function CallsCard({ run }) {
  if (!run.calls.length) return null
  const ok = run.calls.filter((c) => c.ok)
  const tokens = ok.reduce((a, c) => a + (c.input_tokens || 0) + (c.output_tokens || 0), 0)
  return (
    <section className="card">
      <details>
        <summary className="card-head as-summary"><h3>Model calls</h3>
          <span className="muted small">{ok.length} ok{run.calls.length - ok.length ? ` · ${run.calls.length - ok.length} retried` : ''}{tokens ? ` · ${tokens.toLocaleString()} tokens` : ''}</span></summary>
        <table className="calls">
          <thead><tr><th>Step</th><th>Provider / model</th><th className="num">Time</th></tr></thead>
          <tbody>
            {run.calls.map((c, i) => (
              <tr key={i} className={c.ok ? '' : 'failed'}>
                <td>{c.purpose}</td>
                <td><span className="mono">{c.provider}/{c.model}</span>{!c.ok && <div className="call-err">{c.error}: {c.message}</div>}</td>
                <td className="num mono">{c.ok ? secs(c.latency_ms) : '✕'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  )
}
