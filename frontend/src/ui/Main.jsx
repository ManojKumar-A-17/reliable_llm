import { useEffect, useRef, useState } from 'react'
import { Icon, LEVEL, Markdown, Spinner, VERDICT, pct, secs } from './bits.jsx'

// ── Composer ──────────────────────────────────────────────────────────────
export function Composer({ value, onChange, onSubmit, onStop, running, mode, setMode, disabled, disabledReason }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 220) + 'px'
  }, [value])

  const submit = () => { if (!running && value.trim() && !disabled) onSubmit() }
  return (
    <div className={`composer ${running ? 'is-running' : ''}`}>
      <textarea
        ref={ref} rows={1} value={value} placeholder="Ask a question about your documents…"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
        aria-label="Question"
      />
      <div className="composer-bar">
        <div className="segmented" role="radiogroup" aria-label="Verification depth">
          {[['thorough', 'Thorough', 'Answer + claim audit + re-samples + review (≈5 model calls)'],
            ['fast', 'Fast', 'Answer + claim audit only (2 model calls)']].map(([k, label, tip]) => (
            <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? 'on' : ''}
              onClick={() => setMode(k)} title={tip} disabled={running}>{label}</button>
          ))}
        </div>
        <span className="composer-hint">{disabled ? disabledReason : 'Enter to ask · Shift+Enter for a new line'}</span>
        {running ? (
          <button className="btn btn-outline" onClick={onStop}><Icon name="stop" /> Stop</button>
        ) : (
          <button className="btn btn-primary" onClick={submit} disabled={!value.trim() || disabled}>
            Ask <Icon name="send" />
          </button>
        )}
      </div>
    </div>
  )
}

// ── Status bar ────────────────────────────────────────────────────────────
export function StatusBar({ health, healthError, theme, onToggleTheme, onMenu, onInspector }) {
  const providers = health?.providers || []
  return (
    <header className="statusbar">
      <button className="icon-btn only-mobile" onClick={onMenu} aria-label="Open library"><Icon name="menu" /></button>
      <div className="status-items">
        <span className={`status-pill ${healthError ? 'tone-bad' : health ? 'tone-ok' : 'tone-neutral'}`}>
          <span className="dot" />{healthError ? 'Backend offline' : health ? 'Connected' : 'Connecting…'}
        </span>
        {health && !health.models_ready && <span className="status-pill tone-neutral"><Spinner size={10} /> Loading local models</span>}
        {health && (providers.length ? providers.map((p, i) => (
          <span key={p.name} className="status-pill tone-neutral" title={`Models: ${Object.values(p.models).join(', ')}`}>
            {i === 0 ? 'Primary' : 'Fallback'}: <b>{p.label}</b> <span className="mono">{p.models.strong}</span>
          </span>
        )) : <span className="status-pill tone-bad"><span className="dot" />No LLM key configured</span>)}
      </div>
      <div className="status-actions">
        <button className="icon-btn only-narrow" onClick={onInspector} aria-label="Show evidence panel" title="Evidence panel"><Icon name="link" /></button>
        <button className="icon-btn" onClick={onToggleTheme} aria-label="Toggle colour theme" title="Toggle theme">
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
        </button>
      </div>
    </header>
  )
}

// ── Onboarding / empty state ──────────────────────────────────────────────
export function Welcome({ health, docCount, onPick }) {
  const hasKey = !!health?.llm_ready
  const hasDocs = docCount > 0
  const examples = hasDocs
    ? ['Summarise the main findings, with citations.', 'What evaluation metrics are defined?',
       'What limitations or open questions are mentioned?']
    : []
  return (
    <div className="welcome">
      <h1>Answers you can check.</h1>
      <p className="lede">Every answer is written only from your documents, each claim is audited against its source,
        and you get a calibrated confidence score — or an honest “not in your documents”.</p>
      <ol className="checklist">
        <li className={hasKey ? 'done' : ''}>
          <span className="step-mark">{hasKey ? <Icon name="check" size={14} /> : '1'}</span>
          <div><b>Connect a model</b>
            <span>{hasKey ? `Using ${health.providers.map((p) => p.label).join(' → ')}`
              : 'Add a free GEMINI_API_KEY or GROQ_API_KEY to backend/.env and restart the backend.'}</span></div>
        </li>
        <li className={hasDocs ? 'done' : ''}>
          <span className="step-mark">{hasDocs ? <Icon name="check" size={14} /> : '2'}</span>
          <div><b>Add documents</b><span>{hasDocs ? `${docCount} in your library` : 'Drop PDFs, Word, Markdown, HTML or text files into the library.'}</span></div>
        </li>
        <li>
          <span className="step-mark">3</span>
          <div><b>Ask a question</b><span>Thorough mode runs the full audit; Fast mode is cheaper on free-tier limits.</span></div>
        </li>
      </ol>
      {examples.length > 0 && (
        <div className="examples">
          {examples.map((q) => <button key={q} className="chip-btn" onClick={() => onPick(q)}>{q}</button>)}
        </div>
      )}
    </div>
  )
}

// ── Notices (rate-limit waits, failovers) ─────────────────────────────────
export function NoticeStrip({ notices, running }) {
  const [, tick] = useState(0)
  const last = notices[notices.length - 1]
  const waiting = running && last?.level === 'wait' && last.wait_secs
    ? Math.max(0, Math.ceil(last.wait_secs - (Date.now() - last.at) / 1000)) : 0
  useEffect(() => {
    if (!waiting) return
    const t = setInterval(() => tick((n) => n + 1), 500)
    return () => clearInterval(t)
  }, [waiting])
  if (!last || !running) return null
  const tone = last.level === 'error' ? 'bad' : last.level === 'failover' ? 'accent' : 'warn'
  return (
    <div className={`notice tone-${tone}`} role="status">
      <Icon name={last.level === 'wait' ? 'clock' : last.level === 'error' ? 'alert' : 'refresh'} />
      <span>{last.level === 'wait' && waiting > 0
        ? last.message.replace(/waiting \d+s/, `resuming in ${waiting}s`) : last.message}</span>
    </div>
  )
}

// ── Answer ────────────────────────────────────────────────────────────────
export function AnswerCard({ run, activeRef, setActiveRef }) {
  const streaming = run.status === 'running' && run.stages.answer.status === 'running'
  const conf = run.confidence
  const lv = conf ? LEVEL[conf.level] : null
  const nSources = run.sources.length

  const cite = (n, key) => (
    <button key={key} type="button"
      className={`cite ${activeRef?.n === n ? 'on' : ''} ${n > nSources ? 'invalid' : ''}`}
      onMouseEnter={() => setActiveRef(n)} onFocus={() => setActiveRef(n)}
      onClick={() => setActiveRef(n, true)}
      title={run.sources[n - 1]?.label || 'Source not found'}>{n}</button>
  )

  return (
    <article className="card answer-card">
      <header className="answer-head">
        <h2 className="question">{run.question}</h2>
        <div className="answer-meta">
          {lv && <span className={`badge tone-${lv.tone}`}>{lv.label} confidence{conf.level !== 'ABSTAINED' && conf.level !== 'NO_EVIDENCE' ? ` · ${pct(conf.score)}` : ''}</span>}
          {run.mode === 'fast' && <span className="badge tone-neutral">Fast mode</span>}
          {run.model && <span className="muted small mono">{run.model}</span>}
          {run.durationMs != null && <span className="muted small">{secs(run.durationMs)}</span>}
        </div>
      </header>

      {run.abstained && run.status !== 'running' && (
        <div className="callout tone-neutral"><Icon name="alert" /> Your documents don't seem to cover this, so the model declined rather than guess.</div>
      )}
      {conf && conf.level === 'LOW' && (
        <div className="callout tone-bad"><Icon name="alert" /> Low confidence — check the flagged claims below before relying on this answer.</div>
      )}

      <div className="answer-body">
        {run.answer ? <Markdown text={run.answer} cite={cite} />
          : run.status === 'running' ? <div className="skeleton"><span /><span /><span /></div> : null}
        {streaming && <span className="caret" aria-hidden="true" />}
      </div>
      {run.error && <div className="callout tone-bad"><Icon name="alert" /> {run.error}</div>}
    </article>
  )
}

// ── Claims ────────────────────────────────────────────────────────────────
export function ClaimsCard({ run, setActiveRef }) {
  const v = run.verification
  const st = run.stages.verify
  if (!v && st.status !== 'running') return null
  return (
    <section className="card">
      <header className="card-head">
        <h3>Claim audit</h3>
        {v && (
          <div className="tally">
            {['supported', 'partial', 'not_found', 'contradicted'].map((k) => v.counts[k] ? (
              <span key={k} className={`tally-item tone-${VERDICT[k].tone}`}><b>{v.counts[k]}</b> {VERDICT[k].label.toLowerCase()}</span>
            ) : null)}
          </div>
        )}
      </header>
      {!v ? <div className="loading-row"><Spinner /> Checking every claim against its source…</div> : (
        <>
          {!v.judge_ok && <p className="muted small">The auditor model was unavailable, so claims were checked by local semantic matching only.</p>}
          <ul className="claims">
            {v.claims.map((c, i) => {
              const vd = VERDICT[c.verdict] || VERDICT.not_found
              return (
                <li key={i} className={`claim tone-${vd.tone}`}>
                  <span className={`claim-icon tone-${vd.tone}`} title={vd.label}><Icon name={vd.icon} size={13} /></span>
                  <div className="claim-body">
                    <div className="claim-text">{c.claim}</div>
                    <div className="claim-meta">
                      <span className={`verdict tone-${vd.tone}`}>{vd.label}</span>
                      {c.source && <button className="linkish" onClick={() => setActiveRef(c.source, true)}>source {c.source}</button>}
                      <span title="Semantic similarity to the closest source sentence (computed locally)">grounding {pct(c.grounding)}</span>
                      {c.citation_ok === false && <span className="warn-text">citation missing or mismatched</span>}
                      {c.note && <span className="muted">— {c.note}</span>}
                    </div>
                    {c.quote && (
                      <blockquote className={c.quote_found === false ? 'unverified' : ''}>
                        “{c.quote}”
                        {c.quote_found === true && <span className="q-ok" title="Quote found verbatim in the source"><Icon name="check" size={12} /> verified</span>}
                        {c.quote_found === false && <span className="q-bad" title="This quote does not appear in the source">not found in source</span>}
                      </blockquote>
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        </>
      )}
    </section>
  )
}

// ── Review + consistency ──────────────────────────────────────────────────
export function ReviewCard({ run }) {
  const r = run.reflection
  const c = run.consistency
  const busy = ['reflect', 'consistency'].some((k) => run.stages[k].status === 'running')
  if (!r && !c && !busy) return null
  return (
    <section className="card">
      <header className="card-head"><h3>Review</h3></header>
      <div className="review-grid">
        <div>
          <div className="eyebrow">Critical review {r && <b>{pct(r.score)}</b>}</div>
          {!r ? (run.stages.reflect.status === 'running' ? <div className="loading-row"><Spinner /> Reviewing…</div>
            : <p className="muted small">{run.stages.reflect.summary || 'Not run'}</p>) : (
            <>
              {r.summary && <p className="review-summary">{r.summary}</p>}
              {r.issues.length === 0 ? <p className="muted small">No issues raised.</p> : (
                <ul className="issues">
                  {r.issues.map((it, i) => (
                    <li key={i}><span className={`sev sev-${it.severity}`}>{it.severity}</span>
                      <span className="issue-type">{it.type}</span> {it.detail}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
        <div>
          <div className="eyebrow">Self-consistency {c && <b>{pct(c.score)}</b>}</div>
          {!c ? (run.stages.consistency.status === 'running' ? <div className="loading-row"><Spinner /> Re-sampling…</div>
            : <p className="muted small">{run.stages.consistency.summary || 'Not run'}</p>) : (
            <>
              <p className="muted small">The question was answered {c.samples.length} more times independently; similarity to the main answer:</p>
              <div className="sims">
                {c.similarities.map((s, i) => (
                  <details key={i} className="sim">
                    <summary><span className="mono">#{i + 1}</span><span className="bar"><span style={{ width: pct(Math.max(0, s)) }} /></span><span className="mono">{s.toFixed(2)}</span></summary>
                    <p className="sample-text">{c.samples[i]}</p>
                  </details>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}
