import { useEffect, useRef } from 'react'
import styles from './Terminal.module.css'

const CLS_MAP = {
  div:   styles.cDiv,
  head:  styles.cHead,
  step:  styles.cStep,
  ok:    styles.cOk,
  score: styles.cScore,
  info:  styles.cMuted,
  warn:  styles.cWarn,
  final: styles.cFinal,
  done:  styles.cDone,
  err:   styles.cErr,
  blank: styles.cBlank,
}

// Model color palette (cycles if more than 4 models)
const MODEL_COLORS = [
  { border: '#38bdf8', bg: 'rgba(56,189,248,0.07)', label: '#38bdf8' },  // sky
  { border: '#a78bfa', bg: 'rgba(167,139,250,0.07)', label: '#a78bfa' },  // violet
  { border: '#34d399', bg: 'rgba(52,211,153,0.07)',  label: '#34d399' },  // emerald
  { border: '#fb923c', bg: 'rgba(251,146,60,0.07)',  label: '#fb923c' },  // orange
]

function LLMResponseCard({ resp, colorIdx }) {
  const c = MODEL_COLORS[colorIdx % MODEL_COLORS.length]
  return (
    <div style={{
      border: `1px solid ${c.border}`,
      background: c.bg,
      borderRadius: 6,
      padding: '10px 14px',
      marginBottom: 10,
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 6,
      }}>
        <span style={{ color: c.label, fontWeight: 600, fontSize: 12, letterSpacing: '0.04em' }}>
          {resp.display_name || resp.model_name}
        </span>
        {resp.generation_time != null && (
          <span style={{ color: '#6b7280', fontSize: 11 }}>
            {Number(resp.generation_time).toFixed(1)}s
          </span>
        )}
      </div>
      <div style={{
        color: resp.error ? '#f87171' : '#d1d5db',
        fontSize: 13,
        lineHeight: 1.55,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {resp.error
          ? `Error: ${resp.error}`
          : (resp.answer || '—')
        }
      </div>
    </div>
  )
}

export default function Terminal({ lines, showCursor, answer, llmResponses }) {
  const ref = useRef(null)

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [lines, llmResponses])

  return (
    <div className={styles.wrap}>
      {/* window chrome */}
      <div className={styles.bar}>
        <span className={`${styles.dot} ${styles.r}`} />
        <span className={`${styles.dot} ${styles.y}`} />
        <span className={`${styles.dot} ${styles.g}`} />
        <span className={styles.title}>bash — reliable_llm</span>
      </div>

      {/* output */}
      <div className={styles.body} ref={ref}>
        {lines.length === 0 ? (
          <div className={styles.idle}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="1.5" opacity="0.3">
              <polyline points="4 17 10 11 4 5" />
              <line x1="12" y1="19" x2="20" y2="19" />
            </svg>
            <p>Enter a question and click <strong style={{ color: 'var(--teal)' }}>Run Pipeline</strong></p>
            <p className={styles.sub}>Results stream live from your Python agents</p>
          </div>
        ) : (
          <>
            {lines.map((l, i) => (
              <span key={i} className={`${styles.ln} ${CLS_MAP[l.type] ?? ''}`}>
                {l.text}
              </span>
            ))}
            {showCursor && <span className={styles.cursor} />}
          </>
        )}
      </div>

      {/* Final Answer strip */}
      {answer && (
        <div className={styles.answerStrip}>
          <div className={styles.answerLabel}>Final Answer</div>
          <div className={styles.answerText}>{answer}</div>
        </div>
      )}

      {/* Multi-LLM comparison panel */}
      {llmResponses && llmResponses.length > 0 && (
        <div style={{
          borderTop: '1px solid #1f2937',
          padding: '14px 16px',
          background: '#0d1117',
        }}>
          <div style={{
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.08em',
            color: '#6b7280',
            textTransform: 'uppercase',
            marginBottom: 12,
          }}>
            Multi-LLM Comparison
          </div>
          {llmResponses.map((resp, i) => (
            <LLMResponseCard key={resp.model_name || i} resp={resp} colorIdx={i} />
          ))}
        </div>
      )}
    </div>
  )
}
