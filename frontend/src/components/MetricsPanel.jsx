import styles from './MetricsPanel.module.css'

const METRICS = [
  { key: 'fact_check',   label: 'Fact Check',   color: 'var(--green)',  weight: '30%' },
  { key: 'reflection',   label: 'Reflection',   color: 'var(--blue)',   weight: '25%' },
  { key: 'reason',       label: 'Reasoning',    color: 'var(--purple)', weight: '20%' },
  { key: 'consistency',  label: 'Consistency',  color: 'var(--amber)',  weight: '15%' },
  { key: 'similarity',   label: 'Similarity',   color: 'var(--orange)', weight: '10%' },
]

function ScoreRow({ label, color, weight, value }) {
  return (
    <div className={styles.scoreRow}>
      <div className={styles.scoreTop}>
        <span className={styles.scoreLabel}>{label}</span>
        <span
          className={styles.scoreVal}
          style={{ color: value != null ? color : 'var(--muted2)' }}
        >
          {value != null ? value.toFixed(4) : '—'}
        </span>
      </div>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{ width: value != null ? `${value * 100}%` : '0%', background: color }}
        />
      </div>
      <div className={styles.weightNote}>weight {weight}</div>
    </div>
  )
}

export default function MetricsPanel({ scores, verdict, breakdown }) {
  const isHigh = verdict && verdict.score >= 0.70

  return (
    <aside className={styles.panel}>
      <div className={styles.heading}>Confidence Scores</div>

      {METRICS.map(m => (
        <ScoreRow key={m.key} {...m} value={scores[m.key] ?? null} />
      ))}

      {/* verdict badge */}
      <div className={styles.verdictWrap}>
        <div className={[styles.badge, verdict ? (isHigh ? styles.high : styles.low) : ''].join(' ')}>
          <div
            className={styles.bigScore}
            style={{ color: verdict ? (isHigh ? 'var(--green)' : 'var(--red)') : 'var(--muted2)' }}
          >
            {verdict ? verdict.score.toFixed(4) : '—'}
          </div>
          <div
            className={styles.level}
            style={{ color: verdict ? (isHigh ? 'var(--green)' : 'var(--red)') : 'var(--muted)' }}
          >
            {verdict ? verdict.level : 'Awaiting results'}
          </div>
          {verdict && (
            <div className={styles.explanation}>{verdict.explanation}</div>
          )}
        </div>
      </div>

      {/* weighted breakdown */}
      {breakdown && (
        <div className={styles.breakdown}>
          <div className={styles.bdTitle}>Weighted Breakdown</div>
          {[
            ['Reflection',  '25%', breakdown.reflection],
            ['Fact Check',  '30%', breakdown.fact_check],
            ['Reason',      '20%', breakdown.reason],
            ['Consistency', '15%', breakdown.consistency],
            ['Similarity',  '10%', breakdown.similarity],
          ].map(([name, w, val]) => (
            <div key={name} className={styles.bdRow}>
              <span>{name} ({w})</span>
              <span className={styles.bdVal}>{val.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  )
}
