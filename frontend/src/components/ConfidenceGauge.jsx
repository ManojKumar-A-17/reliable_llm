/**
 * ConfidenceGauge — SVG semicircle arc gauge
 * Shows overall confidence score with evidence/consistency/reflection breakdown.
 */

const LEVEL_COLOR = { HIGH: '#10b981', MEDIUM: '#f59e0b', LOW: '#ef4444' }

function ArcGauge({ score = 0, level = 'LOW', size = 120 }) {
  const color   = LEVEL_COLOR[level] || '#7b9ab8'
  const r       = 44
  const cx      = size / 2
  const cy      = size / 2 + 10
  const circumf = Math.PI * r                   // semicircle circumference
  const pct     = Math.max(0, Math.min(1, score))
  const dash    = circumf * pct
  const gap     = circumf - dash
  const startX  = cx - r
  const startY  = cy

  // Semicircle arc path (left → right, top half)
  const trackD  = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`

  return (
    <div style={{ position: 'relative', width: size, height: size * 0.65 }}>
      <svg
        className="gauge-svg"
        width={size}
        height={size * 0.65}
        viewBox={`0 0 ${size} ${size * 0.65}`}
      >
        {/* Track */}
        <path
          d={trackD}
          fill="none"
          stroke="rgba(255,255,255,.06)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={trackD}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap + 1}`}
          strokeDashoffset="0"
          style={{ transition: 'stroke-dasharray .8s cubic-bezier(.4,0,.2,1)', filter: `drop-shadow(0 0 6px ${color}66)` }}
        />
      </svg>
      {/* Center text */}
      <div style={{
        position: 'absolute',
        top: '52%', left: '50%',
        transform: 'translate(-50%, -50%)',
        textAlign: 'center',
        pointerEvents: 'none',
      }}>
        <div className="gauge-score" style={{ color }}>
          {Math.round(pct * 100)}
          <span style={{ fontSize: '12px', fontWeight: 500 }}>%</span>
        </div>
        <div className="gauge-level" style={{ color }}>{level}</div>
      </div>
    </div>
  )
}

function BreakdownBar({ label, value, color }) {
  if (value === null || value === undefined) return null
  const pct = Math.round(value * 100)
  return (
    <div className="bd-row">
      <span className="bd-name">{label}</span>
      <div className="bd-bar-bg">
        <div className="bd-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="bd-pct" style={{ color }}>{pct}%</span>
    </div>
  )
}

export default function ConfidenceGauge({ verdict = {}, breakdown = {} }) {
  const { score = 0, level = 'LOW', explanation = '' } = verdict

  return (
    <div className="rcard-body">
      <div className="confidence-layout">
        <div className="gauge-wrap">
          <ArcGauge score={score} level={level} size={130} />
        </div>
        <div className="conf-detail">
          {explanation && (
            <p className="conf-explanation">{explanation}</p>
          )}
          <div className="breakdown-chips">
            <BreakdownBar
              label="Evidence"
              value={breakdown.evidence_score}
              color="#10b981"
            />
            <BreakdownBar
              label="Consistency"
              value={breakdown.consistency_score}
              color="#3b82f6"
            />
            <BreakdownBar
              label="Reflection"
              value={breakdown.reflection_score}
              color="#8b5cf6"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
