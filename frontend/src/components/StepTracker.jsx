const STEPS = [
  'Evidence Retrieval',
  'Cited Answer Generation',
  'Claim Verification',
  'Consistency Sampling',
  'Self-Reflection',
  'Confidence Calibration',
]

const ICONS = ['🔍', '✍️', '🔬', '🔄', '🪞', '📊']

export default function StepTracker({ steps }) {
  return (
    <div className="step-tracker">
      <div className="tracker-label">Pipeline</div>
      {steps.map((step, i) => (
        <div key={step.id} className={`step-row step-${step.status}`}>
          <span className="step-icon">
            {step.status === 'running' ? <span className="dot-spin" /> :
             step.status === 'done'    ? '✓' :
             step.status === 'skip'   ? '–' :
             ICONS[i]}
          </span>
          <div className="step-info">
            <span className="step-name">{STEPS[i]}</span>
            {step.detail && <span className="step-detail">{step.detail}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}
