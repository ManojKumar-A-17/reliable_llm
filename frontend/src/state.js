// Run state: a reducer fed by the backend's SSE events (or a saved run).

export const STAGE_ORDER = ['retrieve', 'answer', 'verify', 'consistency', 'reflect', 'calibrate']

export const STAGE_META = {
  retrieve: { name: 'Evidence retrieval', blurb: 'Hybrid search + rerank over your library' },
  answer: { name: 'Cited answer', blurb: 'Written only from retrieved passages' },
  verify: { name: 'Claim audit', blurb: 'Each claim checked against the sources' },
  consistency: { name: 'Self-consistency', blurb: 'Independent re-samples compared' },
  reflect: { name: 'Critical review', blurb: 'Looks for omissions and overclaims' },
  calibrate: { name: 'Confidence', blurb: 'Signals combined into one score' },
}

const blankStages = () =>
  Object.fromEntries(STAGE_ORDER.map((id) => [id, { id, status: 'idle', summary: '', duration_ms: null }]))

export const initialRun = {
  status: 'idle', // idle | running | done | error
  id: null,
  question: '',
  mode: 'thorough',
  stages: blankStages(),
  answer: '',
  abstained: false,
  sources: [],
  verification: null,
  consistency: null,
  reflection: null,
  confidence: null,
  notices: [],
  calls: [],
  error: null,
  startedAt: null,
  durationMs: null,
  fromHistory: false,
}

export function runReducer(state, action) {
  switch (action.type) {
    case 'reset':
      return initialRun
    case 'start':
      return { ...initialRun, status: 'running', question: action.question, mode: action.mode,
               stages: blankStages(), startedAt: Date.now() }
    case 'load': {
      const r = action.result
      const stages = blankStages()
      for (const id of STAGE_ORDER) {
        const saved = r.stages?.[id]
        stages[id] = { id, status: saved?.status || (r.timings_ms?.[id] != null ? 'done' : 'skipped'),
                       summary: saved?.summary || '', duration_ms: r.timings_ms?.[id] ?? null }
      }
      return {
        ...initialRun, status: 'done', fromHistory: true, id: r.id, question: r.question, mode: r.mode,
        stages, answer: r.answer?.text || '', abstained: !!r.answer?.abstained, sources: r.sources || [],
        verification: r.verification, consistency: r.consistency, reflection: r.reflection,
        confidence: r.confidence, notices: r.notices || [], calls: r.llm_calls || [],
        durationMs: r.duration_ms, createdAt: r.created_at, model: r.answer?.model, provider: r.answer?.provider,
      }
    }
    case 'abort':
      return { ...state, status: state.answer ? 'done' : 'idle', error: state.answer ? 'Stopped.' : null,
               stages: Object.fromEntries(Object.entries(state.stages).map(([k, s]) =>
                 [k, s.status === 'running' ? { ...s, status: 'skipped', summary: 'Stopped' } : s])) }
    case 'fail':
      return { ...state, status: 'error', error: action.message,
               stages: Object.fromEntries(Object.entries(state.stages).map(([k, s]) =>
                 [k, s.status === 'running' ? { ...s, status: 'error' } : s])) }
    case 'event':
      return applyEvent(state, action.event, action.data)
    default:
      return state
  }
}

function applyEvent(state, event, d) {
  switch (event) {
    case 'run_started':
      return { ...state, id: d.run_id }
    case 'stage': {
      const prev = state.stages[d.id] || { id: d.id }
      const next = { ...state, stages: { ...state.stages, [d.id]: { ...prev, status: d.status,
        summary: d.summary ?? prev.summary, duration_ms: d.duration_ms ?? prev.duration_ms,
        startedAt: d.status === 'running' ? Date.now() : prev.startedAt } } }
      if (d.sources) next.sources = d.sources
      if (d.verification) next.verification = d.verification
      if (d.consistency) next.consistency = d.consistency
      if (d.reflection) next.reflection = d.reflection
      if (d.confidence) next.confidence = d.confidence
      if (d.model) { next.model = d.model; next.provider = d.provider }
      return next
    }
    case 'token':
      return { ...state, answer: state.answer + d.text }
    case 'answer_reset':
      return { ...state, answer: '' }
    case 'answer':
      return { ...state, answer: d.text, abstained: d.abstained }
    case 'notice':
      return { ...state, notices: [...state.notices, { ...d, at: Date.now() }] }
    case 'llm_call':
      return { ...state, calls: [...state.calls, d] }
    case 'complete':
      return { ...state, status: 'done', durationMs: d.result.duration_ms, createdAt: d.result.created_at }
    case 'error':
      return runReducer(state, { type: 'fail', message: d.message })
    default:
      return state
  }
}
