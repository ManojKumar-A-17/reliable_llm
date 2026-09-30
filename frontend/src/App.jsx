import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api, streamQuery } from './api.js'
import { initialRun, runReducer } from './state.js'
import Sidebar from './ui/Sidebar.jsx'
import Inspector from './ui/Inspector.jsx'
import { AnswerCard, ClaimsCard, Composer, NoticeStrip, ReviewCard, StatusBar, Welcome } from './ui/Main.jsx'
import { Icon } from './ui/bits.jsx'

function readTheme() {
  try { return localStorage.getItem('rllm-theme') } catch { return null }
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)
  const [docs, setDocs] = useState([])
  const [runs, setRuns] = useState([])
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('thorough')
  const [run, dispatch] = useReducer(runReducer, initialRun)
  const [activeRef, setActive] = useState(null)
  const [toasts, setToasts] = useState([])
  const [theme, setTheme] = useState(readTheme)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const abortRef = useRef(null)

  // ── Theme ────────────────────────────────────────────────────────────
  const effectiveTheme = theme || (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
  }, [theme])
  const toggleTheme = () => {
    const next = effectiveTheme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try { localStorage.setItem('rllm-theme', next) } catch { /* private mode */ }
  }

  // ── Toasts ───────────────────────────────────────────────────────────
  const toast = useCallback((message, tone = 'neutral') => {
    const id = Math.random().toString(36).slice(2)
    setToasts((t) => [...t, { id, message, tone }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5200)
  }, [])

  // ── Data loading ─────────────────────────────────────────────────────
  const refreshHealth = useCallback(async () => {
    try { setHealth(await api.health()); setHealthError(false) } catch { setHealthError(true) }
  }, [])
  const refreshDocs = useCallback(async () => {
    try { setDocs((await api.documents()).documents) } catch { /* surfaced by health */ }
  }, [])
  const refreshRuns = useCallback(async () => {
    try { setRuns((await api.runs()).runs) } catch { /* ignore */ }
  }, [])

  useEffect(() => { refreshHealth(); refreshDocs(); refreshRuns() }, [refreshHealth, refreshDocs, refreshRuns])
  useEffect(() => {
    // Poll faster while offline or while local models are still loading.
    const slow = health && !healthError && health.models_ready
    const t = setInterval(() => {
      refreshHealth()
      if (healthError) { refreshDocs(); refreshRuns() }
    }, slow ? 30000 : 3000)
    return () => clearInterval(t)
  }, [health, healthError, refreshHealth, refreshDocs, refreshRuns])

  // ── Actions ──────────────────────────────────────────────────────────
  const setActiveRef = useCallback((n, scroll = false) => {
    setActive(n == null ? null : { n, scroll, t: Date.now() })
    if (scroll) setInspectorOpen(true)
  }, [])

  const upload = async (files) => {
    setUploading(true)
    try {
      const res = await api.upload(files)
      for (const r of res.results) {
        if (r.status === 'added') toast(`Added ${r.name} · ${r.document.chunks} passages`, 'ok')
        else if (r.status === 'duplicate') toast(`${r.name} is already in your library`, 'neutral')
        else toast(`${r.name}: ${r.error}`, 'bad')
      }
      await Promise.all([refreshDocs(), refreshHealth()])
    } catch (e) {
      toast(e.message, 'bad')
    } finally {
      setUploading(false)
    }
  }

  const deleteDoc = async (id) => {
    try { await api.deleteDocument(id); await Promise.all([refreshDocs(), refreshHealth()]); toast('Removed from library') }
    catch (e) { toast(e.message, 'bad') }
  }

  const ask = async (q = question) => {
    const text = q.trim()
    if (!text) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setActive(null)
    setSidebarOpen(false)
    dispatch({ type: 'start', question: text, mode })
    try {
      let failed = false
      await streamQuery({ question: text, mode }, (event, data) => {
        if (event === 'error') failed = true
        dispatch({ type: 'event', event, data })
      }, ctrl.signal)
      if (!failed) setQuestion('')
    } catch (e) {
      if (e.name === 'AbortError') dispatch({ type: 'abort' })
      else dispatch({ type: 'fail', message: e.message })
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null
      refreshRuns()
    }
  }

  const stop = () => abortRef.current?.abort()

  const openRun = async (id) => {
    if (run.status === 'running') return toast('Wait for the current question to finish, or stop it first.')
    try {
      dispatch({ type: 'load', result: await api.run(id) })
      setActive(null)
      setSidebarOpen(false)
    } catch (e) { toast(e.message, 'bad') }
  }

  const deleteRun = async (id) => {
    try { await api.deleteRun(id); refreshRuns(); if (run.id === id) dispatch({ type: 'reset' }) }
    catch (e) { toast(e.message, 'bad') }
  }

  const newQuestion = () => {
    if (run.status === 'running') stop()
    dispatch({ type: 'reset' })
    setQuestion('')
    setSidebarOpen(false)
  }

  const running = run.status === 'running'
  const blocked = !health ? 'Connecting to backend…' : !health.llm_ready ? 'Add an API key in backend/.env first'
    : docs.length === 0 ? 'Add a document to your library first' : ''
  const hasRun = run.status !== 'idle' && run.question

  return (
    <div className="app">
      <Sidebar health={health} docs={docs} onUpload={upload} uploading={uploading} onDeleteDoc={deleteDoc}
        runs={runs} activeRunId={run.id} onOpenRun={openRun} onDeleteRun={deleteRun} onNew={newQuestion}
        open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      {(sidebarOpen || inspectorOpen) && <div className="scrim" onClick={() => { setSidebarOpen(false); setInspectorOpen(false) }} />}

      <main className="main">
        <StatusBar health={health} healthError={healthError} theme={effectiveTheme} onToggleTheme={toggleTheme}
          onMenu={() => setSidebarOpen(true)} onInspector={() => setInspectorOpen(true)} />
        {healthError && (
          <div className="banner tone-bad"><Icon name="alert" />
            <span>Can't reach the backend. Start it from <code>backend/</code> with <code>uvicorn api.server:app --reload --port 8000</code>, or run <code>./start.sh</code>.</span>
          </div>
        )}
        <div className="main-scroll">
          <div className="column">
            {hasRun ? (
              <>
                <NoticeStrip notices={run.notices} running={running} />
                <AnswerCard run={run} activeRef={activeRef} setActiveRef={setActiveRef} />
                <ClaimsCard run={run} setActiveRef={setActiveRef} />
                <ReviewCard run={run} />
              </>
            ) : (
              <Welcome health={health} docCount={docs.length} onPick={(q) => { setQuestion(q) }} />
            )}
          </div>
        </div>
        <div className="composer-dock">
          <div className="column">
            <Composer value={question} onChange={setQuestion} onSubmit={() => ask()} onStop={stop}
              running={running} mode={mode} setMode={setMode} disabled={!!blocked} disabledReason={blocked} />
          </div>
        </div>
      </main>

      <Inspector run={run} activeRef={activeRef} setActiveRef={setActiveRef}
        open={inspectorOpen} onClose={() => setInspectorOpen(false)} />

      <div className="toasts" aria-live="polite">
        {toasts.map((t) => <div key={t.id} className={`toast tone-${t.tone}`}>{t.message}</div>)}
      </div>
    </div>
  )
}
