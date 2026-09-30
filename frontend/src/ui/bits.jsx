// Small shared pieces: icons, formatting helpers, markdown-lite renderer.
import { Fragment } from 'react'

// ── Icons (1.6px stroke, 20px grid) ────────────────────────────────────────
const P = {
  upload: 'M10 13V3m0 0L6 7m4-4l4 4M4 13v2.5A1.5 1.5 0 005.5 17h9a1.5 1.5 0 001.5-1.5V13',
  trash: 'M4 6h12M8 6V4.5A1.5 1.5 0 019.5 3h1A1.5 1.5 0 0112 4.5V6m2 0l-.6 9.1A2 2 0 0111.4 17H8.6a2 2 0 01-2-1.9L6 6',
  doc: 'M6 2.5h5.5L15 6v10a1.5 1.5 0 01-1.5 1.5h-7A1.5 1.5 0 015 16V4a1.5 1.5 0 011-1.5zM11 2.5V6.5h4',
  send: 'M3.5 10h11m0 0l-4.5-4.5M14.5 10L10 14.5',
  stop: 'M6 6h8v8H6z',
  check: 'M4.5 10.5l3.5 3.5 7.5-8',
  x: 'M5.5 5.5l9 9m0-9l-9 9',
  dash: 'M5.5 10h9',
  alert: 'M10 7v4m0 2.6v.1M8.6 3.3L2.8 13.8A1.6 1.6 0 004.2 16.2h11.6a1.6 1.6 0 001.4-2.4L11.4 3.3a1.6 1.6 0 00-2.8 0z',
  clock: 'M10 17a7 7 0 100-14 7 7 0 000 14zm0-10.5V10l2.5 1.5',
  sun: 'M10 13.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7zM10 1.5v2m0 13v2M3.99 3.99l1.41 1.41m9.2 9.2l1.41 1.41M1.5 10h2m13 0h2M3.99 16.01l1.41-1.41m9.2-9.2l1.41-1.41',
  moon: 'M16.5 11.8A7 7 0 118.2 3.5a5.5 5.5 0 008.3 8.3z',
  history: 'M3 10a7 7 0 107-7 7.2 7.2 0 00-5 2.1L3 7m0-4v4h4M10 6.5V10l2.5 2',
  chevron: 'M7.5 5l5 5-5 5',
  link: 'M8.5 11.5a3.5 3.5 0 005 0l2-2a3.5 3.5 0 00-5-5l-1 1m1 5a3.5 3.5 0 00-5 0l-2 2a3.5 3.5 0 005 5l1-1',
  spark: 'M10 2.5v4m0 7v4M2.5 10h4m7 0h4M5 5l2.2 2.2m5.6 5.6L15 15m0-10l-2.2 2.2m-5.6 5.6L5 15',
  menu: 'M3.5 6h13M3.5 10h13M3.5 14h13',
  quote: 'M5 8.5h3.5V13H4V9.5A5 5 0 017 5m4.5 3.5H15V13h-4.5V9.5A5 5 0 0113.5 5',
  refresh: 'M16 10a6 6 0 11-1.8-4.3M16 3.5v3h-3',
}

export function Icon({ name, size = 16, className = '', title }) {
  return (
    <svg className={`icon ${className}`} width={size} height={size} viewBox="0 0 20 20" fill="none"
      stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden={title ? undefined : true} role={title ? 'img' : undefined}>
      {title && <title>{title}</title>}
      <path d={P[name]} />
    </svg>
  )
}

export function Spinner({ size = 14 }) {
  return <span className="spinner" style={{ width: size, height: size }} aria-label="Working" />
}

// ── Formatting ────────────────────────────────────────────────────────────
export const pct = (v, digits = 0) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

export function secs(ms) {
  if (ms == null) return ''
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  return s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

export function bytes(n) {
  if (!n && n !== 0) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function ago(iso) {
  if (!iso) return ''
  const d = (Date.now() - new Date(iso).getTime()) / 1000
  if (d < 60) return 'just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  if (d < 86400 * 7) return `${Math.floor(d / 86400)}d ago`
  return new Date(iso).toLocaleDateString()
}

export const LEVEL = {
  HIGH: { tone: 'ok', label: 'High' },
  MEDIUM: { tone: 'warn', label: 'Medium' },
  LOW: { tone: 'bad', label: 'Low' },
  ABSTAINED: { tone: 'neutral', label: 'Declined' },
  NO_EVIDENCE: { tone: 'neutral', label: 'No evidence' },
}

export const VERDICT = {
  supported: { tone: 'ok', label: 'Supported', icon: 'check' },
  partial: { tone: 'warn', label: 'Partial', icon: 'dash' },
  not_found: { tone: 'bad', label: 'Not in sources', icon: 'x' },
  contradicted: { tone: 'bad', label: 'Contradicted', icon: 'alert' },
}

// ── Markdown-lite with citation chips ─────────────────────────────────────
// Supports paragraphs, bullet & numbered lists, **bold**, *italic*, `code`,
// and [n] / [n, m] citations. Rendered as React nodes (no innerHTML).

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\d+(?:\s*[,;]\s*\d+)*\])/g

function renderInline(text, cite, keyBase) {
  const parts = text.split(INLINE)
  return parts.map((part, i) => {
    const k = `${keyBase}-${i}`
    if (!part) return null
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={k}>{part.slice(2, -2)}</strong>
    if (/^\*[^*]/.test(part) && part.endsWith('*') && part.length > 2) return <em key={k}>{part.slice(1, -1)}</em>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={k}>{part.slice(1, -1)}</code>
    const m = part.match(/^\[(\d+(?:\s*[,;]\s*\d+)*)\]$/)
    if (m) {
      const nums = m[1].split(/[,;]/).map((n) => parseInt(n.trim(), 10))
      return <Fragment key={k}>{nums.map((n) => cite(n, `${k}-${n}`))}</Fragment>
    }
    return <Fragment key={k}>{part}</Fragment>
  })
}

export function Markdown({ text, cite }) {
  const lines = (text || '').split('\n')
  const blocks = []
  let list = null
  let para = []
  const flushPara = () => {
    if (para.length) { blocks.push({ type: 'p', text: para.join(' ') }); para = [] }
  }
  const flushList = () => { if (list) { blocks.push(list); list = null } }

  for (const raw of lines) {
    const line = raw.trimEnd()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/)
    if (bullet || numbered) {
      flushPara()
      const type = bullet ? 'ul' : 'ol'
      if (!list || list.type !== type) { flushList(); list = { type, items: [] } }
      list.items.push(bullet ? bullet[1] : numbered[2])
    } else if (!line.trim()) {
      flushPara(); flushList()
    } else if (list && /^\s{2,}/.test(raw)) {
      list.items[list.items.length - 1] += ' ' + line.trim()
    } else {
      flushList()
      para.push(line.replace(/^#{1,6}\s+/, ''))
    }
  }
  flushPara(); flushList()

  return blocks.map((b, i) => {
    if (b.type === 'p') return <p key={i}>{renderInline(b.text, cite, `p${i}`)}</p>
    const Tag = b.type
    return <Tag key={i}>{b.items.map((it, j) => <li key={j}>{renderInline(it, cite, `l${i}-${j}`)}</li>)}</Tag>
  })
}
