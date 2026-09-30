/**
 * AnswerDisplay — cited answer text with interactive hover popovers.
 * Parses [Source N] markers and renders inline citation badges.
 */

import { useState } from 'react'

function CitationBadge({ num, sources }) {
  const [open, setOpen] = useState(false)
  const source = sources && sources[num - 1]

  return (
    <span
      className="cite-badge"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {num}
      {open && source && (
        <span className="cite-popover">
          <span className="cite-pop-header">{source.source || `Source ${num}`}</span>
          <span className="cite-pop-text">{(source.text || '').slice(0, 240)}{source.text?.length > 240 ? '…' : ''}</span>
        </span>
      )}
    </span>
  )
}

function parseAnswer(text, sources) {
  // Matches [Source N] and [N]
  const parts = []
  const re = /\[(?:Source\s*)?(\d+)\]/gi
  let last = 0
  let match

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parts.push({ type: 'text', content: text.slice(last, match.index) })
    }
    parts.push({ type: 'cite', num: parseInt(match[1], 10) })
    last = match.index + match[0].length
  }
  if (last < text.length) {
    parts.push({ type: 'text', content: text.slice(last) })
  }
  return parts
}

export default function AnswerDisplay({ answer = '', sources = [] }) {
  if (!answer) return null
  const parts = parseAnswer(answer, sources)

  return (
    <div className="answer-body">
      <p className="answer-prose">
        {parts.map((part, i) =>
          part.type === 'text'
            ? <span key={i}>{part.content}</span>
            : <CitationBadge key={i} num={part.num} sources={sources} />
        )}
      </p>
    </div>
  )
}
