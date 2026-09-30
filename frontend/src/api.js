// Thin client for the FastAPI backend. All paths are relative; Vite proxies /api.

async function request(path, opts = {}) {
  let res
  try {
    res = await fetch(path, opts)
  } catch {
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  const text = await res.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = { detail: text } }
  if (!res.ok) {
    const detail = data?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${res.status})`)
  }
  return data
}

export const api = {
  health: () => request('/api/health'),
  documents: () => request('/api/documents'),
  upload: (files) => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    return request('/api/documents', { method: 'POST', body: fd })
  },
  deleteDocument: (id) => request(`/api/documents/${id}`, { method: 'DELETE' }),
  documentChunks: (id) => request(`/api/documents/${id}/chunks`),
  runs: (limit = 60) => request(`/api/runs?limit=${limit}`),
  run: (id) => request(`/api/runs/${id}`),
  deleteRun: (id) => request(`/api/runs/${id}`, { method: 'DELETE' }),
  clearRuns: () => request('/api/runs', { method: 'DELETE' }),
  metrics: () => request('/api/metrics'),
}

/**
 * POST a question and consume the Server-Sent Events stream.
 * onEvent(eventName, payload) is called for every event.
 */
export async function streamQuery(body, onEvent, signal) {
  let res
  try {
    res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  if (!res.ok || !res.body) {
    let msg = `Request failed (${res.status})`
    try {
      const d = await res.json()
      if (typeof d.detail === 'string') msg = d.detail
      else if (Array.isArray(d.detail)) msg = d.detail.map((x) => x.msg).join('; ')
    } catch { /* ignore */ }
    throw new Error(msg)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      let event = 'message'
      const data = []
      for (const line of block.split('\n')) {
        if (line.startsWith(':')) continue
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
      }
      if (!data.length) continue
      try { onEvent(event, JSON.parse(data.join('\n'))) } catch { /* malformed event */ }
    }
  }
}
