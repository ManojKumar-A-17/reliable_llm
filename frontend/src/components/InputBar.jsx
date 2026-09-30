import styles from './InputBar.module.css'

export default function InputBar({ question, setQuestion, onRun, running }) {
  return (
    <div className={styles.bar}>
      <span className={styles.prompt}>$ question →</span>
      <input
        className={styles.input}
        value={question}
        onChange={e => setQuestion(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && !running && onRun()}
        disabled={running}
        placeholder="Ask anything..."
        spellCheck={false}
      />
      <button
        className={styles.btn}
        onClick={onRun}
        disabled={running}
      >
        {running ? '⟳ Running…' : '▶ Run Pipeline'}
      </button>
    </div>
  )
}
