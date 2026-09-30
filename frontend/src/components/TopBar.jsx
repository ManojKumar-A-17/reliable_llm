import styles from './TopBar.module.css'

export default function TopBar({ status }) {
  const dotClass = [
    styles.dot,
    status === 'running' ? styles.running :
    status === 'done'    ? styles.done    :
    status === 'error'   ? styles.error   : ''
  ].join(' ')

  return (
    <header className={styles.bar}>
      <div className={styles.logo}>
        reliable_<em>llm</em>
      </div>
      <div className={styles.chip}>v1.0</div>
      <div className={`${styles.chip} ${styles.amber}`}>Groq · qwen3</div>

      <div className={styles.spacer} />

      <div className={styles.statusWrap}>
        <div className={dotClass} />
        <span className={styles.statusLabel}>{status}</span>
      </div>
    </header>
  )
}
