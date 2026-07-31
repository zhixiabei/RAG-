export function canRecoverAnswerFailure(cause) {
  return !cause?.status || [404, 408, 502, 503, 504].includes(cause.status)
}

export async function recoverCompletedAnswer({
  loadMessages,
  shouldContinue,
  previousMessageCount,
  timeoutMs = 60_000,
  initialDelayMs = 800,
  maxDelayMs = 3_000,
  now = Date.now,
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
}) {
  const deadline = now() + timeoutMs
  let delay = initialDelayMs
  while (now() < deadline) {
    if (!shouldContinue()) return null
    try {
      const history = await loadMessages()
      const answerCompleted = history.length >= previousMessageCount + 2
        && history.at(-1)?.role === 'assistant'
      if (answerCompleted) return history
    } catch (cause) {
      // A missing conversation cannot recover; transport failures may be transient.
      if (cause?.status === 401 || cause?.status === 404) return null
    }
    await wait(delay)
    delay = Math.min(maxDelayMs, Math.round(delay * 1.5))
  }
  return null
}
