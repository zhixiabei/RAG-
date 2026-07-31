import assert from 'node:assert/strict'
import test from 'node:test'

import { canRecoverAnswerFailure, recoverCompletedAnswer } from './chatRecovery.js'

test('recognizes failures that may finish after the HTTP response is lost', () => {
  assert.equal(canRecoverAnswerFailure(new TypeError('network error')), true)
  assert.equal(canRecoverAnswerFailure({ status: 404 }), true)
  assert.equal(canRecoverAnswerFailure({ status: 504 }), true)
  assert.equal(canRecoverAnswerFailure({ status: 422 }), false)
})

test('returns canonical history when the missing answer is persisted', async () => {
  const pendingHistory = [{ role: 'user', content: 'old question' }, { role: 'assistant', content: 'old answer' }]
  const completedHistory = [
    ...pendingHistory,
    { role: 'user', content: 'new question' },
    { role: 'assistant', content: 'new answer' },
  ]
  const responses = [pendingHistory, completedHistory]
  let clock = 0

  const result = await recoverCompletedAnswer({
    loadMessages: async () => responses.shift(),
    shouldContinue: () => true,
    previousMessageCount: pendingHistory.length,
    timeoutMs: 10,
    initialDelayMs: 1,
    now: () => clock,
    wait: async (milliseconds) => { clock += milliseconds },
  })

  assert.deepEqual(result, completedHistory)
})

test('stops immediately when the conversation itself is missing', async () => {
  let attempts = 0
  const result = await recoverCompletedAnswer({
    loadMessages: async () => {
      attempts += 1
      throw Object.assign(new Error('missing'), { status: 404 })
    },
    shouldContinue: () => true,
    previousMessageCount: 0,
  })

  assert.equal(result, null)
  assert.equal(attempts, 1)
})
