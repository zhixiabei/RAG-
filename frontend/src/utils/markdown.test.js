import assert from 'node:assert/strict'
import test from 'node:test'
import { marked } from 'marked'

import { normalizeMathDelimiters } from './markdown.js'

test('normalizes common LaTeX delimiters', () => {
  assert.equal(normalizeMathDelimiters('inline \\(x^2\\)'), 'inline $x^2$')
  assert.equal(normalizeMathDelimiters('\\[x^2\\]'), '\n$$\nx^2\n$$\n')
})

test('repairs model output that uses plain parentheses around LaTeX', () => {
  const input = '反射系数：(\\Gamma = \\frac{\\eta_2 - \\eta_1}{\\eta_2 + \\eta_1})'
  const expected = '反射系数：$\\Gamma = \\frac{\\eta_2 - \\eta_1}{\\eta_2 + \\eta_1}$'

  assert.equal(normalizeMathDelimiters(input), expected)
  assert.match(marked.parse(normalizeMathDelimiters(input)), /class="katex"/)
})

test('does not alter normal parentheses or code', () => {
  const input = '普通括号 (说明)\n`(\\Gamma)`\n```text\n(\\frac{a}{b})\n```'

  assert.equal(normalizeMathDelimiters(input), input)
})
