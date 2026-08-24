import assert from 'node:assert/strict'
import test from 'node:test'
import { marked } from 'marked'

import {
  citationSourceMeta,
  injectInlineCitations,
  normalizeMathDelimiters,
  referencedCitations,
} from './markdown.js'

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

test('renders valid evidence markers as numbered inline citations', () => {
  const citations = [
    { chunk_id: 'chunk-1', title: '构造演化.pdf', page_number: 3, excerpt: '原文片段' },
    { chunk_id: 'chunk-2', title: '盆地研究.pdf', page_number: 8, excerpt: '另一片段' },
  ]

  const rendered = injectInlineCitations(
    '第一段。[证据:chunk-1]\n\n第二段。[证据:chunk-1,chunk-2]',
    citations,
  )

  assert.match(rendered, /class="inline-citation"[^>]*>\[1\]<\/sup>/)
  assert.match(rendered, />\[2\]<\/sup>/)
  assert.match(rendered, /构造演化\.pdf/)
})

test('formats citation location with a page or section and includes the chunk id', () => {
  assert.equal(
    citationSourceMeta({ chunk_id: 'doc-1:0', page_number: 3, section_path: '摘要' }),
    '第 3 页 · Chunk ID doc-1:0',
  )
  assert.equal(
    citationSourceMeta({ chunk_id: 'doc-2:0', section_path: '第一章/范围' }),
    '章节 第一章/范围 · Chunk ID doc-2:0',
  )
})

test('uses the section in inline citation tooltips when no page is available', () => {
  const rendered = injectInlineCitations(
    '结论。[证据:doc-1:0]',
    [{ chunk_id: 'doc-1:0', title: '制度.pdf', section_path: '第一章/范围' }],
  )

  assert.match(rendered, /制度\.pdf · 章节 第一章\/范围/)
})

test('marks unknown evidence ids and leaves code examples unchanged', () => {
  const rendered = injectInlineCitations(
    '正文。[证据:missing]\n`示例 [证据:chunk-1]`',
    [{ chunk_id: 'chunk-1', title: '来源.pdf' }],
  )

  assert.match(rendered, /invalid-citation/)
  assert.match(rendered, /`示例 \[证据:chunk-1\]`/)
})

test('returns only sources that the answer actually cites', () => {
  const sources = referencedCitations('结论。[证据:chunk-2]', [
    { chunk_id: 'chunk-1', title: '未引用.pdf' },
    { chunk_id: 'chunk-2', title: '已引用.pdf' },
  ])

  assert.deepEqual(sources, [{ chunk_id: 'chunk-2', title: '已引用.pdf', number: 1 }])
})
