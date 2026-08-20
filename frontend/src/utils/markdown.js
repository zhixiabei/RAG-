import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'

marked.use(markedKatex({ nonStandard: true, strict: false, throwOnError: false }))

const codeSegmentPattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g
const latexCommandPattern = /\\(?:frac|dfrac|tfrac|sqrt|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|psi|omega|sin|cos|tan|cot|sec|csc|log|ln|exp|lim|sum|prod|int|partial|nabla|infty|times|cdot|pm|mp|leq|geq|neq|approx|equiv|begin|left|right)\b/
const citationPattern = /\[证据:([^\]\n]+)\]/g

function citationLookup(citations = []) {
  const byId = new Map()
  for (const citation of citations) {
    const chunkId = String(citation?.chunk_id || '').trim()
    if (!chunkId || byId.has(chunkId)) continue
    byId.set(chunkId, citation)
  }
  return byId
}

function escapeAttribute(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character])
}

function citationTitle(citation) {
  const location = citation.page_number ? ` · 第 ${citation.page_number} 页` : ''
  return `${citation.title || '未命名来源'}${location}`
}

function citationIds(rawIds, lookup) {
  const ids = rawIds.split(/[,，\s]+/).map((item) => item.trim()).filter(Boolean)
  if (!ids.length || !lookup) return ids
  let prefix = ''
  return ids.map((chunkId) => {
    if (lookup.has(chunkId)) {
      const separator = chunkId.lastIndexOf(':')
      prefix = separator >= 0 ? chunkId.slice(0, separator + 1) : ''
      return chunkId
    }
    if (!prefix || chunkId.includes(':')) return chunkId
    const expanded = `${prefix}${chunkId}`
    return lookup.has(expanded) ? expanded : chunkId
  })
}

export function injectInlineCitations(content = '', citations = []) {
  const lookup = citationLookup(citations)
  const numberById = new Map()
  return String(content)
    .split(codeSegmentPattern)
    .map((segment, index) => {
      if (index % 2 === 1) return segment
      return segment.replace(citationPattern, (_, rawIds) => citationIds(rawIds, lookup).map((chunkId) => {
        const citation = lookup.get(chunkId)
        if (!citation) {
          return `<sup class="inline-citation invalid-citation" title="无法定位证据片段 ${escapeAttribute(chunkId)}">[?]</sup>`
        }
        if (!numberById.has(chunkId)) numberById.set(chunkId, numberById.size + 1)
        const number = numberById.get(chunkId)
        return `<sup class="inline-citation" title="${escapeAttribute(citationTitle(citation))}">[${number}]</sup>`
      }).join(''))
    })
    .join('')
}

export function referencedCitations(content = '', citations = []) {
  const lookup = citationLookup(citations)
  const referenced = []
  const seen = new Set()
  String(content).split(codeSegmentPattern).forEach((segment, index) => {
    if (index % 2 === 1) return
    for (const match of segment.matchAll(citationPattern)) {
      citationIds(match[1], lookup).forEach((chunkId) => {
        if (!lookup.has(chunkId) || seen.has(chunkId)) return
        seen.add(chunkId)
        referenced.push(chunkId)
      })
    }
  })
  return referenced.map((chunkId, index) => ({ ...lookup.get(chunkId), number: index + 1 }))
}

function normalizeTextSegment(segment) {
  return segment
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => `\n$$\n${formula.trim()}\n$$\n`)
    .replace(/\\\(([^\n]*?)\\\)/g, (_, formula) => `$${formula.trim()}$`)
    .replace(/\(([^()\n]+)\)/g, (match, formula) => (
      latexCommandPattern.test(formula) ? `$${formula.trim()}$` : match
    ))
}

export function normalizeMathDelimiters(content = '') {
  return String(content)
    .split(codeSegmentPattern)
    .map((segment, index) => (index % 2 === 0 ? normalizeTextSegment(segment) : segment))
    .join('')
}

export function renderMarkdown(content, citations = []) {
  const normalized = normalizeMathDelimiters(content)
  const withCitations = injectInlineCitations(normalized, citations)
  const html = marked.parse(withCitations, { async: false, breaks: true, gfm: true })
  return DOMPurify.sanitize(html)
}
