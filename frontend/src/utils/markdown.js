import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'

marked.use(markedKatex({ nonStandard: true, strict: false, throwOnError: false }))

const codeSegmentPattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g
const latexCommandPattern = /\\(?:frac|dfrac|tfrac|sqrt|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|psi|omega|sin|cos|tan|cot|sec|csc|log|ln|exp|lim|sum|prod|int|partial|nabla|infty|times|cdot|pm|mp|leq|geq|neq|approx|equiv|begin|left|right)\b/

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

export function renderMarkdown(content) {
  const normalized = normalizeMathDelimiters(content)
  const html = marked.parse(normalized, { async: false, breaks: true, gfm: true })
  return DOMPurify.sanitize(html)
}
