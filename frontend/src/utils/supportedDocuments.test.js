import assert from 'node:assert/strict'
import test from 'node:test'

import {
  SUPPORTED_DOCUMENT_ACCEPT,
  documentExtension,
  isSupportedDocument,
} from './supportedDocuments.js'

test('allows only the configured document formats', () => {
  const allowed = [
    'records.jsonl',
    'data.JSON',
    'report.pdf',
    'report.docx',
    'table.xlsx',
    'slides.pptx',
    'notes.txt',
    'readme.md',
    'readme.markdown',
  ]
  const rejected = ['legacy.doc', 'table.xls', 'data.csv', 'map.gdb', 'README']

  assert.ok(allowed.every(isSupportedDocument))
  assert.ok(rejected.every((name) => !isSupportedDocument(name)))
})

test('builds the file picker accept value from the same extension list', () => {
  assert.equal(documentExtension('REPORT.PDF'), '.pdf')
  assert.equal(
    SUPPORTED_DOCUMENT_ACCEPT,
    '.jsonl,.json,.pdf,.docx,.xlsx,.pptx,.txt,.md,.markdown',
  )
})
