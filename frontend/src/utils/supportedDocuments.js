const extensions = [
  '.jsonl',
  '.json',
  '.pdf',
  '.doc',
  '.docx',
  '.xlsx',
  '.pptx',
  '.txt',
  '.md',
  '.markdown',
]

export const SUPPORTED_DOCUMENT_ACCEPT = extensions.join(',')

export function documentExtension(name) {
  const index = name.lastIndexOf('.')
  return index > 0 ? name.slice(index).toLowerCase() : ''
}

export function isSupportedDocument(name) {
  return extensions.includes(documentExtension(name))
}
