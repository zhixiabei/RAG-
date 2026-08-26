// Keep production requests same-origin so a remote browser never resolves the
// backend address against its own localhost. VITE_API_BASE_URL remains an
// explicit override for deployments that expose the API on another origin.
const configuredApiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const browserHost = typeof globalThis.location?.hostname === 'string' ? globalThis.location.hostname : ''
let loopbackApi = false
try {
  loopbackApi = ['127.0.0.1', 'localhost', '[::1]'].includes(new URL(configuredApiBase).hostname)
} catch {
  // Relative and invalid overrides are left untouched.
}
const browserIsLoopback = browserHost === '127.0.0.1' || browserHost === 'localhost' || browserHost === '[::1]'
const API_BASE_URL = loopbackApi && !browserIsLoopback ? '' : configuredApiBase

async function request(path, init) {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) {
    let message = `请求失败（${response.status}）`
    try {
      const payload = await response.json()
      message = payload.detail || message
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    const error = new Error(message)
    error.status = response.status
    throw error
  }
  if (response.status === 204) return null
  return response.json()
}

export function listKnowledgeBases() {
  return request('/api/v1/knowledge-bases')
}

export function createKnowledgeBase(name, description) {
  return request('/api/v1/knowledge-bases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  })
}

export function deleteKnowledgeBase(knowledgeBaseId) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}`, { method: 'DELETE' })
}

export function listDocuments(knowledgeBaseId) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/documents`)
}

export function getDocument(knowledgeBaseId, documentId) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`)
}

export function uploadDocument(knowledgeBaseId, file, folderPath = '') {
  const form = new FormData()
  form.append('file', file, file.name)
  if (folderPath) {
    form.append('folder_path', folderPath)
  }
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/documents`, { method: 'POST', body: form })
}

export function deleteDocument(knowledgeBaseId, documentId) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, { method: 'DELETE' })
}

export function deleteDocumentFolder(knowledgeBaseId, folderPath) {
  const query = new URLSearchParams({ folder_path: folderPath })
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/documents?${query}`, { method: 'DELETE' })
}

export function listEvaluationDatasets(knowledgeBaseId) {
  return request(`/api/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/evaluation-datasets`)
}

export function syncKnowledgeBaseToTestset(knowledgeBaseId) {
  return request(`/api/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/testset-sync`, {
    method: 'POST',
  })
}

export function listEvaluationSamples(knowledgeBaseId, source = 'workshop', datasetId = '') {
  const query = new URLSearchParams({ source })
  if (datasetId) query.set('dataset', datasetId)
  return request(`/api/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/evaluation-samples?${query}`)
}

export function evaluateKnowledgeBase(knowledgeBaseId, questionIds = [], source = 'workshop', datasetId = '') {
  return request(`/api/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/evaluation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question_ids: questionIds, dataset_source: source, dataset_id: datasetId || null }),
  })
}

export function listChatModels() {
  return request('/api/v1/models')
}

export function askKnowledgeBase(knowledgeBaseId, conversationId, question, model) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, question, model: model || null }),
  })
}

export function askKnowledgeBaseWithAttachments(knowledgeBaseId, conversationId, question, model, files) {
  const form = new FormData()
  form.append('conversation_id', conversationId)
  form.append('question', question)
  if (model) form.append('model', model)
  files.forEach((file) => form.append('files', file, file.name))
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/chat-with-attachments`, {
    method: 'POST',
    body: form,
  })
}

export function parseChatAttachment(knowledgeBaseId, file) {
  const form = new FormData()
  form.append('file', file, file.name)
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/chat-attachments/parse`, {
    method: 'POST',
    body: form,
  })
}

export function askKnowledgeBaseWithParsedAttachments(knowledgeBaseId, conversationId, question, model, attachments) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/chat-with-parsed-attachments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: conversationId,
      question,
      model: model || null,
      attachments,
    }),
  })
}

export function listConversations(knowledgeBaseId) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/conversations`)
}

export function createConversation(knowledgeBaseId, title) {
  return request(`/api/v1/knowledge-bases/${knowledgeBaseId}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

export function listConversationMessages(conversationId) {
  return request(`/api/v1/conversations/${conversationId}/messages`)
}

export function renameConversation(conversationId, title) {
  return request(`/api/v1/conversations/${conversationId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

export function deleteConversation(conversationId) {
  return request(`/api/v1/conversations/${conversationId}`, { method: 'DELETE' })
}
