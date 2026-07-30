const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080').replace(/\/$/, '')

async function request(path, init) {
  const response = await fetch(`${API_BASE_URL}${path}`, { credentials: 'include', ...init })
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
    if (response.status === 401 && !path.startsWith('/api/v1/auth/')) {
      window.dispatchEvent(new Event('rag:unauthorized'))
    }
    throw error
  }
  if (response.status === 204) return null
  return response.json()
}

export function login(username, password) {
  return request('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
}

export function logout() {
  return request('/api/v1/auth/logout', { method: 'POST' })
}

export function getCurrentUser() {
  return request('/api/v1/auth/me')
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
