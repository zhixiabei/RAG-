import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import {
  createKnowledgeBase,
  deleteDocument as deleteDocumentRequest,
  deleteDocumentFolder as deleteDocumentFolderRequest,
  deleteKnowledgeBase as deleteKnowledgeBaseRequest,
  listDocuments,
  listKnowledgeBases,
} from '../services/api'

export const useKnowledgeBaseStore = defineStore('knowledgeBase', () => {
  const items = ref([])
  const documents = ref([])
  const selectedId = ref(null)
  const loading = ref(false)
  const error = ref('')
  const documentRequests = new Map()
  const selected = computed(() => items.value.find((item) => item.id === selectedId.value) || null)
  const hasProcessingDocuments = computed(() => documents.value.some((document) => document.status === 'processing'))

  async function load() {
    loading.value = true
    error.value = ''
    try {
      items.value = await listKnowledgeBases()
      if (!selectedId.value || !items.value.some((item) => item.id === selectedId.value)) selectedId.value = items.value[0]?.id || null
      if (selectedId.value) await loadDocuments(selectedId.value)
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : '知识库加载失败'
    } finally {
      loading.value = false
    }
  }

  async function loadDocuments(knowledgeBaseId = selectedId.value) {
    if (!knowledgeBaseId) {
      documents.value = []
      return
    }
    let request = documentRequests.get(knowledgeBaseId)
    if (!request) {
      request = listDocuments(knowledgeBaseId).finally(() => {
        if (documentRequests.get(knowledgeBaseId) === request) documentRequests.delete(knowledgeBaseId)
      })
      documentRequests.set(knowledgeBaseId, request)
    }
    const result = await request
    if (selectedId.value === knowledgeBaseId) documents.value = result
  }

  async function select(id) {
    selectedId.value = id
    await loadDocuments(id)
  }

  async function create(name, description) {
    const item = await createKnowledgeBase(name, description)
    items.value.unshift(item)
    selectedId.value = item.id
    documents.value = []
  }

  async function removeKnowledgeBase(id) {
    const index = items.value.findIndex((item) => item.id === id)
    await deleteKnowledgeBaseRequest(id)
    items.value = items.value.filter((item) => item.id !== id)
    if (selectedId.value !== id) return

    selectedId.value = items.value[Math.min(index, items.value.length - 1)]?.id || null
    await loadDocuments(selectedId.value)
  }

  async function removeDocument(knowledgeBaseId, documentId) {
    await deleteDocumentRequest(knowledgeBaseId, documentId)
    if (selectedId.value === knowledgeBaseId) {
      documents.value = documents.value.filter((document) => document.id !== documentId)
    }
  }

  async function removeDocumentFolder(knowledgeBaseId, folderPath) {
    await deleteDocumentFolderRequest(knowledgeBaseId, folderPath)
    if (selectedId.value === knowledgeBaseId) {
      const normalizedPath = folderPath.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '')
      const prefix = `${normalizedPath}/`
      documents.value = documents.value.filter((document) => {
        const documentPath = String(document.folder_path || '').replaceAll('\\', '/').replace(/^\/+|\/+$/g, '')
        return documentPath !== normalizedPath && !documentPath.startsWith(prefix)
      })
    }
  }

  function reset() {
    items.value = []
    documents.value = []
    selectedId.value = null
    loading.value = false
    error.value = ''
  }

  return {
    items,
    documents,
    selectedId,
    selected,
    hasProcessingDocuments,
    loading,
    error,
    load,
    loadDocuments,
    select,
    create,
    removeKnowledgeBase,
    removeDocument,
    removeDocumentFolder,
    reset,
  }
})
