import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { createKnowledgeBase, listDocuments, listKnowledgeBases } from '../services/api'

export const useKnowledgeBaseStore = defineStore('knowledgeBase', () => {
  const items = ref([])
  const documents = ref([])
  const selectedId = ref(null)
  const loading = ref(false)
  const error = ref('')
  const selected = computed(() => items.value.find((item) => item.id === selectedId.value) || null)

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
    documents.value = knowledgeBaseId ? await listDocuments(knowledgeBaseId) : []
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

  return { items, documents, selectedId, selected, loading, error, load, loadDocuments, select, create }
})

