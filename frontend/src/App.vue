<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { FileStack, MessageSquareText, Plus, Search, X } from 'lucide-vue-next'
import KnowledgeSidebar from './components/KnowledgeSidebar.vue'
import CreateKnowledgeBaseDialog from './components/CreateKnowledgeBaseDialog.vue'
import DocumentList from './components/DocumentList.vue'
import ImportPanel from './components/ImportPanel.vue'
import ChatWorkspace from './components/ChatWorkspace.vue'
import { useKnowledgeBaseStore } from './stores/knowledgeBase'

const store = useKnowledgeBaseStore()
const activeTab = ref('documents')
const createDialogOpen = ref(false)
const creating = ref(false)
const documentQuery = ref('')

const filteredDocuments = computed(() => {
  const terms = documentQuery.value
    .trim()
    .toLocaleLowerCase('zh-CN')
    .split(/\s+/)
    .filter(Boolean)

  if (!terms.length) return store.documents

  return store.documents.filter((document) => {
    const searchableText = `${document.title || ''} ${document.file_name || ''}`.toLocaleLowerCase('zh-CN')
    return terms.every((term) => searchableText.includes(term))
  })
})

const documentCountText = computed(() => {
  if (!documentQuery.value.trim()) return `${store.documents.length} 个文件`
  return `${filteredDocuments.value.length} / ${store.documents.length} 个文件`
})

onMounted(() => store.load())

watch(() => store.selectedId, () => {
  documentQuery.value = ''
})

async function createKnowledgeBase(name, description) {
  creating.value = true
  try {
    await store.create(name, description)
    createDialogOpen.value = false
  } finally {
    creating.value = false
  }
}

async function refreshDocuments() {
  await store.loadDocuments()
}
</script>

<template>
  <div class="app-shell">
    <KnowledgeSidebar
      :items="store.items"
      :selected-id="store.selectedId"
      :loading="store.loading"
      @select="store.select"
      @create="createDialogOpen = true"
      @refresh="store.load"
    />

    <main class="main-area">
      <header class="topbar">
        <div class="breadcrumb"><span>工作区</span><span class="breadcrumb-separator">/</span><strong>{{ store.selected?.name || '未选择知识库' }}</strong></div>
      </header>

      <template v-if="store.selected">
        <section class="content-header">
          <div>
            <span class="eyebrow">KNOWLEDGE BASE</span>
            <h1>{{ store.selected.name }}</h1>
            <p>{{ store.selected.description || '管理文档并基于资料进行问答。' }}</p>
          </div>
          <button class="button primary" @click="createDialogOpen = true"><Plus :size="16" />新建知识库</button>
        </section>

        <nav class="view-tabs" aria-label="知识库视图">
          <button :class="{ active: activeTab === 'documents' }" @click="activeTab = 'documents'"><FileStack :size="16" />文档</button>
          <button :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'"><MessageSquareText :size="16" />问答</button>
        </nav>

        <section v-if="activeTab === 'documents'" class="document-view">
          <ImportPanel :kb-id="store.selected.id" @completed="refreshDocuments" />
          <div class="section-heading">
            <div><h2>文档</h2><span>{{ documentCountText }}</span></div>
            <label class="table-search">
              <Search :size="15" />
              <input v-model="documentQuery" type="search" placeholder="搜索文档" aria-label="按标题或文件名搜索文档" />
              <button v-if="documentQuery" type="button" title="清空搜索" aria-label="清空搜索" @click="documentQuery = ''">
                <X :size="14" />
              </button>
            </label>
          </div>
          <DocumentList
            :documents="filteredDocuments"
            :loading="store.loading"
            :empty-message="documentQuery.trim() ? `没有找到包含“${documentQuery.trim()}”的文档` : ''"
          />
        </section>

        <ChatWorkspace v-else :key="store.selected.id" :kb-id="store.selected.id" />
      </template>

      <section v-else class="empty-workspace">
        <div class="empty-mark"><FileStack :size="26" /></div>
        <h1>创建你的第一个知识库</h1>
        <p>把文档集中到一个可检索、可引用的工作空间。</p>
        <button class="button primary" @click="createDialogOpen = true"><Plus :size="16" />新建知识库</button>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>
      </section>
    </main>

    <CreateKnowledgeBaseDialog v-if="createDialogOpen" :saving="creating" @close="createDialogOpen = false" @submit="createKnowledgeBase" />
  </div>
</template>
