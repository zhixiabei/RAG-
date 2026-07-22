<script setup>
import { onMounted, ref } from 'vue'
import { FileStack, MessageSquareText, Plus, Search, Server } from 'lucide-vue-next'
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

onMounted(() => store.load())

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
        <div class="topbar-status"><span class="online-dot" />本机服务 <Server :size="15" /></div>
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
            <div><h2>文档</h2><span>{{ store.documents.length }} 个文件</span></div>
            <div class="table-search"><Search :size="15" /><input placeholder="搜索文档" /></div>
          </div>
          <DocumentList :documents="store.documents" :loading="store.loading" />
        </section>

        <ChatWorkspace v-else :kb-id="store.selected.id" />
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

