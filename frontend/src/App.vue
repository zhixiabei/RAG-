<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { AlertTriangle, CheckCircle2, FileStack, FlaskConical, LoaderCircle, MessageSquareText, Plus, RefreshCw, Search, X } from 'lucide-vue-next'
import KnowledgeSidebar from './components/KnowledgeSidebar.vue'
import CreateKnowledgeBaseDialog from './components/CreateKnowledgeBaseDialog.vue'
import DeleteConfirmDialog from './components/DeleteConfirmDialog.vue'
import DocumentList from './components/DocumentList.vue'
import ImportPanel from './components/ImportPanel.vue'
import ChatWorkspace from './components/ChatWorkspace.vue'
import EvaluationDialog from './components/EvaluationDialog.vue'
import { evaluateKnowledgeBase, syncKnowledgeBaseToTestset } from './services/api'
import { useKnowledgeBaseStore } from './stores/knowledgeBase'

const store = useKnowledgeBaseStore()
const activeTab = ref('chat')
const sidebarCollapsed = ref(localStorage.getItem('rag-sidebar-collapsed') === 'true')
const createDialogOpen = ref(false)
const creating = ref(false)
const documentQuery = ref('')
const deleteTarget = ref(null)
const deletingId = ref(null)
const deleteError = ref('')
const appLoading = ref(true)
const importActive = ref(false)
const evaluationOpen = ref(false)
const evaluationBusy = ref(false)
const evaluationError = ref('')
const evaluationResult = ref(null)
const testsetSyncBusy = ref(false)
const testsetSyncNotice = ref(null)
let documentPollTimer = null

const testsetSyncTitle = computed(() => {
  if (store.hasProcessingDocuments) return '文档处理完成后可同步到测试集工坊'
  return '将当前知识库的文档和 Chunk 同步到测试集工坊'
})

const deleteDialogTitle = computed(() => {
  if (deleteTarget.value?.type === 'knowledge-base') return '删除知识库'
  if (deleteTarget.value?.type === 'document-folder') return '删除文件夹'
  return '删除文档'
})

const deleteDialogMessage = computed(() => {
  const target = deleteTarget.value
  if (!target) return ''
  if (target.type === 'knowledge-base') {
    return `知识库“${target.item.name}”中的文档、向量和历史对话都会被永久删除。`
  }
  if (target.type === 'document-folder') {
    return `文件夹“${target.item.name}”及其子文件夹中的 ${target.item.documentCount} 个文档、原文件和向量索引都会被永久删除。`
  }
  return `文档“${target.item.title}”及其原文件和向量索引都会被永久删除。`
})

const filteredDocuments = computed(() => {
  const terms = documentQuery.value
    .trim()
    .toLocaleLowerCase('zh-CN')
    .split(/\s+/)
    .filter(Boolean)

  if (!terms.length) return store.documents

  return store.documents.filter((document) => {
    const searchableText = `${document.title || ''} ${document.file_name || ''} ${document.folder_path || ''}`.toLocaleLowerCase('zh-CN')
    return terms.every((term) => searchableText.includes(term))
  })
})

const documentCountText = computed(() => {
  if (!documentQuery.value.trim()) return `${store.documents.length} 个文件`
  return `${filteredDocuments.value.length} / ${store.documents.length} 个文件`
})

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem('rag-sidebar-collapsed', String(sidebarCollapsed.value))
}

function stopDocumentPolling() {
  if (documentPollTimer) clearTimeout(documentPollTimer)
  documentPollTimer = null
}

function scheduleDocumentPolling() {
  stopDocumentPolling()
  if (!store.selectedId || (!importActive.value && !store.hasProcessingDocuments)) return
  const knowledgeBaseId = store.selectedId
  documentPollTimer = setTimeout(async () => {
    try {
      await store.loadDocuments(knowledgeBaseId)
    } catch {
      // The next poll or manual refresh will surface persistent errors.
    } finally {
      scheduleDocumentPolling()
    }
  }, 2000)
}

async function initializeApp() {
  try {
    await store.load()
  } finally {
    appLoading.value = false
  }
}

onMounted(() => {
  initializeApp()
})

onBeforeUnmount(() => {
  stopDocumentPolling()
})

watch(() => store.selectedId, () => {
  documentQuery.value = ''
  evaluationOpen.value = false
  testsetSyncNotice.value = null
})

watch(
  [
    () => store.selectedId,
    () => store.hasProcessingDocuments,
    () => importActive.value,
  ],
  scheduleDocumentPolling,
)

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
  importActive.value = false
  await store.loadDocuments()
}

function trackImport() {
  importActive.value = true
  scheduleDocumentPolling()
}

function openEvaluation() {
  evaluationError.value = ''
  evaluationResult.value = null
  evaluationOpen.value = true
}

function closeEvaluation() {
  if (evaluationBusy.value) return
  evaluationOpen.value = false
}

async function runEvaluation(questionIds, source, datasetId) {
  if (!store.selectedId || evaluationBusy.value) return
  evaluationBusy.value = true
  evaluationError.value = ''
  evaluationResult.value = null
  try {
    evaluationResult.value = await evaluateKnowledgeBase(store.selectedId, questionIds, source, datasetId)
  } catch (cause) {
    evaluationError.value = cause instanceof Error ? cause.message : '评测失败'
  } finally {
    evaluationBusy.value = false
  }
}

async function syncTestsetWorkshop() {
  if (!store.selectedId || testsetSyncBusy.value || store.hasProcessingDocuments) return
  testsetSyncBusy.value = true
  testsetSyncNotice.value = null
  try {
    const result = await syncKnowledgeBaseToTestset(store.selectedId)
    const syncedDocuments = Number(result.synced_document_count || 0)
    const syncedChunks = Number(result.synced_chunk_count || 0)
    const failedDocuments = Number(result.failed_document_count || 0)
    const skippedDocuments = Number(result.skipped_document_count || 0)
    const extras = []
    if (failedDocuments) extras.push(`失败 ${failedDocuments} 个`)
    if (skippedDocuments) extras.push(`跳过 ${skippedDocuments} 个`)
    testsetSyncNotice.value = {
      tone: failedDocuments ? 'warning' : 'success',
      message: `${failedDocuments ? '部分同步完成' : '同步完成'}：${syncedDocuments} 个文档，${syncedChunks} 个 Chunk${extras.length ? `；${extras.join('，')}` : ''}`,
    }
  } catch (cause) {
    testsetSyncNotice.value = {
      tone: 'error',
      message: cause instanceof Error ? cause.message : '同步测试集工坊失败',
    }
  } finally {
    testsetSyncBusy.value = false
  }
}

function requestKnowledgeBaseDelete(item) {
  deleteError.value = ''
  deleteTarget.value = { type: 'knowledge-base', item }
}

function requestDocumentDelete(item) {
  deleteError.value = ''
  deleteTarget.value = { type: 'document', item, knowledgeBaseId: store.selectedId }
}

function requestDocumentFolderDelete(item) {
  deleteError.value = ''
  deleteTarget.value = { type: 'document-folder', item, knowledgeBaseId: store.selectedId }
}

function closeDeleteDialog() {
  if (deletingId.value) return
  deleteTarget.value = null
  deleteError.value = ''
}

async function confirmDelete() {
  if (!deleteTarget.value || deletingId.value) return
  const target = deleteTarget.value
  deletingId.value = target.item.id
  deleteError.value = ''
  try {
    if (target.type === 'knowledge-base') {
      await store.removeKnowledgeBase(target.item.id)
    } else if (target.type === 'document-folder') {
      await store.removeDocumentFolder(target.knowledgeBaseId, target.item.path)
    } else {
      await store.removeDocument(target.knowledgeBaseId, target.item.id)
    }
    deleteTarget.value = null
  } catch (cause) {
    deleteError.value = cause instanceof Error ? cause.message : '删除失败'
  } finally {
    deletingId.value = null
  }
}
</script>

<template>
  <main v-if="appLoading" class="app-loading-shell">
    <LoaderCircle :size="24" class="spinning" />
  </main>

  <div v-else class="app-shell">
    <KnowledgeSidebar
      :items="store.items"
      :selected-id="store.selectedId"
      :loading="store.loading"
      :collapsed="sidebarCollapsed"
      :deleting-id="deleteTarget?.type === 'knowledge-base' ? deletingId : null"
      @select="store.select"
      @create="createDialogOpen = true"
      @refresh="store.load"
      @delete="requestKnowledgeBaseDelete"
      @toggle="toggleSidebar"
    />

    <main class="main-area">
      <header class="topbar">
        <div class="breadcrumb"><span>工作区</span><span class="breadcrumb-separator">/</span><strong>{{ store.selected?.name || '未选择知识库' }}</strong></div>
        <nav v-if="store.selected" class="workspace-switcher" aria-label="知识库视图">
          <button :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'"><MessageSquareText :size="15" />对话</button>
          <button :class="{ active: activeTab === 'documents' }" @click="activeTab = 'documents'"><FileStack :size="15" />文档</button>
        </nav>
        <div v-if="store.selected" class="topbar-actions">
          <button
            class="topbar-action-button"
            type="button"
            :title="testsetSyncTitle"
            :disabled="testsetSyncBusy || store.hasProcessingDocuments"
            @click="syncTestsetWorkshop"
          >
            <LoaderCircle v-if="testsetSyncBusy" :size="15" class="spinning" />
            <RefreshCw v-else :size="15" />
            <span class="topbar-action-label">{{ testsetSyncBusy ? '同步中' : '同步工坊' }}</span>
          </button>
          <button
            class="topbar-action-button"
            type="button"
            title="选择测试集并运行评测"
            @click="openEvaluation"
          >
            <FlaskConical :size="15" />
            <span class="topbar-action-label">测试</span>
          </button>
        </div>
      </header>

      <div
        v-if="testsetSyncNotice"
        class="testset-sync-notice"
        :class="`notice-${testsetSyncNotice.tone}`"
        role="status"
        aria-live="polite"
      >
        <CheckCircle2 v-if="testsetSyncNotice.tone === 'success'" :size="16" />
        <AlertTriangle v-else :size="16" />
        <span>{{ testsetSyncNotice.message }}</span>
        <button type="button" title="关闭同步结果" aria-label="关闭同步结果" @click="testsetSyncNotice = null">
          <X :size="14" />
        </button>
      </div>

      <template v-if="store.selected">
        <section v-show="activeTab === 'documents'" class="content-header">
          <div>
            <span class="eyebrow">KNOWLEDGE BASE</span>
            <h1>{{ store.selected.name }}</h1>
            <p>{{ store.selected.description || '管理文档并基于资料进行问答。' }}</p>
          </div>
          <button class="button primary" @click="createDialogOpen = true"><Plus :size="16" />新建知识库</button>
        </section>

        <section v-show="activeTab === 'documents'" class="document-view">
          <ImportPanel :kb-id="store.selected.id" @started="trackImport" @completed="refreshDocuments" />
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
            :key="store.selected.id"
            :documents="filteredDocuments"
            :loading="store.loading"
            :empty-message="documentQuery.trim() ? `没有找到包含“${documentQuery.trim()}”的文档` : ''"
            :deleting-id="deleteTarget?.type === 'knowledge-base' ? null : deletingId"
            :folder-actions="!documentQuery.trim()"
            @delete="requestDocumentDelete"
            @delete-folder="requestDocumentFolderDelete"
          />
        </section>

        <ChatWorkspace v-show="activeTab === 'chat'" :key="store.selected.id" :kb-id="store.selected.id" @documents-updated="refreshDocuments" />
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
    <DeleteConfirmDialog
      v-if="deleteTarget"
      :title="deleteDialogTitle"
      :message="deleteDialogMessage"
      :busy="Boolean(deletingId)"
      :error="deleteError"
      @close="closeDeleteDialog"
      @confirm="confirmDelete"
    />
    <EvaluationDialog
      v-if="evaluationOpen && store.selected"
      :kb-id="store.selected.id"
      :busy="evaluationBusy"
      :error="evaluationError"
      :result="evaluationResult"
      @close="closeEvaluation"
      @run="runEvaluation"
    />
  </div>
</template>
