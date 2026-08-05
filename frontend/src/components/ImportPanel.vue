<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { AlertTriangle, ChevronRight, FileUp, Folder, FolderOpen, LoaderCircle, UploadCloud, X } from 'lucide-vue-next'
import { getDocument, uploadDocument } from '../services/api'
import { SUPPORTED_DOCUMENT_ACCEPT, documentExtension, isSupportedDocument } from '../utils/supportedDocuments'

const props = defineProps({ kbId: { type: String, required: true } })
const emit = defineEmits(['started', 'completed'])
const fileInput = ref(null)
const files = ref([])
const importing = ref(false)
const current = ref(0)
const error = ref('')
const failed = ref([])
const skippedCount = ref(0)
const skippedTypes = ref([])
const importedPaths = ref(new Set())
const skippedDuplicateCount = ref(0)
const recoveryMessage = ref('')
const importGeneration = ref(0)
const expandedFolders = ref(new Set())

// 每个文件的状态: queued | uploading | done | skipped | failed
const fileStatus = ref({})
const fileErrors = ref({})
const MAX_CONCURRENT = 2
const DOCUMENT_POLL_INTERVAL_MS = 1500

const SESSION_KEY_PREFIX = 'rag-import-queue'

function sessionKey(kbId) {
  return `${SESSION_KEY_PREFIX}:${kbId}`
}

const progress = computed(() => {
  const total = files.value.length
  if (!total) return 0
  const done = Object.values(fileStatus.value).filter(s => s === 'done' || s === 'skipped').length
  return Math.round((done / total) * 100)
})

const queueCounts = computed(() => {
  const counts = { queued: 0, uploading: 0, done: 0, skipped: 0, failed: 0 }
  Object.values(fileStatus.value).forEach((status) => {
    if (status in counts) counts[status]++
  })
  return counts
})

const queueRows = computed(() => {
  const root = { folders: new Map(), files: [] }

  files.value.forEach((file, index) => {
    const relativePath = (file.webkitRelativePath || file.name).replaceAll('\\', '/')
    const parts = relativePath.split('/').filter(Boolean)
    const folderNames = parts.slice(0, -1)
    let node = root
    const pathParts = []

    folderNames.forEach((name) => {
      pathParts.push(name)
      if (!node.folders.has(name)) {
        node.folders.set(name, {
          id: `folder:${pathParts.join('/')}`,
          name,
          path: pathParts.join('/'),
          folders: new Map(),
          files: [],
        })
      }
      node = node.folders.get(name)
    })

    node.files.push({
      id: `file:${index}:${relativePath}`,
      type: 'file',
      file,
      index,
      name: file.name,
      path: relativePath,
    })
  })

  function summarize(node) {
    const counts = { queued: 0, uploading: 0, done: 0, skipped: 0, failed: 0 }
    const fileIndices = []

    node.folders.forEach((folder) => {
      summarize(folder)
      Object.keys(counts).forEach((status) => { counts[status] += folder.counts[status] })
      fileIndices.push(...folder.fileIndices)
    })
    node.files.forEach((entry) => {
      const status = fileStatus.value[entry.index] || 'queued'
      counts[status]++
      fileIndices.push(entry.index)
    })

    node.counts = counts
    node.fileIndices = fileIndices
    node.fileCount = fileIndices.length
  }

  summarize(root)
  const rows = []

  function appendRows(node, depth) {
    node.folders.forEach((folder) => {
      rows.push({ ...folder, type: 'folder', depth })
      if (expandedFolders.value.has(folder.id)) {
        appendRows(folder, depth + 1)
      }
    })
    node.files.forEach((entry) => rows.push({ ...entry, depth }))
  }

  appendRows(root, 0)
  return rows
})

function queueStatusLabel(status) {
  return {
    queued: '等待导入',
    uploading: '导入中',
    done: '已完成',
    skipped: '已跳过',
    failed: '导入失败',
  }[status] || '等待导入'
}

function folderStatus(folder) {
  if (folder.counts.uploading) return 'uploading'
  if (folder.counts.failed) return 'failed'
  if (folder.counts.done + folder.counts.skipped === folder.fileCount) {
    return folder.counts.done ? 'done' : 'skipped'
  }
  return 'queued'
}

function folderSummary(folder) {
  const parts = [`${folder.fileCount} 个文件`]
  if (folder.counts.queued) parts.push(`等待 ${folder.counts.queued}`)
  if (folder.counts.uploading) parts.push(`导入中 ${folder.counts.uploading}`)
  if (folder.counts.done) parts.push(`完成 ${folder.counts.done}`)
  if (folder.counts.skipped) parts.push(`跳过 ${folder.counts.skipped}`)
  if (folder.counts.failed) parts.push(`失败 ${folder.counts.failed}`)
  return parts.join(' · ')
}

function toggleFolder(folderId) {
  const next = new Set(expandedFolders.value)
  if (next.has(folderId)) next.delete(folderId)
  else next.add(folderId)
  expandedFolders.value = next
}

// beforeunload 始终注册，通过 importing 状态决定是否拦截
function beforeUnloadHandler(event) {
  if (importing.value) {
    event.preventDefault()
    // 现代浏览器忽略自定义文案，但设置 returnValue 触发默认对话框
    event.returnValue = ''
    return ''
  }
}

// ===== sessionStorage 持久化（按知识库分 key） =====
function saveQueueSnapshot(kbId) {
  const key = sessionKey(kbId || props.kbId)
  const snapshot = {
    kbId: kbId || props.kbId,
    count: files.value.length,
    importing: importing.value,
    files: files.value.map(f => ({
      name: f.name,
      path: f.webkitRelativePath || f.name,
      size: f.size,
    })),
    timestamp: Date.now(),
  }
  try { sessionStorage.setItem(key, JSON.stringify(snapshot)) } catch { /* quota exceeded */ }
}

function clearQueueSnapshot(kbId) {
  try { sessionStorage.removeItem(sessionKey(kbId || props.kbId)) } catch { /* ignore */ }
}

function loadQueueSnapshot(kbId) {
  try {
    const raw = sessionStorage.getItem(sessionKey(kbId || props.kbId))
    if (!raw) return null
    const snapshot = JSON.parse(raw)
    // 只认 1 小时内的快照
    if (Date.now() - snapshot.timestamp > 3600000) {
      sessionStorage.removeItem(sessionKey(kbId || props.kbId))
      return null
    }
    return snapshot
  } catch {
    sessionStorage.removeItem(sessionKey(kbId || props.kbId))
    return null
  }
}

onMounted(() => {
  window.addEventListener('beforeunload', beforeUnloadHandler)
  const snapshot = loadQueueSnapshot()
  if (snapshot && snapshot.count > 0 && snapshot.kbId === props.kbId) {
    const status = snapshot.importing ? '导入中断' : '已选择但未导入'
    recoveryMessage.value = `检测到上次${status}：${snapshot.count} 个文件（${new Date(snapshot.timestamp).toLocaleString()}）。请重新选择文件继续导入。`
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', beforeUnloadHandler)
  clearQueueSnapshot()
})

function resetState() {
  importing.value = false
  importGeneration.value++
  files.value = []
  fileStatus.value = {}
  fileErrors.value = {}
  error.value = ''
  failed.value = []
  current.value = 0
  importedPaths.value = new Set()
  skippedCount.value = 0
  skippedDuplicateCount.value = 0
  skippedTypes.value = []
  recoveryMessage.value = ''
  expandedFolders.value = new Set()
  if (fileInput.value) fileInput.value.value = ''
}

watch(() => props.kbId, (newId, oldId) => {
  // 离开旧知识库：保存快照到 sessionStorage
  if (oldId && files.value.length) {
    saveQueueSnapshot(oldId)
  }
  // 重置面板
  resetState()
  // 进入新知识库：尝试恢复快照
  const snapshot = loadQueueSnapshot(newId)
  if (snapshot && snapshot.count > 0 && snapshot.kbId === newId) {
    const status = snapshot.importing ? '导入中断' : '已选择但未导入'
    recoveryMessage.value = `检测到上次${status}：${snapshot.count} 个文件（${new Date(snapshot.timestamp).toLocaleString()}）。请重新选择文件继续导入。`
  }
})

function isSystemFile(file) {
  const path = (file.webkitRelativePath || file.name).replaceAll('\\', '/').toLowerCase()
  const name = file.name.toLowerCase()
  return name === '.ds_store' || name.startsWith('._') || path.includes('/__macosx/')
}

function folderPathFor(file) {
  const relative = file.webkitRelativePath || ''
  if (!relative) return ''
  const parts = relative.replaceAll('\\', '/').split('/')
  if (parts.length <= 1) return ''
  return parts.slice(0, -1).join('/')
}

function fileIdentity(file) {
  const path = folderPathFor(file)
  return path ? `${path}/${file.name}` : file.name
}

function chooseFiles(event) {
  const selected = Array.from(event.target.files || [])
  const typeCounts = new Map()
  let duplicateCount = 0
  files.value = selected.filter((file) => {
    const extension = documentExtension(file.name)
    if (isSystemFile(file) || !isSupportedDocument(file.name)) {
      const label = isSystemFile(file) ? '系统文件' : (extension || '无扩展名')
      typeCounts.set(label, (typeCounts.get(label) || 0) + 1)
      return false
    }
    if (importedPaths.value.has(fileIdentity(file))) {
      duplicateCount++
      return false
    }
    return true
  })
  skippedCount.value = selected.length - files.value.length
  skippedDuplicateCount.value = duplicateCount
  skippedTypes.value = Array.from(typeCounts.entries())
    .sort((left, right) => right[1] - left[1])
    .slice(0, 8)
    .map(([type, count]) => `${type} ${count}`)
  current.value = 0
  failed.value = []
  fileStatus.value = {}
  fileErrors.value = {}
  expandedFolders.value = new Set()
  files.value.forEach((_, idx) => { fileStatus.value[idx] = 'queued' })
  error.value = files.value.length ? '' : (duplicateCount > 0 ? '所选文件均已导入，无需重复导入' : '所选文件夹中没有可导入的文档')
  recoveryMessage.value = ''
  saveQueueSnapshot()
}

function removeFiles(indices) {
  const removed = new Set(indices)
  const remaining = []
  const newStatus = {}
  const newErrors = {}

  files.value.forEach((file, oldIndex) => {
    if (removed.has(oldIndex)) return
    const newIndex = remaining.length
    remaining.push(file)
    newStatus[newIndex] = fileStatus.value[oldIndex] || 'queued'
    if (fileErrors.value[oldIndex]) {
      newErrors[newIndex] = fileErrors.value[oldIndex]
    }
  })

  files.value = remaining
  fileStatus.value = newStatus
  fileErrors.value = newErrors
  if (files.value.length) {
    saveQueueSnapshot()
  } else {
    clearQueueSnapshot()
  }
}

function removeFile(index) {
  removeFiles([index])
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function waitForDocumentCompletion(knowledgeBaseId, documentId, generation) {
  while (importGeneration.value === generation && props.kbId === knowledgeBaseId) {
    const document = await getDocument(knowledgeBaseId, documentId)
    if (document.status === 'ready') return document
    if (document.status === 'failed') {
      throw new Error(document.error_message || '文档后台处理失败')
    }
    await wait(DOCUMENT_POLL_INTERVAL_MS)
  }
  return null
}

async function uploadOne(knowledgeBaseId, file, index, generation) {
  fileStatus.value[index] = 'uploading'
  try {
    const result = await uploadDocument(knowledgeBaseId, file, folderPathFor(file))
    if (result?.status === 'skipped') {
      fileStatus.value[index] = 'skipped'
      fileErrors.value[index] = null
      skippedCount.value++
      skippedDuplicateCount.value++
      importedPaths.value.add(fileIdentity(file))
      return
    }
    if (result?.status === 'processing' && result.id) {
      const completed = await waitForDocumentCompletion(knowledgeBaseId, result.id, generation)
      if (!completed) return
    }
    if (importGeneration.value !== generation || props.kbId !== knowledgeBaseId) return
    fileStatus.value[index] = 'done'
    fileErrors.value[index] = null
    importedPaths.value.add(fileIdentity(file))
  } catch (cause) {
    if (cause?.status === 409) {
      fileStatus.value[index] = 'skipped'
      fileErrors.value[index] = null
      skippedCount.value++
      skippedDuplicateCount.value++
      importedPaths.value.add(fileIdentity(file))
      return
    }
    fileStatus.value[index] = 'failed'
    fileErrors.value[index] = cause instanceof Error ? cause.message : '导入失败'
    failed.value.push({ name: file.name, message: cause instanceof Error ? cause.message : '导入失败' })
  }
}

async function startImport() {
  if (!files.value.length || importing.value) return
  importing.value = true
  const gen = importGeneration.value
  emit('started')
  const knowledgeBaseId = props.kbId
  failed.value = []

  files.value.forEach((_, idx) => {
    fileStatus.value[idx] = 'queued'
    fileErrors.value[idx] = null
  })

  saveQueueSnapshot()
  const pending = files.value.map((file, index) => ({ file, index }))
  let nextPendingIndex = 0

  async function runWorker() {
    while (importGeneration.value === gen && nextPendingIndex < pending.length) {
      const item = pending[nextPendingIndex]
      nextPendingIndex++
      await uploadOne(knowledgeBaseId, item.file, item.index, gen)
      current.value++
      saveQueueSnapshot()
    }
  }

  const workerCount = Math.min(MAX_CONCURRENT, pending.length)
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()))

  // 如果知识库已切换，不再更新后续状态
  if (importGeneration.value !== gen) return

  importing.value = false

  const failedIndices = []
  files.value.forEach((_, idx) => {
    if (fileStatus.value[idx] === 'failed') {
      failedIndices.push(idx)
    }
  })
  const remaining = failedIndices.map(i => files.value[i])
  files.value = remaining
  fileStatus.value = {}
  fileErrors.value = {}
  files.value.forEach((_, idx) => { fileStatus.value[idx] = 'queued' })
  current.value = 0

  if (!remaining.length) {
    if (fileInput.value) fileInput.value.value = ''
    clearQueueSnapshot()
  } else {
    saveQueueSnapshot()
    error.value = ''
  }
  emit('completed')
}
</script>

<template>
  <section class="import-panel">
    <div class="import-copy">
      <div class="section-icon"><FolderOpen :size="19" /></div>
      <div>
        <h3>批量导入资料</h3>
        <p>选择文件后，系统会逐个解析并写入知识库。</p>
      </div>
    </div>
    <input
      ref="fileInput"
      class="hidden-input"
      type="file"
      multiple
      :accept="SUPPORTED_DOCUMENT_ACCEPT"
      @change="chooseFiles"
    />
    <div class="import-actions">
      <button class="button secondary" :disabled="importing" @click="fileInput?.click()"><FileUp :size="16" />选择文件</button>
    </div>

    <div v-if="skippedCount" class="skip-summary">
      <strong>已跳过 {{ skippedCount }} 个文件</strong>
      <span v-if="skippedDuplicateCount">（其中 {{ skippedDuplicateCount }} 个重复文件）</span>
      <span>{{ skippedTypes.join(' · ') }}</span>
    </div>

    <div v-if="recoveryMessage" class="recovery-banner">
      <AlertTriangle :size="16" />
      <span>{{ recoveryMessage }}</span>
      <button class="icon-button subtle" title="关闭" @click="recoveryMessage = ''"><X :size="14" /></button>
    </div>

    <div v-if="importing" class="import-warning">
      <AlertTriangle :size="16" />
      <span>导入进行中，请勿刷新或关闭页面，否则未完成的文件将丢失。</span>
    </div>

    <div v-if="files.length" class="import-queue">
      <div class="queue-header">
        <strong>{{ files.length }} 个文件</strong>
        <span class="queue-summary">
          等待 {{ queueCounts.queued }}
          <template v-if="importing || queueCounts.uploading">· 导入中 {{ queueCounts.uploading }}</template>
          <template v-if="queueCounts.done">· 完成 {{ queueCounts.done }}</template>
          <template v-if="queueCounts.skipped">· 跳过 {{ queueCounts.skipped }}</template>
          <template v-if="queueCounts.failed">· 失败 {{ queueCounts.failed }}</template>
        </span>
      </div>
      <div class="progress-track"><span :style="{ width: `${progress}%` }" /></div>
      <div class="queue-list">
        <div
          v-for="row in queueRows"
          :key="row.id"
          class="queue-item"
          :class="{
            'queue-folder': row.type === 'folder',
            'queue-item-done': row.type === 'folder' ? ['done', 'skipped'].includes(folderStatus(row)) : ['done', 'skipped'].includes(fileStatus[row.index]),
            'queue-item-failed': row.type === 'folder' ? folderStatus(row) === 'failed' : fileStatus[row.index] === 'failed',
          }"
          :style="{ '--queue-depth': row.depth }"
        >
          <template v-if="row.type === 'folder'">
            <button
              class="queue-folder-main"
              type="button"
              :aria-expanded="expandedFolders.has(row.id)"
              :title="row.path"
              @click="toggleFolder(row.id)"
            >
              <ChevronRight :size="14" class="queue-folder-chevron" :class="{ expanded: expandedFolders.has(row.id) }" />
              <Folder :size="15" />
              <span class="queue-item-name">{{ row.name }}</span>
            </button>
            <span class="queue-folder-summary" :data-status="folderStatus(row)">{{ folderSummary(row) }}</span>
            <button v-if="!importing" class="icon-button subtle" title="移除文件夹" @click="removeFiles(row.fileIndices)"><X :size="14" /></button>
          </template>
          <template v-else>
            <span class="queue-item-icon">
              <LoaderCircle v-if="fileStatus[row.index] === 'uploading'" :size="14" class="spinning" />
              <span v-else-if="fileStatus[row.index] === 'done'" class="icon-done">✓</span>
              <span v-else-if="fileStatus[row.index] === 'skipped'" class="icon-queued">—</span>
              <span v-else-if="fileStatus[row.index] === 'failed'" class="icon-fail">✗</span>
              <span v-else class="icon-queued">—</span>
            </span>
            <span class="queue-item-name" :title="row.path">{{ row.name }}</span>
            <span class="queue-item-status" :data-status="fileStatus[row.index] || 'queued'">
              {{ queueStatusLabel(fileStatus[row.index]) }}
            </span>
            <span v-if="fileErrors[row.index]" class="queue-item-error">{{ fileErrors[row.index] }}</span>
            <button v-if="!importing" class="icon-button subtle" title="移除" @click="removeFile(row.index)"><X :size="14" /></button>
          </template>
        </div>
      </div>
      <button class="button primary import-button" :disabled="importing" @click="startImport">
        <UploadCloud :size="16" />{{ importing ? `导入中 ${progress}%` : '开始导入' }}
      </button>
      <div v-if="failed.length" class="error-list">
        <strong>{{ failed.length }} 个文件导入失败</strong>
        <span v-for="item in failed" :key="item.name">{{ item.name }}：{{ item.message }}</span>
      </div>
    </div>
    <p v-if="error" class="error-text">{{ error }}</p>
  </section>
</template>
