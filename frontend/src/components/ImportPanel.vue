<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { AlertTriangle, FileUp, FolderOpen, LoaderCircle, UploadCloud, X } from 'lucide-vue-next'
import { uploadDocument } from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const emit = defineEmits(['started', 'completed'])
const fileInput = ref(null)
const folderInput = ref(null)
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

// 每个文件的状态: queued | uploading | done | failed
const fileStatus = ref({})
const fileErrors = ref({})
const MAX_CONCURRENT = 3
const MAX_RETRIES = 2

const SESSION_KEY_PREFIX = 'rag-import-queue'

function sessionKey(kbId) {
  return `${SESSION_KEY_PREFIX}:${kbId}`
}

const progress = computed(() => {
  const total = files.value.length
  if (!total) return 0
  const done = Object.values(fileStatus.value).filter(s => s === 'done').length
  return Math.round((done / total) * 100)
})

const queueCounts = computed(() => {
  const counts = { queued: 0, uploading: 0, done: 0, failed: 0 }
  Object.values(fileStatus.value).forEach((status) => {
    if (status in counts) counts[status]++
  })
  return counts
})

function queueStatusLabel(status) {
  return {
    queued: '等待导入',
    uploading: '导入中',
    done: '已完成',
    failed: '导入失败',
  }[status] || '等待导入'
}

const SUPPORTED_EXTENSIONS = new Set([
  '', '.pdf', '.doc', '.docx', '.pptx', '.xlsx', '.xlsm', '.xls', '.csv',
  '.md', '.markdown', '.txt', '.html', '.htm', '.xml', '.json',
  '.dll', '.gdb', '.att', '.ptpt', '.jcpt', '.stpt', '.ppt', '.lst',
])

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
    recoveryMessage.value = `检测到上次${status}：${snapshot.count} 个文件（${new Date(snapshot.timestamp).toLocaleString()}）。请重新选择文件夹继续导入。`
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
  if (fileInput.value) fileInput.value.value = ''
  if (folderInput.value) folderInput.value.value = ''
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
    recoveryMessage.value = `检测到上次${status}：${snapshot.count} 个文件（${new Date(snapshot.timestamp).toLocaleString()}）。请重新选择文件夹继续导入。`
  }
})

function fileExtension(name) {
  const index = name.lastIndexOf('.')
  return index > 0 ? name.slice(index).toLowerCase() : ''
}

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

function displayName(file) {
  return file.webkitRelativePath || file.name
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
    const extension = fileExtension(file.name)
    if (isSystemFile(file) || !SUPPORTED_EXTENSIONS.has(extension)) {
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
  files.value.forEach((_, idx) => { fileStatus.value[idx] = 'queued' })
  error.value = files.value.length ? '' : (duplicateCount > 0 ? '所选文件均已导入，无需重复导入' : '所选文件夹中没有可导入的文档')
  recoveryMessage.value = ''
  saveQueueSnapshot()
}

function removeFile(index) {
  files.value.splice(index, 1)
  const newStatus = {}
  const newErrors = {}
  files.value.forEach((_, idx) => {
    newStatus[idx] = fileStatus.value[idx >= index ? idx + 1 : idx] || 'queued'
    if (fileErrors.value[idx >= index ? idx + 1 : idx]) {
      newErrors[idx] = fileErrors.value[idx >= index ? idx + 1 : idx]
    }
  })
  fileStatus.value = newStatus
  fileErrors.value = newErrors
  if (files.value.length) {
    saveQueueSnapshot()
  } else {
    clearQueueSnapshot()
  }
}

async function uploadOne(knowledgeBaseId, file, index) {
  fileStatus.value[index] = 'uploading'
  let lastError = null
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      await uploadDocument(knowledgeBaseId, file, folderPathFor(file))
      fileStatus.value[index] = 'done'
      fileErrors.value[index] = null
      importedPaths.value.add(fileIdentity(file))
      return
    } catch (cause) {
      lastError = cause
      if (cause.status && (cause.status < 500 || cause.status >= 600)) {
        break
      }
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1000 * (2 ** attempt)))
      }
    }
  }
  fileStatus.value[index] = 'failed'
  fileErrors.value[index] = lastError instanceof Error ? lastError.message : '导入失败'
  failed.value.push({ name: file.name, message: lastError instanceof Error ? lastError.message : '导入失败' })
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
      await uploadOne(knowledgeBaseId, item.file, item.index)
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
    if (folderInput.value) folderInput.value.value = ''
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
        <p>选择文件或文件夹后，系统会逐个解析并写入知识库。</p>
      </div>
    </div>
    <input
      ref="fileInput"
      class="hidden-input"
      type="file"
      multiple
      @change="chooseFiles"
    />
    <input
      ref="folderInput"
      class="hidden-input"
      type="file"
      multiple
      webkitdirectory
      directory
      @change="chooseFiles"
    />
    <div class="import-actions">
      <button class="button secondary" :disabled="importing" @click="fileInput?.click()"><FileUp :size="16" />选择文件</button>
      <button class="button secondary" :disabled="importing" @click="folderInput?.click()"><FolderOpen :size="16" />选择文件夹</button>
    </div>

    <div v-if="skippedCount" class="skip-summary">
      <strong>已跳过 {{ skippedCount }} 个文件</strong>
      <span v-if="skippedDuplicateCount">（其中 {{ skippedDuplicateCount }} 个已导入）</span>
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
          <template v-if="queueCounts.failed">· 失败 {{ queueCounts.failed }}</template>
        </span>
      </div>
      <div class="progress-track"><span :style="{ width: `${progress}%` }" /></div>
      <div class="queue-list">
        <div
          v-for="(file, index) in files"
          :key="`${displayName(file)}-${index}`"
          class="queue-item"
          :class="{ 'queue-item-done': fileStatus[index] === 'done', 'queue-item-failed': fileStatus[index] === 'failed' }"
        >
          <span class="queue-item-icon">
            <LoaderCircle v-if="fileStatus[index] === 'uploading'" :size="14" class="spinning" />
            <span v-else-if="fileStatus[index] === 'done'" class="icon-done">✓</span>
            <span v-else-if="fileStatus[index] === 'failed'" class="icon-fail">✗</span>
            <span v-else class="icon-queued">—</span>
          </span>
          <span class="queue-item-name">{{ displayName(file) }}</span>
          <span class="queue-item-status" :data-status="fileStatus[index] || 'queued'">
            {{ queueStatusLabel(fileStatus[index]) }}
          </span>
          <span v-if="fileErrors[index]" class="queue-item-error">{{ fileErrors[index] }}</span>
          <button v-if="!importing" class="icon-button subtle" title="移除" @click="removeFile(index)"><X :size="14" /></button>
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
