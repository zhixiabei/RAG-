<script setup>
import { computed, ref } from 'vue'
import { FileUp, FolderOpen, UploadCloud, X } from 'lucide-vue-next'
import { uploadDocument } from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const emit = defineEmits(['completed'])
const fileInput = ref(null)
const folderInput = ref(null)
const files = ref([])
const importing = ref(false)
const current = ref(0)
const error = ref('')
const failed = ref([])
const skippedCount = ref(0)
const skippedTypes = ref([])
const progress = computed(() => files.value.length ? Math.round((current.value / files.value.length) * 100) : 0)

const SUPPORTED_EXTENSIONS = new Set([
  '', '.pdf', '.doc', '.docx', '.pptx', '.xlsx', '.xlsm', '.xls', '.csv',
  '.md', '.markdown', '.txt', '.html', '.htm', '.xml', '.json',
  '.dll', '.gdb', '.att', '.ptpt', '.jcpt', '.stpt',
])

function fileExtension(name) {
  const index = name.lastIndexOf('.')
  return index > 0 ? name.slice(index).toLowerCase() : ''
}

function isSystemFile(file) {
  const path = (file.webkitRelativePath || file.name).replaceAll('\\', '/').toLowerCase()
  const name = file.name.toLowerCase()
  return name === '.ds_store' || name.startsWith('._') || path.includes('/__macosx/')
}

function chooseFiles(event) {
  const selected = Array.from(event.target.files || [])
  const typeCounts = new Map()
  files.value = selected.filter((file) => {
    const extension = fileExtension(file.name)
    const supported = !isSystemFile(file) && SUPPORTED_EXTENSIONS.has(extension)
    if (!supported) {
      const label = isSystemFile(file) ? '系统文件' : (extension || '无扩展名')
      typeCounts.set(label, (typeCounts.get(label) || 0) + 1)
    }
    return supported
  })
  skippedCount.value = selected.length - files.value.length
  skippedTypes.value = Array.from(typeCounts.entries())
    .sort((left, right) => right[1] - left[1])
    .slice(0, 8)
    .map(([type, count]) => `${type} ${count}`)
  current.value = 0
  failed.value = []
  error.value = files.value.length ? '' : '所选文件夹中没有可导入的文档'
}

function removeFile(index) {
  files.value.splice(index, 1)
}

async function startImport() {
  if (!files.value.length || importing.value) return
  importing.value = true
  current.value = 0
  failed.value = []
  const failedFiles = []
  for (const file of files.value) {
    try {
      await uploadDocument(props.kbId, file)
    } catch (cause) {
      failedFiles.push(file)
      failed.value.push({ name: file.name, message: cause instanceof Error ? cause.message : '导入失败' })
    } finally {
      current.value += 1
    }
  }
  importing.value = false
  files.value = failedFiles
  current.value = 0
  if (!failedFiles.length) {
    if (fileInput.value) fileInput.value.value = ''
    if (folderInput.value) folderInput.value.value = ''
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
      <strong>已跳过 {{ skippedCount }} 个不可检索文件</strong>
      <span>{{ skippedTypes.join(' · ') }}</span>
    </div>

    <div v-if="files.length" class="import-queue">
      <div class="queue-header"><strong>{{ files.length }} 个文件待处理</strong><span>{{ importing ? `${current}/${files.length}` : '等待导入' }}</span></div>
      <div class="progress-track"><span :style="{ width: `${progress}%` }" /></div>
      <div class="queue-list">
        <div v-for="(file, index) in files" :key="`${file.name}-${index}`" class="queue-item">
          <span>{{ file.webkitRelativePath || file.name }}</span>
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
