<script setup>
import { computed, ref } from 'vue'
import { FolderOpen, UploadCloud, X } from 'lucide-vue-next'
import { uploadDocument } from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const emit = defineEmits(['completed'])
const input = ref(null)
const files = ref([])
const importing = ref(false)
const current = ref(0)
const error = ref('')
const failed = ref([])
const progress = computed(() => files.value.length ? Math.round((current.value / files.value.length) * 100) : 0)

function chooseFiles(event) {
  files.value = Array.from(event.target.files || [])
  current.value = 0
  failed.value = []
  error.value = ''
}

function removeFile(index) {
  files.value.splice(index, 1)
}

async function startImport() {
  if (!files.value.length || importing.value) return
  importing.value = true
  current.value = 0
  failed.value = []
  for (const file of files.value) {
    try {
      await uploadDocument(props.kbId, file)
    } catch (cause) {
      failed.value.push({ name: file.name, message: cause instanceof Error ? cause.message : '导入失败' })
    } finally {
      current.value += 1
    }
  }
  importing.value = false
  emit('completed')
}
</script>

<template>
  <section class="import-panel">
    <div class="import-copy">
      <div class="section-icon"><FolderOpen :size="19" /></div>
      <div>
        <h3>批量导入资料</h3>
        <p>选择文件夹后，系统会逐个解析并写入知识库。</p>
      </div>
    </div>
    <input ref="input" class="hidden-input" type="file" multiple webkitdirectory directory @change="chooseFiles" />
    <button class="button secondary" :disabled="importing" @click="input?.click()"><FolderOpen :size="16" />选择文件夹</button>

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

