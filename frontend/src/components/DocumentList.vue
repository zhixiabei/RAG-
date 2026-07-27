<script setup>
import { computed, ref } from 'vue'
import { ChevronRight, FileText, Folder, LoaderCircle, RefreshCw, Trash2 } from 'lucide-vue-next'

const props = defineProps({
  documents: { type: Array, default: () => [] },
  loading: Boolean,
  emptyMessage: { type: String, default: '' },
  deletingId: { type: String, default: null },
  folderActions: { type: Boolean, default: true },
})

const emit = defineEmits(['delete', 'delete-folder'])
const expandedFolders = ref(new Set())

const documentRows = computed(() => {
  const root = { folders: new Map(), documents: [] }

  props.documents.forEach((document) => {
    const folderNames = String(document.folder_path || '')
      .replaceAll('\\', '/')
      .split('/')
      .filter(Boolean)
    const pathParts = []
    let node = root

    folderNames.forEach((name) => {
      pathParts.push(name)
      if (!node.folders.has(name)) {
        node.folders.set(name, {
          id: `folder:${pathParts.join('/')}`,
          name,
          path: pathParts.join('/'),
          folders: new Map(),
          documents: [],
        })
      }
      node = node.folders.get(name)
    })

    node.documents.push(document)
  })

  function summarize(node) {
    const statusCounts = { ready: 0, processing: 0, failed: 0 }
    let documentCount = 0
    let chunkCount = 0
    let updatedAt = null

    node.folders.forEach((folder) => {
      summarize(folder)
      documentCount += folder.documentCount
      chunkCount += folder.chunkCount
      Object.keys(statusCounts).forEach((status) => { statusCounts[status] += folder.statusCounts[status] })
      if (folder.updatedAt && (!updatedAt || new Date(folder.updatedAt) > new Date(updatedAt))) updatedAt = folder.updatedAt
    })
    node.documents.forEach((document) => {
      documentCount++
      chunkCount += Number(document.chunk_count || 0)
      const status = document.status in statusCounts ? document.status : 'failed'
      statusCounts[status]++
      if (document.updated_at && (!updatedAt || new Date(document.updated_at) > new Date(updatedAt))) updatedAt = document.updated_at
    })

    node.documentCount = documentCount
    node.chunkCount = chunkCount
    node.statusCounts = statusCounts
    node.updatedAt = updatedAt
  }

  summarize(root)
  const rows = []

  function appendRows(node, depth) {
    node.folders.forEach((folder) => {
      rows.push({ ...folder, type: 'folder', depth, key: folder.id })
      if (expandedFolders.value.has(folder.id)) appendRows(folder, depth + 1)
    })
    node.documents.forEach((document) => {
      rows.push({ type: 'document', depth, key: `document:${document.id}`, document })
    })
  }

  appendRows(root, 0)
  return rows
})

function toggleFolder(folderId) {
  const next = new Set(expandedFolders.value)
  if (next.has(folderId)) next.delete(folderId)
  else next.add(folderId)
  expandedFolders.value = next
}

function statusText(status) {
  return { ready: '已就绪', processing: '处理中', failed: '失败' }[status] || status
}

function progressValue(document) {
  const value = Number(document.progress || 0)
  return Math.min(100, Math.max(0, value))
}

function folderStatus(folder) {
  if (folder.statusCounts.processing) return 'processing'
  if (folder.statusCounts.failed) return 'failed'
  return 'ready'
}

function folderStatusText(folder) {
  if (folder.statusCounts.processing) return `处理中 ${folder.statusCounts.processing}/${folder.documentCount}`
  if (folder.statusCounts.failed) return `失败 ${folder.statusCounts.failed}/${folder.documentCount}`
  return '已就绪'
}

function formatDate(value) {
  if (!value) return '-'
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}
</script>

<template>
  <div class="table-wrap">
    <table class="document-table">
      <thead>
        <tr>
          <th>文档</th>
          <th>状态</th>
          <th>片段</th>
          <th>更新时间</th>
          <th><span class="visually-hidden">操作</span></th>
        </tr>
      </thead>
      <tbody>
        <tr v-if="loading">
          <td colspan="5" class="table-state"><RefreshCw :size="17" class="spinning" />正在加载文档</td>
        </tr>
        <tr v-else-if="!documents.length">
          <td colspan="5" class="table-state">{{ emptyMessage || '还没有文档，先导入一个文件夹' }}</td>
        </tr>
        <template v-for="row in documentRows" :key="row.key">
          <tr v-if="row.type === 'folder'" class="document-folder-row">
            <td :style="{ '--document-depth': row.depth }">
              <button
                class="document-folder-button"
                type="button"
                :aria-expanded="expandedFolders.has(row.id)"
                :title="row.path"
                @click="toggleFolder(row.id)"
              >
                <ChevronRight :size="15" class="document-folder-chevron" :class="{ expanded: expandedFolders.has(row.id) }" />
                <Folder :size="17" />
                <span>{{ row.name }}</span>
              </button>
              <small class="document-file">{{ row.documentCount }} 个文件</small>
            </td>
            <td>
              <span class="status" :class="`status-${folderStatus(row)}`"><i />{{ folderStatusText(row) }}</span>
            </td>
            <td>{{ row.chunkCount }}</td>
            <td class="muted">{{ formatDate(row.updatedAt) }}</td>
            <td class="document-actions">
              <button
                v-if="folderActions"
                class="icon-button danger-icon"
                :title="`删除文件夹 ${row.name}`"
                :aria-label="`删除文件夹 ${row.name} 及其中 ${row.documentCount} 个文档`"
                :disabled="Boolean(deletingId)"
                @click="emit('delete-folder', { id: row.id, name: row.name, path: row.path, documentCount: row.documentCount })"
              >
                <LoaderCircle v-if="deletingId === row.id" :size="15" class="spinning" />
                <Trash2 v-else :size="15" />
              </button>
            </td>
          </tr>
          <tr v-else class="document-file-row">
            <td :style="{ '--document-depth': row.depth }">
              <div class="document-name"><FileText :size="17" /><span>{{ row.document.title }}</span></div>
              <small class="document-file">{{ row.document.file_name }}</small>
            </td>
            <td>
              <span class="status" :class="`status-${row.document.status}`">
                <i />{{ statusText(row.document.status) }}<template v-if="row.document.status === 'processing'"> {{ progressValue(row.document) }}%</template>
              </span>
              <div v-if="row.document.status === 'processing'" class="document-progress" :aria-label="`导入进度 ${progressValue(row.document)}%`">
                <span :style="{ width: `${progressValue(row.document)}%` }" />
              </div>
            </td>
            <td>{{ row.document.chunk_count || 0 }}</td>
            <td class="muted">{{ formatDate(row.document.updated_at) }}</td>
            <td class="document-actions">
              <button
                class="icon-button danger-icon"
                :title="`删除文档 ${row.document.title}`"
                :aria-label="`删除文档 ${row.document.title}`"
                :disabled="Boolean(deletingId)"
                @click="emit('delete', row.document)"
              >
                <LoaderCircle v-if="deletingId === row.document.id" :size="15" class="spinning" />
                <Trash2 v-else :size="15" />
              </button>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</template>
