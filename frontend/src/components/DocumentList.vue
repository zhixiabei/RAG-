<script setup>
import { FileText, LoaderCircle, RefreshCw, Trash2 } from 'lucide-vue-next'

defineProps({
  documents: { type: Array, default: () => [] },
  loading: Boolean,
  emptyMessage: { type: String, default: '' },
  deletingId: { type: String, default: null },
})

const emit = defineEmits(['delete'])

function statusText(status) {
  return { ready: '已就绪', processing: '处理中', failed: '失败' }[status] || status
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
        <tr v-for="document in documents" :key="document.id">
          <td>
            <div class="document-name"><FileText :size="17" /><span>{{ document.title }}</span></div>
            <small class="document-file">{{ document.file_name }}</small>
          </td>
          <td><span class="status" :class="`status-${document.status}`"><i />{{ statusText(document.status) }}</span></td>
          <td>{{ document.chunk_count || 0 }}</td>
          <td class="muted">{{ formatDate(document.updated_at) }}</td>
          <td class="document-actions">
            <button
              class="icon-button danger-icon"
              :title="`删除文档 ${document.title}`"
              :aria-label="`删除文档 ${document.title}`"
              :disabled="Boolean(deletingId)"
              @click="emit('delete', document)"
            >
              <LoaderCircle v-if="deletingId === document.id" :size="15" class="spinning" />
              <Trash2 v-else :size="15" />
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
