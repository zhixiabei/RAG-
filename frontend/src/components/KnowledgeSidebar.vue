<script setup>
import { Database, Plus, RefreshCw } from 'lucide-vue-next'

defineProps({ items: { type: Array, default: () => [] }, selectedId: { type: String, default: null }, loading: Boolean })

const emit = defineEmits(['select', 'create', 'refresh'])
</script>

<template>
  <aside class="sidebar">
    <div class="brand-block">
      <div class="brand-mark"><Database :size="18" /></div>
      <div>
        <strong>知识库助手</strong>
        <span>RAG Workspace</span>
      </div>
    </div>

    <div class="sidebar-heading">
      <span>知识库</span>
      <button class="icon-button subtle" title="刷新知识库" @click="emit('refresh')">
        <RefreshCw :size="15" :class="{ spinning: loading }" />
      </button>
    </div>

    <nav class="knowledge-list" aria-label="知识库列表">
      <button
        v-for="item in items"
        :key="item.id"
        class="knowledge-item"
        :class="{ active: item.id === selectedId }"
        @click="emit('select', item.id)"
      >
        <span class="knowledge-dot" />
        <span class="knowledge-copy">
          <strong>{{ item.name }}</strong>
          <small>{{ item.description || '暂无描述' }}</small>
        </span>
      </button>
      <div v-if="!items.length" class="sidebar-empty">还没有知识库</div>
    </nav>

    <button class="sidebar-create" @click="emit('create')">
      <Plus :size="16" />
      新建知识库
    </button>

    <div class="sidebar-footer">本机开发环境 · Ollama</div>
  </aside>
</template>
