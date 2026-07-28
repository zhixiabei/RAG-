<script setup>
import { Database, LoaderCircle, PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, Trash2 } from 'lucide-vue-next'

defineProps({
  items: { type: Array, default: () => [] },
  selectedId: { type: String, default: null },
  loading: Boolean,
  deletingId: { type: String, default: null },
  collapsed: Boolean,
})

const emit = defineEmits(['select', 'create', 'refresh', 'delete', 'toggle'])
</script>

<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="brand-block">
      <div class="brand-mark"><Database :size="18" /></div>
      <div class="brand-copy">
        <strong>知识库助手</strong>
        <span>RAG Workspace</span>
      </div>
      <button class="sidebar-toggle" :title="collapsed ? '展开侧边栏' : '收起侧边栏'" :aria-label="collapsed ? '展开侧边栏' : '收起侧边栏'" @click="emit('toggle')">
        <PanelLeftOpen v-if="collapsed" :size="17" />
        <PanelLeftClose v-else :size="17" />
      </button>
    </div>

    <div class="sidebar-heading">
      <span>知识库</span>
      <button class="icon-button subtle" title="刷新知识库" aria-label="刷新知识库" @click="emit('refresh')">
        <RefreshCw :size="15" :class="{ spinning: loading }" />
      </button>
    </div>

    <nav class="knowledge-list" aria-label="知识库列表">
      <div
        v-for="item in items"
        :key="item.id"
        class="knowledge-item"
        :class="{ active: item.id === selectedId }"
      >
        <button class="knowledge-select" :title="collapsed ? item.name : undefined" :aria-label="item.name" :disabled="Boolean(deletingId)" @click="emit('select', item.id)">
          <span class="knowledge-dot" />
          <span class="knowledge-copy">
            <strong>{{ item.name }}</strong>
            <small>{{ item.description || '暂无描述' }}</small>
          </span>
        </button>
        <button
          class="knowledge-delete"
          :title="`删除知识库 ${item.name}`"
          :aria-label="`删除知识库 ${item.name}`"
          :disabled="Boolean(deletingId)"
          @click="emit('delete', item)"
        >
          <LoaderCircle v-if="deletingId === item.id" :size="14" class="spinning" />
          <Trash2 v-else :size="14" />
        </button>
      </div>
      <div v-if="!items.length" class="sidebar-empty">还没有知识库</div>
    </nav>

    <button class="sidebar-create" :title="collapsed ? '新建知识库' : undefined" @click="emit('create')">
      <Plus :size="16" />
      新建知识库
    </button>
  </aside>
</template>
