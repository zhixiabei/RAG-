<script setup>
import { ref } from 'vue'
import { X } from 'lucide-vue-next'

defineProps({ saving: Boolean })
const emit = defineEmits(['close', 'submit'])

const name = ref('')
const description = ref('')

function submit() {
  if (!name.value.trim()) return
  emit('submit', name.value.trim(), description.value.trim())
}
</script>

<template>
  <div class="modal-backdrop" @click.self="emit('close')">
    <section class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="create-kb-title">
      <div class="modal-heading">
        <div>
          <span class="eyebrow">新建空间</span>
          <h2 id="create-kb-title">创建知识库</h2>
        </div>
        <button class="icon-button" title="关闭" @click="emit('close')"><X :size="18" /></button>
      </div>
      <label class="field-label" for="kb-name">名称</label>
      <input id="kb-name" v-model="name" class="text-input" placeholder="例如：财务制度库" autofocus @keyup.enter="submit" />
      <label class="field-label" for="kb-description">描述 <span>可选</span></label>
      <textarea id="kb-description" v-model="description" class="text-input textarea" placeholder="说明这个知识库覆盖的资料范围" />
      <div class="modal-actions">
        <button class="button secondary" @click="emit('close')">取消</button>
        <button class="button primary" :disabled="saving || !name.trim()" @click="submit">{{ saving ? '创建中…' : '创建知识库' }}</button>
      </div>
    </section>
  </div>
</template>
