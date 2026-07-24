<script setup>
import { AlertTriangle, Trash2, X } from 'lucide-vue-next'

defineProps({
  title: { type: String, required: true },
  message: { type: String, required: true },
  busy: Boolean,
  error: { type: String, default: '' },
})

const emit = defineEmits(['close', 'confirm'])
</script>

<template>
  <div class="modal-backdrop" @click.self="!busy && emit('close')">
    <section class="modal-panel delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-dialog-title">
      <div class="modal-heading">
        <div class="delete-dialog-heading">
          <span class="delete-warning"><AlertTriangle :size="18" /></span>
          <div>
            <span class="eyebrow">不可撤销</span>
            <h2 id="delete-dialog-title">{{ title }}</h2>
          </div>
        </div>
        <button class="icon-button" title="关闭" :disabled="busy" @click="emit('close')"><X :size="18" /></button>
      </div>
      <p class="delete-dialog-message">{{ message }}</p>
      <p v-if="error" class="error-text">{{ error }}</p>
      <div class="modal-actions">
        <button class="button secondary" :disabled="busy" @click="emit('close')">取消</button>
        <button class="button danger" :disabled="busy" @click="emit('confirm')">
          <Trash2 :size="15" />{{ busy ? '正在删除…' : '确认删除' }}
        </button>
      </div>
    </section>
  </div>
</template>
