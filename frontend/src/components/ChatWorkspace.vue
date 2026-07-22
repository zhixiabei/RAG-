<script setup>
import { nextTick, ref } from 'vue'
import { Bot, Send, UserRound } from 'lucide-vue-next'
import { askKnowledgeBase } from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const question = ref('')
const sending = ref(false)
const error = ref('')
const messages = ref([])
const transcript = ref(null)

async function send() {
  const value = question.value.trim()
  if (!value || sending.value) return
  messages.value.push({ role: 'user', content: value })
  question.value = ''
  sending.value = true
  error.value = ''
  await nextTick()
  transcript.value?.scrollTo({ top: transcript.value.scrollHeight, behavior: 'smooth' })
  try {
    const result = await askKnowledgeBase(props.kbId, value)
    messages.value.push({ role: 'assistant', content: result.answer, citations: result.citations || [] })
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '问答失败'
  } finally {
    sending.value = false
    await nextTick()
    transcript.value?.scrollTo({ top: transcript.value.scrollHeight, behavior: 'smooth' })
  }
}
</script>

<template>
  <section class="chat-layout">
    <div ref="transcript" class="transcript">
      <div v-if="!messages.length" class="chat-empty">
        <div class="chat-empty-icon"><Bot :size="22" /></div>
        <h3>从知识库开始提问</h3>
        <p>回答会基于当前知识库的文档，并附带可追溯引用。</p>
      </div>
      <article v-for="(message, index) in messages" :key="index" class="message-row" :class="message.role">
        <div class="message-avatar"><UserRound v-if="message.role === 'user'" :size="16" /><Bot v-else :size="16" /></div>
        <div class="message-content">
          <div class="message-label">{{ message.role === 'user' ? '你' : '知识库助手' }}</div>
          <div class="message-text">{{ message.content }}</div>
          <div v-if="message.citations?.length" class="citations">
            <span class="citation-label">引用</span>
            <span v-for="citation in message.citations" :key="citation.chunk_id" class="citation">
              {{ citation.title }}<span v-if="citation.page_number"> · 第 {{ citation.page_number }} 页</span>
            </span>
          </div>
        </div>
      </article>
      <div v-if="sending" class="typing"><span /><span /><span />正在检索和组织答案</div>
    </div>
    <div class="chat-composer">
      <textarea v-model="question" rows="3" placeholder="输入你的问题，例如：差旅费报销需要哪些材料？" :disabled="sending" @keydown.enter.exact.prevent="send" />
      <div class="composer-footer">
        <span>Enter 发送 · Shift + Enter 换行</span>
        <button class="icon-button send-button" title="发送问题" :disabled="sending || !question.trim()" @click="send"><Send :size="17" /></button>
      </div>
      <p v-if="error" class="error-text">{{ error }}</p>
    </div>
  </section>
</template>

