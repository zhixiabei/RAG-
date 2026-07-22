<script setup>
import { nextTick, ref, watch } from 'vue'
import { Bot, BrainCircuit, MessageSquareText, Plus, Send, UserRound } from 'lucide-vue-next'
import {
  askKnowledgeBase,
  createConversation,
  listChatModels,
  listConversationMessages,
  listConversations,
} from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const conversations = ref([])
const activeConversationId = ref(null)
const messages = ref([])
const models = ref([])
const selectedModel = ref('')
const question = ref('')
const sending = ref(false)
const loadingConversations = ref(false)
const loadingMessages = ref(false)
const error = ref('')
const transcript = ref(null)
let conversationRequestId = 0
let messageRequestId = 0

async function scrollToBottom(behavior = 'auto') {
  await nextTick()
  transcript.value?.scrollTo({ top: transcript.value.scrollHeight, behavior })
}

function conversationMeta(conversation) {
  if (!conversation.message_count) return '新对话'
  const date = new Date(conversation.updated_at)
  return `${conversation.message_count} 条消息 · ${date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })}`
}

async function refreshConversations(knowledgeBaseId = props.kbId) {
  const requestId = ++conversationRequestId
  const items = await listConversations(knowledgeBaseId)
  if (requestId === conversationRequestId && knowledgeBaseId === props.kbId) {
    conversations.value = items
  }
  return items
}

async function selectConversation(conversationId) {
  if (sending.value || activeConversationId.value === conversationId) return
  const requestId = ++messageRequestId
  activeConversationId.value = conversationId
  messages.value = []
  error.value = ''
  loadingMessages.value = true
  try {
    const history = await listConversationMessages(conversationId)
    if (requestId !== messageRequestId || activeConversationId.value !== conversationId) return
    messages.value = history
    await scrollToBottom()
  } catch (cause) {
    if (requestId === messageRequestId) {
      error.value = cause instanceof Error ? cause.message : '聊天记录加载失败'
    }
  } finally {
    if (requestId === messageRequestId) loadingMessages.value = false
  }
}

function startNewConversation() {
  if (sending.value) return
  messageRequestId += 1
  activeConversationId.value = null
  messages.value = []
  question.value = ''
  error.value = ''
  loadingMessages.value = false
}

async function loadWorkspace(knowledgeBaseId) {
  conversationRequestId += 1
  messageRequestId += 1
  conversations.value = []
  models.value = []
  selectedModel.value = ''
  activeConversationId.value = null
  messages.value = []
  question.value = ''
  error.value = ''
  sending.value = false
  loadingConversations.value = true
  loadingMessages.value = false
  try {
    const [conversationResult, modelResult] = await Promise.allSettled([
      refreshConversations(knowledgeBaseId),
      listChatModels(),
    ])
    if (knowledgeBaseId !== props.kbId) return
    if (conversationResult.status === 'rejected') throw conversationResult.reason
    const items = conversationResult.value
    const availableModels = modelResult.status === 'fulfilled' ? modelResult.value : []
    models.value = availableModels
    selectedModel.value = availableModels.find((model) => model.is_default)?.id || availableModels[0]?.id || ''
    if (items.length) await selectConversation(items[0].id)
  } catch (cause) {
    if (knowledgeBaseId === props.kbId) {
      error.value = cause instanceof Error ? cause.message : '历史对话加载失败'
    }
  } finally {
    if (knowledgeBaseId === props.kbId) loadingConversations.value = false
  }
}

watch(() => props.kbId, loadWorkspace, { immediate: true })

async function send() {
  const value = question.value.trim()
  if (!value || sending.value || loadingMessages.value) return
  const knowledgeBaseId = props.kbId
  sending.value = true
  error.value = ''

  try {
    let conversationId = activeConversationId.value
    if (!conversationId) {
      const conversation = await createConversation(knowledgeBaseId, value.slice(0, 50))
      if (knowledgeBaseId !== props.kbId) return
      conversationId = conversation.id
      activeConversationId.value = conversationId
      conversations.value = [conversation, ...conversations.value]
    }

    messages.value.push({ id: `pending:user:${Date.now()}`, role: 'user', content: value })
    question.value = ''
    await scrollToBottom('smooth')

    const result = await askKnowledgeBase(knowledgeBaseId, conversationId, value, selectedModel.value)
    if (knowledgeBaseId !== props.kbId || activeConversationId.value !== conversationId) return
    messages.value.push({
      id: `pending:assistant:${Date.now()}`,
      role: 'assistant',
      content: result.answer,
      citations: result.citations || [],
    })
    await refreshConversations(knowledgeBaseId)
  } catch (cause) {
    if (knowledgeBaseId === props.kbId) {
      error.value = cause instanceof Error ? cause.message : '问答失败'
    }
  } finally {
    if (knowledgeBaseId === props.kbId) {
      sending.value = false
      await scrollToBottom('smooth')
    }
  }
}
</script>

<template>
  <section class="chat-workspace">
    <aside class="conversation-sidebar">
      <div class="conversation-heading">
        <span>历史对话</span>
        <button class="icon-button" title="新建对话" :disabled="sending" @click="startNewConversation">
          <Plus :size="16" />
        </button>
      </div>
      <div v-if="loadingConversations" class="conversation-state">正在加载...</div>
      <nav v-else class="conversation-list" aria-label="历史对话列表">
        <button
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ active: activeConversationId === conversation.id }"
          :disabled="sending"
          @click="selectConversation(conversation.id)"
        >
          <MessageSquareText :size="15" />
          <span class="conversation-copy">
            <strong>{{ conversation.title }}</strong>
            <small>{{ conversationMeta(conversation) }}</small>
          </span>
        </button>
        <div v-if="!conversations.length" class="conversation-state">还没有历史对话</div>
      </nav>
    </aside>

    <section class="chat-layout">
      <div ref="transcript" class="transcript">
        <div v-if="loadingMessages" class="chat-empty">
          <div class="typing"><span /><span /><span />正在加载聊天记录</div>
        </div>
        <div v-else-if="!messages.length" class="chat-empty">
          <div class="chat-empty-icon"><Bot :size="22" /></div>
          <h3>{{ activeConversationId ? '这个对话还没有消息' : '开始一个新对话' }}</h3>
          <p>回答会基于当前知识库的文档，并保存在历史对话中。</p>
        </div>
        <article v-for="(message, index) in messages" :key="message.id || index" class="message-row" :class="message.role">
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
        <textarea v-model="question" rows="3" placeholder="输入你的问题" :disabled="sending || loadingMessages" @keydown.enter.exact.prevent="send" />
        <div class="composer-footer">
          <label class="model-picker" title="选择回答模型">
            <BrainCircuit :size="14" />
            <select v-model="selectedModel" aria-label="选择回答模型" :disabled="sending || !models.length">
              <option v-if="!models.length" value="">默认模型</option>
              <option v-for="model in models" :key="model.id" :value="model.id">
                {{ model.provider }} · {{ model.name }}{{ model.is_default ? '（默认）' : '' }}
              </option>
            </select>
          </label>
          <button class="icon-button send-button" title="发送问题" :disabled="sending || loadingMessages || !question.trim()" @click="send"><Send :size="17" /></button>
        </div>
        <p v-if="error" class="error-text">{{ error }}</p>
      </div>
    </section>
  </section>
</template>
