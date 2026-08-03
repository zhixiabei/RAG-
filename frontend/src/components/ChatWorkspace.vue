<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { Bot, BrainCircuit, Check, FileText, LoaderCircle, MessageSquareText, Paperclip, Pencil, Plus, RefreshCw, Send, Trash2, UserRound, X } from 'lucide-vue-next'
import DeleteConfirmDialog from './DeleteConfirmDialog.vue'
import { renderMarkdown } from '../utils/markdown'
import { canRecoverAnswerFailure, recoverCompletedAnswer } from '../utils/chatRecovery'
import { SUPPORTED_DOCUMENT_ACCEPT, isSupportedDocument } from '../utils/supportedDocuments'
import {
  askKnowledgeBase,
  askKnowledgeBaseWithParsedAttachments,
  createConversation,
  deleteConversation,
  listChatModels,
  listConversationMessages,
  listConversations,
  parseChatAttachment,
  renameConversation,
  uploadDocument,
} from '../services/api'

const props = defineProps({ kbId: { type: String, required: true } })
const emit = defineEmits(['documents-updated'])
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
const renamingId = ref(null)
const renameTitle = ref('')
const renameInput = ref(null)
const savingRename = ref(false)
const conversationDeleteTarget = ref(null)
const deletingConversation = ref(false)
const conversationDeleteError = ref('')
const attachmentInput = ref(null)
const attachments = ref([])
const saveAttachments = ref(false)
const attachmentNotice = ref('')
let conversationRequestId = 0
let messageRequestId = 0
const MAX_ATTACHMENT_CONCURRENCY = 1

const parsingAttachmentCount = computed(() => attachments.value.filter((item) => item.status === 'parsing').length)
const attachmentsReady = computed(() => attachments.value.every((item) => item.status === 'ready'))
const canSend = computed(() => (
  Boolean(question.value.trim())
  && !sending.value
  && !loadingMessages.value
  && attachmentsReady.value
))

async function parseAttachment(entry, knowledgeBaseId) {
  try {
    const parsed = await parseChatAttachment(knowledgeBaseId, entry.file)
    if (knowledgeBaseId !== props.kbId || !attachments.value.includes(entry)) return
    entry.parsed = parsed
    entry.status = 'ready'
  } catch (cause) {
    if (knowledgeBaseId !== props.kbId || !attachments.value.includes(entry)) return
    entry.status = 'failed'
    entry.error = cause instanceof Error ? cause.message : '附件解析失败'
  }
}

async function parseAttachmentQueue(entries, knowledgeBaseId) {
  let nextIndex = 0
  async function runWorker() {
    while (nextIndex < entries.length) {
      const entry = entries[nextIndex]
      nextIndex++
      await parseAttachment(entry, knowledgeBaseId)
    }
  }
  const workerCount = Math.min(MAX_ATTACHMENT_CONCURRENCY, entries.length)
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()))
}

function chooseAttachments(event) {
  const selected = Array.from(event.target.files || [])
  const existing = new Set(attachments.value.map((item) => item.id))
  const next = []
  let skipped = 0
  for (const file of selected) {
    const identity = `${file.name}:${file.size}:${file.lastModified}`
    if (!isSupportedDocument(file.name) || existing.has(identity) || attachments.value.length + next.length >= 10) {
      skipped++
      continue
    }
    existing.add(identity)
    next.push({ id: identity, file, status: 'parsing', parsed: null, error: '' })
  }
  attachments.value = [...attachments.value, ...next]
  attachmentNotice.value = skipped ? `已跳过 ${skipped} 个不支持、重复或超出数量限制的文件` : ''
  if (attachmentInput.value) attachmentInput.value.value = ''
  const knowledgeBaseId = props.kbId
  parseAttachmentQueue(next, knowledgeBaseId)
}

function removeAttachment(index) {
  attachments.value = attachments.value.filter((_, itemIndex) => itemIndex !== index)
  attachmentNotice.value = ''
}

function retryAttachment(entry) {
  entry.status = 'parsing'
  entry.error = ''
  entry.parsed = null
  parseAttachment(entry, props.kbId)
}

async function persistAttachments(knowledgeBaseId, entries) {
  const results = []
  for (const entry of entries) {
    try {
      results.push({ status: 'fulfilled', value: await uploadDocument(knowledgeBaseId, entry.file) })
    } catch (reason) {
      results.push({ status: 'rejected', reason })
    }
  }
  emit('documents-updated')
  const failures = results.filter((result) => result.status === 'rejected' && result.reason?.status !== 409)
  return failures.length
}

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
  if (sending.value || deletingConversation.value) return
  messageRequestId += 1
  activeConversationId.value = null
  messages.value = []
  question.value = ''
  attachments.value = []
  attachmentNotice.value = ''
  error.value = ''
  loadingMessages.value = false
}

async function startRename(conversation) {
  if (sending.value || deletingConversation.value) return
  renamingId.value = conversation.id
  renameTitle.value = conversation.title
  error.value = ''
  await nextTick()
  renameInput.value?.focus()
  renameInput.value?.select()
}

function setRenameInput(element) {
  renameInput.value = element
}

function cancelRename() {
  if (savingRename.value) return
  renamingId.value = null
  renameTitle.value = ''
}

async function saveRename(conversation) {
  const title = renameTitle.value.trim()
  if (!title || savingRename.value) return
  savingRename.value = true
  error.value = ''
  try {
    const updated = await renameConversation(conversation.id, title)
    Object.assign(conversation, updated)
    renamingId.value = null
    renameTitle.value = ''
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '对话改名失败'
  } finally {
    savingRename.value = false
  }
}

function requestConversationDelete(conversation) {
  if (sending.value || deletingConversation.value) return
  cancelRename()
  conversationDeleteError.value = ''
  conversationDeleteTarget.value = conversation
}

function closeConversationDelete() {
  if (deletingConversation.value) return
  conversationDeleteTarget.value = null
  conversationDeleteError.value = ''
}

async function confirmConversationDelete() {
  const target = conversationDeleteTarget.value
  if (!target || deletingConversation.value) return
  deletingConversation.value = true
  conversationDeleteError.value = ''
  let nextConversationId = null
  try {
    await deleteConversation(target.id)
    conversations.value = conversations.value.filter((conversation) => conversation.id !== target.id)
    if (activeConversationId.value === target.id) {
      messageRequestId += 1
      activeConversationId.value = null
      messages.value = []
      loadingMessages.value = false
      nextConversationId = conversations.value[0]?.id || null
    }
    conversationDeleteTarget.value = null
  } catch (cause) {
    conversationDeleteError.value = cause instanceof Error ? cause.message : '对话删除失败'
  } finally {
    deletingConversation.value = false
  }
  if (nextConversationId) await selectConversation(nextConversationId)
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
  attachments.value = []
  attachmentNotice.value = ''
  error.value = ''
  sending.value = false
  renamingId.value = null
  conversationDeleteTarget.value = null
  deletingConversation.value = false
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
  if (!canSend.value) return
  const knowledgeBaseId = props.kbId
  let sentAttachments = []
  let conversationId = activeConversationId.value
  const previousMessageCount = messages.value.length
  sending.value = true
  error.value = ''

  try {
    if (!conversationId) {
      const conversation = await createConversation(knowledgeBaseId, value.slice(0, 50))
      if (knowledgeBaseId !== props.kbId) return
      conversationId = conversation.id
      activeConversationId.value = conversationId
      conversations.value = [conversation, ...conversations.value]
    }

    const entries = [...attachments.value]
    sentAttachments = entries
    const userContent = entries.length ? `${value}\n\n附件：${entries.map((entry) => entry.file.name).join('、')}` : value
    messages.value.push({ id: `pending:user:${Date.now()}`, role: 'user', content: userContent })
    question.value = ''
    attachments.value = []
    attachmentNotice.value = ''
    await scrollToBottom('smooth')

    const parsedAttachments = entries.map((entry) => ({
      name: entry.parsed.name,
      context: entry.parsed.context,
      citations: entry.parsed.citations,
    }))
    const answerRequest = entries.length
      ? askKnowledgeBaseWithParsedAttachments(knowledgeBaseId, conversationId, value, selectedModel.value, parsedAttachments)
      : askKnowledgeBase(knowledgeBaseId, conversationId, value, selectedModel.value)
    const saveRequest = entries.length && saveAttachments.value
      ? persistAttachments(knowledgeBaseId, entries)
      : Promise.resolve(0)
    const [result, saveFailureCount] = await Promise.all([answerRequest, saveRequest])
    if (knowledgeBaseId !== props.kbId || activeConversationId.value !== conversationId) return
    messages.value.push({
      id: `pending:assistant:${Date.now()}`,
      role: 'assistant',
      content: result.answer,
      citations: result.citations || [],
    })
    if (saveFailureCount) attachmentNotice.value = `${saveFailureCount} 个附件未能保存到知识库，但已用于本次回答`
    try {
      await refreshConversations(knowledgeBaseId)
    } catch (refreshFailure) {
      // The answer is already visible and persisted; sidebar refresh errors are non-fatal.
      if (refreshFailure?.status === 401) throw refreshFailure
    }
  } catch (cause) {
    if (knowledgeBaseId === props.kbId) {
      if (conversationId && canRecoverAnswerFailure(cause)) {
        const recoveredMessages = await recoverCompletedAnswer({
          loadMessages: () => listConversationMessages(conversationId),
          shouldContinue: () => (
            knowledgeBaseId === props.kbId
            && activeConversationId.value === conversationId
          ),
          previousMessageCount,
        })
        if (recoveredMessages) {
          messages.value = recoveredMessages
          error.value = ''
          try {
            await refreshConversations(knowledgeBaseId)
          } catch (refreshFailure) {
            if (refreshFailure?.status === 401) error.value = refreshFailure.message
          }
          return
        }
      }
      if (sentAttachments.length && !attachments.value.length) attachments.value = sentAttachments
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
        <div
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ active: activeConversationId === conversation.id }"
        >
          <template v-if="renamingId === conversation.id">
            <input
              :ref="setRenameInput"
              v-model="renameTitle"
              class="conversation-rename-input"
              maxlength="200"
              aria-label="对话名称"
              :disabled="savingRename"
              @keydown.enter.prevent="saveRename(conversation)"
              @keydown.esc.prevent="cancelRename"
            />
            <div class="conversation-actions visible">
              <button class="conversation-action" title="保存名称" :disabled="savingRename || !renameTitle.trim()" @click="saveRename(conversation)">
                <LoaderCircle v-if="savingRename" :size="14" class="spinning" /><Check v-else :size="14" />
              </button>
              <button class="conversation-action" title="取消改名" :disabled="savingRename" @click="cancelRename"><X :size="14" /></button>
            </div>
          </template>
          <template v-else>
            <button class="conversation-select" :disabled="sending || deletingConversation" @click="selectConversation(conversation.id)">
              <MessageSquareText :size="15" />
              <span class="conversation-copy">
                <strong>{{ conversation.title }}</strong>
                <small>{{ conversationMeta(conversation) }}</small>
              </span>
            </button>
            <div class="conversation-actions">
              <button class="conversation-action" title="重命名对话" :disabled="sending || deletingConversation" @click="startRename(conversation)"><Pencil :size="13" /></button>
              <button class="conversation-action danger-icon" title="删除对话" :disabled="sending || deletingConversation" @click="requestConversationDelete(conversation)"><Trash2 :size="13" /></button>
            </div>
          </template>
        </div>
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
            <div v-if="message.role === 'assistant'" class="message-text markdown-body" v-html="renderMarkdown(message.content)" />
            <div v-else class="message-text">{{ message.content }}</div>
            <div v-if="message.citations?.length" class="citations">
              <span class="citation-label">引用</span>
              <span v-for="citation in message.citations" :key="citation.chunk_id" class="citation">
                {{ citation.title }}<span v-if="citation.page_number"> · 第 {{ citation.page_number }} 页</span><span v-if="citation.relevance_score != null"> · 相关度 {{ Math.round(citation.relevance_score * 100) }}%</span>
              </span>
            </div>
          </div>
        </article>
        <div v-if="sending" class="typing"><span /><span /><span />正在判断和组织答案</div>
      </div>
      <div class="chat-composer">
        <input ref="attachmentInput" class="hidden-input" type="file" multiple :accept="SUPPORTED_DOCUMENT_ACCEPT" @change="chooseAttachments" />
        <div v-if="attachments.length" class="attachment-list">
          <div v-for="(attachment, index) in attachments" :key="attachment.id" class="attachment-chip" :class="`attachment-${attachment.status}`">
            <LoaderCircle v-if="attachment.status === 'parsing'" :size="14" class="spinning" />
            <FileText v-else :size="14" />
            <span class="attachment-name" :title="attachment.file.name">{{ attachment.file.name }}</span>
            <small v-if="attachment.status === 'parsing'" class="attachment-status">解析中</small>
            <small v-else-if="attachment.status === 'ready'" class="attachment-status">已解析</small>
            <small v-else class="attachment-status" :title="attachment.error">解析失败</small>
            <button v-if="attachment.status === 'failed'" type="button" title="重新解析" aria-label="重新解析" :disabled="sending" @click="retryAttachment(attachment)"><RefreshCw :size="13" /></button>
            <button type="button" :title="`移除 ${attachment.file.name}`" :aria-label="`移除 ${attachment.file.name}`" :disabled="sending" @click="removeAttachment(index)"><X :size="13" /></button>
          </div>
        </div>
        <textarea v-model="question" rows="3" placeholder="输入你的问题" :disabled="sending || loadingMessages" @keydown.enter.exact.prevent="send" />
        <div class="composer-footer">
          <div class="composer-tools">
            <label class="model-picker" title="选择回答模型">
              <BrainCircuit :size="14" />
              <select v-model="selectedModel" aria-label="选择回答模型" :disabled="sending || !models.length">
                <option v-if="!models.length" value="">默认模型</option>
                <option v-for="model in models" :key="model.id" :value="model.id">
                  {{ model.provider }} · {{ model.name }}{{ model.is_default ? '（默认）' : '' }}
                </option>
              </select>
            </label>
            <label v-if="attachments.length" class="save-attachment-toggle">
              <input v-model="saveAttachments" type="checkbox" :disabled="sending" />
              <span class="toggle-track"><i /></span>
              保存到当前知识库
            </label>
          </div>
          <div class="composer-actions">
            <button class="icon-button attach-button" type="button" title="添加文档" aria-label="添加文档" :disabled="sending" @click="attachmentInput?.click()"><Paperclip :size="16" /></button>
            <button class="icon-button send-button" title="发送问题" :disabled="!canSend" @click="send"><Send :size="17" /></button>
          </div>
        </div>
        <p v-if="parsingAttachmentCount" class="attachment-notice">正在解析 {{ parsingAttachmentCount }} 个附件，完成后可发送。</p>
        <p v-if="attachments.length && !saveAttachments" class="attachment-scope">附件仅用于本次提问，不会保存。</p>
        <p v-if="attachmentNotice" class="attachment-notice">{{ attachmentNotice }}</p>
        <p v-if="error" class="error-text">{{ error }}</p>
      </div>
    </section>

    <DeleteConfirmDialog
      v-if="conversationDeleteTarget"
      title="删除历史对话"
      :message="`对话“${conversationDeleteTarget.title}”及其中的全部消息都会被永久删除。`"
      :busy="deletingConversation"
      :error="conversationDeleteError"
      @close="closeConversationDelete"
      @confirm="confirmConversationDelete"
    />
  </section>
</template>
