<script setup>
import { computed, onMounted, ref } from 'vue'
import { CheckCircle2, CircleAlert, FlaskConical, LoaderCircle, X } from 'lucide-vue-next'
import { listEvaluationSamples } from '../services/api'

const props = defineProps({
  kbId: { type: String, required: true },
  busy: Boolean,
  error: { type: String, default: '' },
  result: { type: Object, default: null },
})

const emit = defineEmits(['close', 'run'])

const samples = ref([])
const loading = ref(true)
const loadError = ref('')
const scope = ref('all')
const selectedIds = ref(new Set())

const selectedCount = computed(() => selectedIds.value.size)
const canRun = computed(() => !props.busy && (scope.value === 'all' || selectedCount.value > 0))

async function loadSamples() {
  loading.value = true
  loadError.value = ''
  try {
    samples.value = await listEvaluationSamples(props.kbId)
  } catch (cause) {
    loadError.value = cause instanceof Error ? cause.message : '测试集加载失败'
  } finally {
    loading.value = false
  }
}

function toggleSample(questionId) {
  const next = new Set(selectedIds.value)
  if (next.has(questionId)) next.delete(questionId)
  else next.add(questionId)
  selectedIds.value = next
}

function submit() {
  if (!canRun.value) return
  emit('run', scope.value === 'all' ? [] : [...selectedIds.value])
}

function percentage(value) {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

function duration(value) {
  const milliseconds = Number(value)
  if (!Number.isFinite(milliseconds)) return '—'
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(2)} s`
}

function integer(value) {
  const number = Number(value)
  return Number.isFinite(number) ? Math.round(number).toLocaleString('zh-CN') : '—'
}

function hitStatus(value) {
  if (value == null) return '—'
  return value ? '命中' : '未命中'
}

onMounted(loadSamples)
</script>

<template>
  <div class="modal-backdrop" @click.self="!busy && emit('close')">
    <section class="modal-panel evaluation-dialog" role="dialog" aria-modal="true" aria-labelledby="evaluation-dialog-title">
      <div class="modal-heading">
        <div class="evaluation-dialog-heading">
          <span class="evaluation-mark"><FlaskConical :size="18" /></span>
          <div>
            <span class="eyebrow">RAG EVALUATION</span>
            <h2 id="evaluation-dialog-title">测试知识库</h2>
          </div>
        </div>
        <button class="icon-button" title="关闭" :disabled="busy" @click="emit('close')"><X :size="18" /></button>
      </div>

      <p class="evaluation-dialog-message">选择要运行的测试集范围，测试结果不会写入对话历史。</p>

      <div v-if="loading" class="evaluation-loading">
        <LoaderCircle :size="17" class="spinning" />正在读取测试集
      </div>
      <p v-else-if="loadError" class="error-text">{{ loadError }}</p>
      <template v-else>
        <div class="evaluation-scope">
          <label class="evaluation-option">
            <input v-model="scope" type="radio" value="all" />
            <span>
              <strong>全部 Approved</strong>
              <small>运行 {{ samples.length }} 道已审核题目</small>
            </span>
          </label>
          <label class="evaluation-option">
            <input v-model="scope" type="radio" value="selected" />
            <span>
              <strong>指定题目</strong>
              <small>已选择 {{ selectedCount }} 道</small>
            </span>
          </label>
        </div>

        <div v-if="scope === 'selected'" class="evaluation-sample-list">
          <label v-for="sample in samples" :key="sample.question_id" class="evaluation-sample">
            <input
              type="checkbox"
              :checked="selectedIds.has(sample.question_id)"
              @change="toggleSample(sample.question_id)"
            />
            <span>
              <strong>{{ sample.question_id }}</strong>
              <small>{{ sample.question }}</small>
            </span>
          </label>
          <p v-if="!samples.length" class="evaluation-empty">没有可测试的 Approved 题目</p>
        </div>
      </template>

      <div v-if="result" class="evaluation-result">
        <div class="evaluation-result-heading" :class="{ stopped: result.summary.stopped_early }">
          <CircleAlert v-if="result.summary.stopped_early" :size="17" />
          <CheckCircle2 v-else :size="17" />
          <strong>{{ result.summary.stopped_early ? '测试已中止' : '测试完成' }}</strong>
        </div>
        <div class="evaluation-metrics">
          <span>
            已处理
            <strong>{{ result.summary.sample_count }}/{{ result.summary.requested_count ?? result.summary.sample_count }}</strong>
          </span>
          <span v-if="result.summary.error_count">错误 <strong>{{ result.summary.error_count }}</strong></span>
          <span>文档命中率 <strong>{{ percentage(result.summary.document_hit_rate) }}</strong></span>
          <span>Chunk 命中率 <strong>{{ percentage(result.summary.chunk_hit_rate) }}</strong></span>
          <span>Chunk F1 <strong>{{ percentage(result.summary.chunk_f1) }}</strong></span>
          <span>答案 F1 <strong>{{ percentage(result.summary.answer_f1) }}</strong></span>
          <span>平均响应 <strong>{{ duration(result.summary.average_response_time_ms) }}</strong></span>
          <span>P95 响应 <strong>{{ duration(result.summary.p95_response_time_ms) }}</strong></span>
          <span>Token 总量 <strong>{{ result.summary.token_usage_sample_count ? integer(result.summary.total_tokens) : '—' }}</strong></span>
        </div>
        <p v-if="result.stop_reason" class="evaluation-stop-message">{{ result.stop_reason }}</p>
        <div v-if="result.results?.length" class="evaluation-result-list">
          <div v-for="item in result.results" :key="item.question_id" class="evaluation-result-item" :class="{ error: item.error }">
            <div class="evaluation-result-info">
              <span class="evaluation-result-id">{{ item.question_id }}</span>
              <small v-if="item.error" class="evaluation-error-detail" :title="item.error">{{ item.error }}</small>
              <small v-else>
                文档 {{ hitStatus(item.document_hit) }} · Chunk {{ hitStatus(item.chunk_hit) }} · Chunk F1 {{ percentage(item.chunk_f1) }} · 答案 F1 {{ percentage(item.answer_f1) }} · {{ duration(item.response_time_ms) }} · {{ item.token_usage?.available ? `${integer(item.token_usage.total_tokens)} Token` : 'Token 未上报' }}
              </small>
            </div>
            <strong v-if="item.error">错误</strong>
          </div>
        </div>
      </div>

      <p v-if="error" class="error-text">{{ error }}</p>
      <div class="modal-actions">
        <button class="button secondary" :disabled="busy" @click="emit('close')">关闭</button>
        <button class="button primary" :disabled="!canRun || loading || !samples.length" @click="submit">
          <LoaderCircle v-if="busy" :size="15" class="spinning" />
          <FlaskConical v-else :size="15" />
          {{ busy ? '测试中…' : '开始测试' }}
        </button>
      </div>
    </section>
  </div>
</template>
