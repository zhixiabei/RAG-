<script setup>
import { ref } from 'vue'
import { FileStack, LoaderCircle, LockKeyhole, LogIn, UserRound } from 'lucide-vue-next'

defineProps({
  busy: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['submit'])
const username = ref('')
const password = ref('')

function submit() {
  if (!username.value.trim() || !password.value) return
  emit('submit', username.value.trim(), password.value)
}
</script>

<template>
  <main class="auth-shell">
    <section class="auth-panel" aria-labelledby="login-title">
      <div class="auth-brand">
        <span class="auth-brand-mark"><FileStack :size="22" /></span>
        <div><strong>RAG Knowledge</strong><span>个人知识库</span></div>
      </div>
      <div class="auth-heading">
        <h1 id="login-title">登录</h1>
        <p>进入你的知识工作区</p>
      </div>
      <form class="auth-form" @submit.prevent="submit">
        <label>
          <span>用户名</span>
          <div class="auth-input"><UserRound :size="16" /><input v-model="username" autocomplete="username" autofocus required /></div>
        </label>
        <label>
          <span>密码</span>
          <div class="auth-input"><LockKeyhole :size="16" /><input v-model="password" type="password" autocomplete="current-password" required /></div>
        </label>
        <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
        <button class="auth-submit" type="submit" :disabled="busy || !username.trim() || !password">
          <LoaderCircle v-if="busy" :size="17" class="spinning" />
          <LogIn v-else :size="17" />
          {{ busy ? '正在登录' : '登录' }}
        </button>
      </form>
    </section>
  </main>
</template>
