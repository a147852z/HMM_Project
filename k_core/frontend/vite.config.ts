import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 前端 5174、後端 8900，避免與舊平台（5173／8800）衝突。
// 需要連到其他埠的後端時，設定環境變數 KCORE_API_URL，例如 http://localhost:8901
const target = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.KCORE_API_URL
  ?? 'http://localhost:8900'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: { '/api': target, '/health': target },
  },
})
