import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  define: {
    // 构建时间戳：注入到前端，用于在页面上确认浏览器加载的是哪个构建（排查缓存问题）
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  server: {
    host: '0.0.0.0',  // 允许公网访问
    port: 5173,       // 默认端口
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // 打包输出到项目根目录的 static 文件夹
    outDir: path.resolve(__dirname, '../../static'),
    emptyOutDir: true,
    // 强制生成完全唯一的文件名，彻底避免缓存问题
    // （前缀 v10：用于区分前端版本，配合 index.html no-cache 头彻底摆脱旧缓存）
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name]-v10-[hash].js`,
        chunkFileNames: `assets/[name]-v10-[hash].js`,
        assetFileNames: `assets/[name]-v10-[hash].[ext]`
      }
    }
  },
})
