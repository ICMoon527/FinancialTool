import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
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
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name]-v9-[hash].js`,
        chunkFileNames: `assets/[name]-v9-[hash].js`,
        assetFileNames: `assets/[name]-v9-[hash].[ext]`
      }
    }
  },
})
