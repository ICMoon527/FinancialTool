// 生产环境使用相对路径（同源），开发环境使用环境变量或 Vite 代理（空字符串）
export const API_BASE_URL = import.meta.env.VITE_API_URL || '';
