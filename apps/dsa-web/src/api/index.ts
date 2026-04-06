import axios from 'axios';
import { API_BASE_URL } from '../utils/constants';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 为策略回测API创建一个单独的实例，超时时间更长
export const strategyBacktestApiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10分钟超时
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

strategyBacktestApiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const path = window.location.pathname + window.location.search;
      if (!path.startsWith('/login')) {
        const redirect = encodeURIComponent(path);
        window.location.assign(`/login?redirect=${redirect}`);
      }
    }
    return Promise.reject(error);
  },
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const path = window.location.pathname + window.location.search;
      if (!path.startsWith('/login')) {
        const redirect = encodeURIComponent(path);
        window.location.assign(`/login?redirect=${redirect}`);
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
