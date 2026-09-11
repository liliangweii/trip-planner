import { createApp } from 'vue'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import App from './App.vue'
import router from './router'
import './styles/main.css'

// 吞掉高德 JSAPI 2.0 的一个已知良性报错：其内部变更上报器（reportAllChanges）
// 由定时器触发，偶尔会对未初始化的对象读 startTime 抛错（"Cannot read properties
// of undefined (reading 'startTime')"）。该错误不影响地图/图片功能，纯属控制台噪音。
// 仅对此特定错误返回 true 抑制，其余真实错误照常抛出。
window.onerror = (message, _source, _lineno, _colno, _error) => {
  const msg = typeof message === 'string' ? message : String(message)
  if (msg.includes("reading 'startTime'") || msg.includes('reportAllChanges')) {
    return true
  }
  return false
}

const app = createApp(App)
app.use(Antd)
app.use(router)
app.mount('#app')
