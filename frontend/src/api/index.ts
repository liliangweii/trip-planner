import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  timeout: 30000,
})

/** POST /api/trip/plan —— 生成旅行计划（Agent 多轮调用可达 1-3 分钟，单独放宽超时） */
export function planTrip(req: TripRequest) {
  return api.post<TripPlan>('/trip/plan', req, { timeout: 300000 })
}

/** POST /api/trip/export/pdf —— 导出行程计划为 PDF（回传 TripPlan，返回 PDF 二进制流） */
export function exportPlanPdf(plan: TripPlan) {
  return api.post('/trip/export/pdf', plan, {
    responseType: 'blob',
    timeout: 60000,
  })
}

/** 触发浏览器下载一段二进制（PDF 等） */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/** POST /api/chat —— RAG 问答（SSE 流式），返回 fetch Response */
export async function chatStream(sessionId: string, question: string) {
  return fetch(`${import.meta.env.VITE_API_BASE || '/api'}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question }),
  })
}

/** GET /api/search —— 裸检索调试 */
export function search(params: { query: string; city?: string; category?: string; top_k?: number }) {
  return api.get('/search', { params })
}

// ===== 类型定义（与后端 schemas 对齐）=====
export interface TripRequest {
  city: string
  start_date: string
  end_date: string
  travel_days: number
  transportation?: string
  accommodation?: string
  preferences?: string[]
  free_text_input?: string
  budget_total?: number
}

export interface Citation {
  chunk_id: string
  source: string
  url?: string
  snippet: string
}

export interface Location {
  longitude: number
  latitude: number
}

export interface Attraction {
  name: string
  address: string
  location: Location
  visit_duration: number
  description: string
  category: string
  ticket_price: number
  citations: Citation[]
  image_url: string
}

export interface Meal {
  type: string
  name: string
  description?: string
  location?: Location | null
}

export interface Hotel {
  name: string
  address: string
  location: Location
  price_per_night: number
  description?: string
  citations?: Citation[]
  image_url?: string
}

export interface WeatherInfo {
  date: string
  day_weather?: string
  night_weather?: string
  temperature?: string
}

export interface Budget {
  currency?: string
  total: number
  breakdown?: Record<string, number>
}

export interface DayPlan {
  date: string
  day_index: number
  description: string
  transportation: string
  accommodation: string
  hotel?: Hotel | null
  attractions: Attraction[]
  meals?: Meal[]
}

export interface TripPlan {
  city: string
  start_date: string
  end_date: string
  days: DayPlan[]
  weather_info: WeatherInfo[]
  overall_suggestions: string
  budget: Budget
  references: Citation[]
  degraded?: boolean
  degraded_reason?: string
}

export default api
