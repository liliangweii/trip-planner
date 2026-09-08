import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      redirect: '/plan',
    },
    {
      path: '/plan',
      name: 'plan',
      component: () => import('@/views/PlanView.vue'),
      meta: { title: '规划表单' },
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('@/views/ChatView.vue'),
      meta: { title: 'RAG 问答' },
    },
    {
      path: '/result',
      name: 'result',
      component: () => import('@/views/ResultView.vue'),
      meta: { title: '行程结果' },
    },
    {
      path: '/search',
      name: 'search',
      component: () => import('@/views/SearchView.vue'),
      meta: { title: '检索调试' },
    },
  ],
})

export default router
