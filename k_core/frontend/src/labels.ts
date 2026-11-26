import type { ConversionStatus } from './types'

export const STATUS_LABEL: Record<ConversionStatus, string> = {
  queued: '已排隊',
  running: '轉換中',
  failed: '失敗',
  needs_review: '待審核',
  approved: '已核准',
  rejected: '已退回',
}

export const STATUS_COLOR: Record<ConversionStatus, 'default' | 'info' | 'error' | 'warning' | 'success'> = {
  queued: 'default',
  running: 'info',
  failed: 'error',
  needs_review: 'warning',
  approved: 'success',
  rejected: 'default',
}

const OPERATION_LABEL: Record<string, string> = {
  insert_activity: '插入活動',
  remove_activity: '刪除活動',
  update_parameters: '更新參數',
  update_activity: '修改活動',
  add_dependency: '新增依賴',
  remove_dependency: '移除依賴',
  rewire_dependency: '調整依賴',
  composite: '複合修改',
  // CVflow
  strategy_switch: '換策略',
  param_tune: '調參',
  add_step: '加入步驟',
  remove_step: '刪除步驟',
  replace_step: '替換步驟',
  rewire: '調整連線',
}

const CATEGORY_LABEL: Record<string, string> = {
  io: '輸入', preprocess: '前處理', segment: '分割', morphology: '形態學', region: '區域', detect: '偵測',
  postprocess: '後處理', output: '輸出', other: '其他', none: '無',
}

export const categoryLabel = (category: string) => CATEGORY_LABEL[category] ?? category

/** 觀測符號（例如 param_tune|detect）轉成中文；未知的部分保留原文。 */
export function tokenLabel(token: string): string {
  if (token === '__UNK__') return '未見過的組合'
  const [op, ...rest] = token.split('|')
  return [OPERATION_LABEL[op] ?? op, ...rest.map((part) => CATEGORY_LABEL[part] ?? part)].join('・')
}

export const METHOD_LABEL: Record<string, string> = {
  hmm: 'HMM',
  hmm_bigram_mix: 'HMM＋上一次（平均）',
  hmm_route_bigram: 'HMM＋路線轉移表',
  bigram: '只看上一次修改',
  unigram: '只看頻率',
}
export const methodLabel = (m: string) => METHOD_LABEL[m] ?? m

export const STATE_COLORS = ['#6a45a8', '#1f8a70', '#d9822b', '#2f6fd6', '#c2410c', '#64748b']
export const stateColor = (state: string) => STATE_COLORS[Number(state.replace(/\D/g, '')) % STATE_COLORS.length]

export const operationLabel = (op: string) => OPERATION_LABEL[op] ?? op

export const DOMAIN_LABEL: Record<string, string> = {
  erp: 'ERP 場域', cvflow: 'CVflow', hardware: '硬體生成', other: '其他',
}

export function formatTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-TW', { hour12: false })
}

export const isActive = (status: ConversionStatus) => status === 'queued' || status === 'running'
