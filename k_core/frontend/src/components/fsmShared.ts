// 有限狀態機頁面（單一演變過程、同一題合併）共用的文字與標籤工具。
import { categoryLabel, operationLabel } from '../labels'
import type { Granularity } from '../types'

export const GRANULARITY_LABEL: Record<Granularity, string> = {
  full: '完整流程圖', structure: '結構（不含參數）', category: '節點類別',
}
export const GRANULARITY_HELP: Record<Granularity, string> = {
  full: '節點類型、參數、連線都相同才算同一個狀態。完整流程圖很少重複，所以通常是一條直線。',
  structure: '節點類型與連線相同就算同一個狀態，參數不同也一樣；調參數會成為停在原地的自迴圈。',
  category: '只看流程圖用到哪幾類節點（前處理、分割、偵測⋯），最粗，最容易看出在幾種做法之間來回。',
}

export const pct = (v: number | undefined | null) => (v === undefined || v === null ? '—' : `${Math.round(v * 100)}%`)
export const shortType = (t: string) => t.replace(/^[a-z]+\./, '')

/** 狀態摘要轉成畫面上的名稱：節點類別翻成中文，節點類型去掉 cv. / flow. 前綴。 */
export const stateItems = (summary: string[], granularity: Granularity) =>
  granularity === 'category' ? summary.map(categoryLabel) : summary.map(shortType)

/** 次數最多的一項（例如隱藏狀態）；沒有時為 null。 */
export function majority(counts: Record<string, number>): string | null {
  const entries = Object.entries(counts)
  if (entries.length === 0) return null
  return entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0]
}

// 中日韓文字與全形符號（U+2E80 到 U+FFEF）約與字級同寬
const WIDE = new RegExp(`[${String.fromCharCode(0x2e80)}-${String.fromCharCode(0xffef)}]`)

/** 估計文字寬度（px）：中文與全形字約等於字級，其他字約 0.58 倍。 */
export function textWidth(text: string, fontSize: number): number {
  let width = 0
  for (const ch of text) width += WIDE.test(ch) ? fontSize : fontSize * 0.58
  return width
}

export function truncateTo(text: string, maxWidth: number, fontSize: number): string {
  if (textWidth(text, fontSize) <= maxWidth) return text
  let out = ''
  for (const ch of text) {
    if (textWidth(`${out}${ch}…`, fontSize) > maxWidth) break
    out += ch
  }
  return `${out}…`
}

/** 依寬度把項目拆成兩行（超過的部分截斷），用於狀態框內。 */
export function wrapTwoLines(items: string[], maxWidth: number, fontSize: number): [string, string] {
  const lines: string[] = ['']
  for (const item of items) {
    const current = lines[lines.length - 1]
    const next = current ? `${current}・${item}` : item
    if (textWidth(next, fontSize) > maxWidth && current) lines.push(item)
    else lines[lines.length - 1] = next
  }
  return [truncateTo(lines[0], maxWidth, fontSize), truncateTo(lines.slice(1).join('・'), maxWidth, fontSize)]
}

/** 轉移標籤：第一行是修改類型，第二行是機率與次數。 */
export function transitionText(
  t: { operations: Record<string, number>; probability: number; count: number }, showProb: boolean,
): [string, string] {
  const ops = Object.keys(t.operations)
  const first = ops.length === 1 ? operationLabel(ops[0]) : `${ops.length} 種修改`
  const parts = [showProb ? pct(t.probability) : '', t.count > 1 ? `×${t.count}` : ''].filter(Boolean)
  return [first, parts.join(' ')]
}

/** CVflow 的 scope：tasks 是第一次考試、retake 是補考；其他案例沒有這個欄位。 */
export function laneLabel(teacher: string | null | undefined, scope: string | null | undefined, fallback: string) {
  const suffix = scope === 'retake' ? '（補考）' : scope && scope !== 'tasks' ? `（${scope}）` : ''
  return `${teacher ?? fallback}${suffix}`
}
