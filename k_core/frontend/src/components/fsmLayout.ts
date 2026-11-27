// 跨題目合併的狀態機（同一個解題 AI）用的排版：橫軸是「平均在第幾版第一次走到這個狀態」。
// 起點多的狀態在最左欄，通常要改幾次才會走到的狀態往右；欄內依相鄰欄鄰居的位置排序，減少交叉。

export interface LayoutNode { id: string; order: number; visits: number; firstReach: number }
export interface LayoutEdge { source: string; target: string; count: number }

export interface Layout {
  /** 第幾欄（由左到右，沒有狀態的版數會略過） */
  column: Map<string, number>
  /** 欄內第幾個（由上到下） */
  row: Map<string, number>
  /** 每一欄有幾個狀態 */
  sizes: number[]
  /** 每一欄代表大約第幾版（平均第一次走到的版次，四捨五入） */
  steps: number[]
}

export function stepLayout(nodes: LayoutNode[], edges: LayoutEdge[]): Layout {
  const stepOf = new Map(nodes.map((n) => [n.id, Math.round(n.firstReach)]))
  const steps = [...new Set(stepOf.values())].sort((a, b) => a - b)
  const column = new Map(nodes.map((n) => [n.id, steps.indexOf(stepOf.get(n.id)!)]))
  const columns: string[][] = steps.map(() => [])
  for (const n of [...nodes].sort((a, b) => b.visits - a.visits || a.order - b.order)) columns[column.get(n.id)!].push(n.id)

  // 位置用「欄內位置 − 欄的中心」，讓長短不同的欄可以互相比較（畫面上每一欄垂直置中）
  const links = edges.filter((e) => e.source !== e.target && column.has(e.source) && column.has(e.target)
    && column.get(e.source) !== column.get(e.target))
  const position = new Map<string, number>()
  const place = (ids: string[]) => ids.forEach((id, i) => position.set(id, i - (ids.length - 1) / 2))
  columns.forEach(place)
  const score = (id: string, left: boolean) => {
    const own = column.get(id)!
    let total = 0
    let weighted = 0
    for (const e of links) {
      const other = e.source === id ? e.target : e.target === id ? e.source : null
      if (other === null) continue
      const near = column.get(other)!
      if (left ? near >= own : near <= own) continue
      total += e.count
      weighted += e.count * position.get(other)!
    }
    return total ? weighted / total : position.get(id)!
  }
  const sweep = (index: number, left: boolean) => {
    const scores = new Map(columns[index].map((id) => [id, score(id, left)]))
    columns[index].sort((a, b) => scores.get(a)! - scores.get(b)!)
    place(columns[index])
  }
  for (let round = 0; round < 4; round += 1) {
    for (let i = 1; i < columns.length; i += 1) sweep(i, true)
    for (let i = columns.length - 2; i >= 0; i -= 1) sweep(i, false)
  }
  const row = new Map<string, number>()
  columns.forEach((ids) => ids.forEach((id, i) => row.set(id, i)))
  return { column, row, sizes: columns.map((c) => c.length), steps }
}
