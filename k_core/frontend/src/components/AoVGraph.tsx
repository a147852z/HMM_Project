import { Box, Chip, Stack, Typography } from '@mui/material'
import { useTheme } from '@mui/material/styles'
import type { AoVEdge, AoVNode, Change } from '../types'

const NODE_W = 150
const NODE_H = 46
const GAP_X = 64
const GAP_Y = 22
const PAD = 16

interface Layout {
  positions: Map<string, { x: number; y: number }>
  width: number
  height: number
}

/** 以「最長路徑分層」排版 DAG：每個節點放在其所有前置節點的右邊。無法排序（含循環）時回傳 null。 */
function layout(nodes: AoVNode[], edges: AoVEdge[]): Layout | null {
  const ids = nodes.map((n) => n.node_id)
  const indegree = new Map(ids.map((id) => [id, 0]))
  const next = new Map<string, string[]>(ids.map((id) => [id, []]))
  for (const edge of edges) {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) continue
    next.get(edge.source)!.push(edge.target)
    indegree.set(edge.target, indegree.get(edge.target)! + 1)
  }
  const layer = new Map(ids.map((id) => [id, 0]))
  const queue = ids.filter((id) => indegree.get(id) === 0)
  let visited = 0
  while (queue.length) {
    const current = queue.shift()!
    visited += 1
    for (const target of next.get(current)!) {
      layer.set(target, Math.max(layer.get(target)!, layer.get(current)! + 1))
      indegree.set(target, indegree.get(target)! - 1)
      if (indegree.get(target) === 0) queue.push(target)
    }
  }
  if (visited !== ids.length) return null

  const columns: string[][] = []
  for (const id of ids) {
    const l = layer.get(id)!
    ;(columns[l] ??= []).push(id)
  }
  const tallest = Math.max(1, ...columns.map((c) => c.length))
  const height = PAD * 2 + tallest * NODE_H + (tallest - 1) * GAP_Y
  const positions = new Map<string, { x: number; y: number }>()
  columns.forEach((column, l) => {
    const columnHeight = column.length * NODE_H + (column.length - 1) * GAP_Y
    const top = (height - columnHeight) / 2
    column.forEach((id, i) => positions.set(id, { x: PAD + l * (NODE_W + GAP_X), y: top + i * (NODE_H + GAP_Y) }))
  })
  const width = PAD * 2 + columns.length * NODE_W + (columns.length - 1) * GAP_X
  return { positions, width, height }
}

const edgeKey = (e: AoVEdge) => `${e.source}→${e.target}`

function truncate(text: string, max: number) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

/** 顯示一個版本的 AoV；傳入 change 時，標示這次修改新增（綠）與更新（橘）的部分，刪除的部分列在圖下方。 */
export function AoVGraph({ nodes, edges, change, beforeNodes }: {
  nodes: AoVNode[]
  edges: AoVEdge[]
  change?: Change
  beforeNodes?: AoVNode[]
}) {
  const theme = useTheme()
  const added = new Set(change?.added_nodes.map((n) => n.node_id) ?? [])
  const updated = new Set(change?.updated_node_attributes.map((u) => u.node_id) ?? [])
  const addedEdges = new Set(change?.added_edges.map(edgeKey) ?? [])
  const result = layout(nodes, edges)
  const nameOf = (id: string) => {
    const node = beforeNodes?.find((n) => n.node_id === id) ?? nodes.find((n) => n.node_id === id)
    return node?.name || node?.activity_type || id
  }

  if (!result) {
    return <Typography color="error.main" variant="body2">這個版本含有循環，無法畫成 AoV。</Typography>
  }

  const colors = {
    stroke: theme.palette.text.secondary,
    text: theme.palette.text.primary,
    muted: theme.palette.text.secondary,
    fill: theme.palette.background.paper,
    added: theme.palette.success.main,
    addedFill: theme.palette.mode === 'dark' ? '#16301f' : '#e9f6ee',
    updated: theme.palette.warning.main,
    updatedFill: theme.palette.mode === 'dark' ? '#34280f' : '#fff4e0',
  }

  return (
    <Stack gap={1}>
      <Box sx={{ overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.default' }}>
        <svg viewBox={`0 0 ${result.width} ${result.height}`}
          role="img" aria-label={`AoV：${nodes.length} 個活動、${edges.length} 條依賴`}
          style={{ display: 'block', width: '100%', height: 'auto', maxWidth: result.width, minWidth: Math.round(result.width * 0.75) }}>
          <defs>
            <marker id="aov-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill={colors.stroke} />
            </marker>
            <marker id="aov-arrow-added" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill={colors.added} />
            </marker>
          </defs>
          {edges.map((edge) => {
            const a = result.positions.get(edge.source)
            const b = result.positions.get(edge.target)
            if (!a || !b) return null
            const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2, x2 = b.x - 2, y2 = b.y + NODE_H / 2
            const mid = (x1 + x2) / 2
            const isAdded = addedEdges.has(edgeKey(edge))
            return (
              <path key={edgeKey(edge)} d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`} fill="none"
                stroke={isAdded ? colors.added : colors.stroke} strokeWidth={isAdded ? 2.4 : 1.4}
                markerEnd={`url(#${isAdded ? 'aov-arrow-added' : 'aov-arrow'})`} />
            )
          })}
          {nodes.map((node) => {
            const p = result.positions.get(node.node_id)!
            const isAdded = added.has(node.node_id)
            const isUpdated = updated.has(node.node_id)
            const stroke = isAdded ? colors.added : isUpdated ? colors.updated : colors.stroke
            const fill = isAdded ? colors.addedFill : isUpdated ? colors.updatedFill : colors.fill
            return (
              <g key={node.node_id}>
                <title>{`${node.node_id}｜${node.activity_type}\n${JSON.stringify(node.parameters ?? {})}`}</title>
                <rect x={p.x} y={p.y} width={NODE_W} height={NODE_H} rx={8} fill={fill} stroke={stroke}
                  strokeWidth={isAdded || isUpdated ? 2.2 : 1.2} />
                <text x={p.x + 12} y={p.y + 19} fontSize={13} fontWeight={700} fill={colors.text}>
                  {truncate(node.name || node.activity_type.replace(/^[a-z]+\./, ''), node.name ? 10 : 17)}
                </text>
                <text x={p.x + 12} y={p.y + 36} fontSize={11} fill={colors.muted} fontFamily="monospace">
                  {truncate(`${node.activity_type}・${node.node_id}`, 20)}
                </text>
              </g>
            )
          })}
        </svg>
      </Box>
      {change && (change.removed_nodes.length > 0 || change.removed_edges.length > 0) && (
        <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center">
          <Typography variant="body2" color="text.secondary">這次刪除：</Typography>
          {change.removed_nodes.map((id) => (
            <Chip key={`n-${id}`} size="small" color="error" variant="outlined" label={`活動 ${nameOf(id)}`} />
          ))}
          {change.removed_edges.map((e) => (
            <Chip key={`e-${edgeKey(e)}`} size="small" color="error" variant="outlined"
              label={`依賴 ${nameOf(e.source)} → ${nameOf(e.target)}`} />
          ))}
        </Stack>
      )}
    </Stack>
  )
}
