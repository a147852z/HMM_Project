import {
  Alert, Box, Chip, CircularProgress, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Typography,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { useEffect, useMemo, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import {
  GRANULARITY_LABEL, laneLabel, majority, pct, shortType, stateItems, textWidth, transitionText, wrapTwoLines,
} from '../components/fsmShared'
import { operationLabel, stateColor, tokenLabel } from '../labels'
import type { Granularity, TaskFsmLane, TaskFsmState, TaskFsmTransition, TaskStateMachine } from '../types'

type Selection =
  | { kind: 'state'; id: string; episodeId: string; version: number }
  | { kind: 'transition'; id: string; episodeId: string; step: number }

// 泳道圖的尺寸：每位解題者一列，橫軸是第幾版
const TW = 156
const TH = 60
const TGX = 58
const LANE_H = 132
const LABEL_W = 150
const TOP = 60
const LEFT = LABEL_W + 34

interface Point { x: number; y: number }

const nodePos = (s: TaskFsmState): Point => ({ x: LEFT + s.column * (TW + TGX), y: TOP + s.lane * LANE_H })
const laneCenter = (lane: number) => TOP + lane * LANE_H + TH / 2

/** 從框的中心往 toward 的方向，與框邊緣的交點。 */
function clip(center: Point, toward: Point, pad = 0): Point {
  const dx = toward.x - center.x
  const dy = toward.y - center.y
  if (dx === 0 && dy === 0) return center
  const scale = Math.min((TW / 2 + pad) / Math.max(Math.abs(dx), 1e-9), (TH / 2 + pad) / Math.max(Math.abs(dy), 1e-9))
  return { x: center.x + dx * scale, y: center.y + dy * scale }
}

function edgeGeometry(t: TaskFsmTransition, byId: Map<string, TaskFsmState>, transitions: TaskFsmTransition[]) {
  const sa = byId.get(t.source)!
  const sb = byId.get(t.target)!
  const a = nodePos(sa)
  const b = nodePos(sb)
  if (t.self_loop) {
    // 自迴圈放在框的左上方，讓出中間給上下兩列之間的轉移
    const x0 = a.x + TW * 0.1
    const x1 = a.x + TW * 0.4
    return { d: `M ${x0} ${a.y} C ${x0 - 12} ${a.y - 50}, ${x1 + 12} ${a.y - 50}, ${x1} ${a.y - 3}`, label: { x: a.x + TW * 0.25, y: a.y - 40 } }
  }
  const ca = { x: a.x + TW / 2, y: a.y + TH / 2 }
  const cb = { x: b.x + TW / 2, y: b.y + TH / 2 }
  const reverse = transitions.some((o) => o.source === t.target && o.target === t.source)
  if (sa.lane === sb.lane && sb.column === sa.column + 1 && !reverse) {
    const p0 = { x: a.x + TW, y: ca.y }
    const p1 = { x: b.x - 6, y: cb.y }
    return { d: `M ${p0.x} ${p0.y} L ${p1.x} ${p1.y}`, label: { x: (p0.x + p1.x) / 2, y: p0.y } }
  }
  if (sa.lane === sb.lane) {
    // 同一列：往右跳的弧線在上方，往左回頭的在下方
    const up = sb.column > sa.column
    const control = { x: (ca.x + cb.x) / 2, y: ca.y + (up ? -1 : 1) * (TH / 2 + 44) * (reverse && Math.abs(sb.column - sa.column) === 1 ? 0.7 : 1.4) }
    const p0 = clip(ca, control, 2)
    const p2 = clip(cb, control, 7)
    return { d: `M ${p0.x} ${p0.y} Q ${control.x} ${control.y} ${p2.x} ${p2.y}`,
      label: { x: 0.25 * p0.x + 0.5 * control.x + 0.25 * p2.x, y: 0.25 * p0.y + 0.5 * control.y + 0.25 * p2.y } }
  }
  if (sb.column > sa.column) {
    // 跨列往右：從框的右側出發，接到目標的左側
    const p0 = { x: a.x + TW, y: ca.y }
    const p3 = { x: b.x - 6, y: cb.y }
    const c1 = { x: p0.x + TGX * 0.9, y: p0.y }
    const c2 = { x: p3.x - TGX * 0.9, y: p3.y }
    return { d: `M ${p0.x} ${p0.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${p3.x} ${p3.y}`,
      label: { x: (p0.x + 3 * c1.x + 3 * c2.x + p3.x) / 8, y: (p0.y + 3 * c1.y + 3 * c2.y + p3.y) / 8 } }
  }
  // 跨列往左或同一欄：從框的上緣或下緣出發，經過列與列之間的空白
  const down = sb.lane > sa.lane
  const shift = reverse ? (sa.lane < sb.lane ? -30 : 30) : 0
  const p0 = { x: ca.x + shift, y: down ? a.y + TH : a.y }
  const p3 = { x: cb.x + shift, y: down ? b.y - 7 : b.y + TH + 7 }
  const lift = down ? 1 : -1
  const c1 = { x: p0.x, y: p0.y + lift * LANE_H * 0.6 }
  const c2 = { x: p3.x, y: p3.y - lift * LANE_H * 0.6 }
  // 來回兩條的標籤往各自的那一側錯開，避免疊在一起
  return { d: `M ${p0.x} ${p0.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${p3.x} ${p3.y}`,
    label: { x: (p0.x + 3 * c1.x + 3 * c2.x + p3.x) / 8 + Math.sign(shift) * 26, y: (p0.y + 3 * c1.y + 3 * c2.y + p3.y) / 8 } }
}

export function TaskStateMachineView({ caseId, taskId, granularity, showProb, showHidden, onModel }: {
  caseId: string
  taskId: string
  granularity: Granularity
  showProb: boolean
  showHidden: boolean
  onModel: (available: boolean) => void
}) {
  const [machine, setMachine] = useState<TaskStateMachine | null>(null)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [highlight, setHighlight] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError('')
    api.taskStateMachine(caseId, taskId, granularity)
      .then((result) => {
        setMachine(result)
        onModel(!!result.hmm)
        setHighlight((current) => (result.episodes.some((e) => e.episode_id === current) ? current : null))
        const first = result.episodes[0]
        setSelection(first ? { kind: 'state', id: first.path[0], episodeId: first.episode_id, version: 0 } : null)
      })
      .catch((e) => { setMachine(null); setError(errorText(e)) })
      .finally(() => setLoading(false))
  }, [caseId, taskId, granularity, onModel])

  if (error) return <Alert severity="error">{error}</Alert>
  if (!machine) return loading ? <CircularProgress size={24} /> : null
  const hiddenOn = showHidden && !!machine.hmm
  const { stats } = machine

  return (
    <Stack spacing={2.5}>
      <Paper sx={{ p: 2.5 }}>
        <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mb: 1.5 }} alignItems="center">
          <Chip size="small" color="primary" label={`${stats.episodes} 條演變過程・${stats.versions} 個版本 → ${stats.states} 個狀態`} />
          <Chip size="small" variant="outlined" label={`${stats.transitions} 種轉移`} />
          <Chip size="small" variant="outlined" label={`被 2 條以上共用的狀態 ${stats.shared_states} 個`} />
          <Chip size="small" variant="outlined" label={`停在原地 ${stats.self_loop_steps} 次`} />
          <Chip size="small" variant="outlined" label={`回到先前的狀態 ${stats.revisits} 次`} />
          {loading && <CircularProgress size={16} />}
          {!machine.hmm && <Typography variant="caption" color="text.secondary">這個案例沒有使用中的模型，不顯示隱藏狀態。</Typography>}
        </Stack>
        <SwimlaneDiagram machine={machine} showProb={showProb} showHidden={hiddenOn} selection={selection}
          onSelect={setSelection} highlight={highlight} onHighlight={setHighlight} />
        <Stack direction="row" gap={1.5} flexWrap="wrap" alignItems="center" sx={{ mt: 1.5 }}>
          <Typography variant="caption" color="text.secondary">
            每位解題者一列；狀態放在最早走到它的那一列（橫軸是第幾版），別人走到同一個狀態時就是路線交會。
            點選左側的解題者可以只凸顯他的路線；框右上角是經過的演變過程數；雙框＝某位解題者的最後一版。
          </Typography>
          {hiddenOn && machine.hmm && machine.hmm.states.map((state) => (
            <Chip key={state} size="small" variant="outlined" sx={{ borderColor: stateColor(state), color: stateColor(state) }}
              label={`${state}：${machine.hmm!.top_observations[state].slice(0, 2).map((o) => tokenLabel(o.observation)).join('、')}`} />
          ))}
        </Stack>
      </Paper>
      <LanePaths machine={machine} showHidden={hiddenOn} selection={selection} onSelect={setSelection} />
      {selection && <TaskDetail machine={machine} selection={selection} onSelect={setSelection} showHidden={hiddenOn} />}
    </Stack>
  )
}

function SwimlaneDiagram({ machine, showProb, showHidden, selection, onSelect, highlight, onHighlight }: {
  machine: TaskStateMachine
  showProb: boolean
  showHidden: boolean
  selection: Selection | null
  onSelect: (s: Selection) => void
  highlight: string | null
  onHighlight: (episodeId: string | null) => void
}) {
  const theme = useTheme()
  const neutral = theme.palette.text.secondary
  const byId = useMemo(() => new Map(machine.states.map((s) => [s.id, s])), [machine])
  const columns = Math.max(1, ...machine.states.map((s) => s.column + 1))
  const width = LEFT + columns * (TW + TGX) + 24
  const height = TOP + machine.episodes.length * LANE_H
  const onPath = useMemo(() => {
    const lane = machine.episodes.find((e) => e.episode_id === highlight)
    return lane ? new Set(lane.path) : null
  }, [machine, highlight])
  const colorOf = (counts: Record<string, number>) => {
    const state = showHidden ? majority(counts) : null
    return state ? stateColor(state) : neutral
  }
  const dim = (visible: boolean) => (visible ? 1 : 0.16)
  const markerColors = Array.from(new Set([neutral, ...(machine.hmm?.states ?? []).map(stateColor)]))
  const markerId = (color: string) => `task-arrow-${color.replace(/[^a-z0-9]/gi, '')}`
  const firstVisit = (s: TaskFsmState) => s.visits.find((v) => v.episode_id === highlight) ?? s.visits[0]
  const firstEvent = (t: TaskFsmTransition) => t.events.find((e) => e.episode_id === highlight) ?? t.events[0]

  return (
    <Box sx={{ overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.default' }}>
      {/* 維持原始大小、橫向捲動：版本多的題目寬度會超過畫面，縮小會讓文字無法閱讀 */}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}
        role="img" aria-label="同一題所有解題者的有限狀態機（泳道圖）">
        <defs>
          {markerColors.map((color) => (
            <marker key={color} id={markerId(color)} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10"
              markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
            </marker>
          ))}
        </defs>

        {machine.episodes.map((lane) => {
          const top = TOP + lane.lane * LANE_H - (LANE_H - TH) / 2
          const active = highlight === lane.episode_id
          return (
            <g key={lane.episode_id} style={{ cursor: 'pointer' }} onClick={() => onHighlight(active ? null : lane.episode_id)}>
              <rect x={0} y={top} width={width} height={LANE_H} fill={lane.lane % 2 ? theme.palette.action.hover : 'transparent'} />
              <rect x={8} y={laneCenter(lane.lane) - 24} width={LABEL_W - 4} height={48} rx={8}
                fill={active ? theme.palette.primary.main : theme.palette.background.paper}
                stroke={active ? theme.palette.primary.main : theme.palette.divider} />
              <text x={20} y={laneCenter(lane.lane) - 4} fontSize={13} fontWeight={700}
                fill={active ? theme.palette.primary.contrastText : theme.palette.text.primary}>
                {laneLabel(lane.teacher, lane.scope, lane.episode_id)}
              </text>
              <text x={20} y={laneCenter(lane.lane) + 14} fontSize={11}
                fill={active ? theme.palette.primary.contrastText : neutral}>{`${lane.events} 次修改`}</text>
            </g>
          )
        })}

        {machine.episodes.map((lane) => {
          const start = byId.get(lane.path[0])!
          const p = nodePos(start)
          const visible = !highlight || highlight === lane.episode_id
          const from = { x: LABEL_W + 8, y: laneCenter(lane.lane) }
          const to = start.lane === lane.lane ? { x: p.x - 4, y: p.y + TH / 2 } : clip({ x: p.x + TW / 2, y: p.y + TH / 2 }, from, 5)
          const d = start.lane === lane.lane
            ? `M ${from.x} ${from.y} L ${to.x} ${to.y}`
            : `M ${from.x} ${from.y} C ${from.x + 40} ${from.y}, ${to.x - 40} ${to.y}, ${to.x} ${to.y}`
          return (
            <path key={`start-${lane.episode_id}`} d={d} fill="none" stroke={neutral} strokeWidth={1.5} strokeDasharray="4 3"
              opacity={dim(visible)} markerEnd={`url(#${markerId(neutral)})`} />
          )
        })}

        {machine.transitions.map((t) => {
          const { d } = edgeGeometry(t, byId, machine.transitions)
          const color = colorOf(t.hidden_states)
          const selected = selection?.kind === 'transition' && selection.id === t.id
          const visible = !highlight || t.episodes.includes(highlight)
          const event = firstEvent(t)
          return (
            <g key={t.id} style={{ cursor: 'pointer' }} opacity={dim(visible)}
              onClick={() => onSelect({ kind: 'transition', id: t.id, episodeId: event.episode_id, step: event.step })}>
              <title>{`${t.source} → ${t.target}：${Object.entries(t.operations).map(([op, n]) => `${operationLabel(op)} ${n} 次`).join('、')}（${pct(t.probability)}，${t.episodes.length} 條演變過程）`}</title>
              <path d={d} fill="none" stroke="transparent" strokeWidth={14} />
              <path d={d} fill="none" stroke={color} strokeOpacity={selected ? 1 : 0.85}
                strokeWidth={Math.min(6, 1.3 + 1.1 * t.count) + (selected ? 1.5 : 0)} markerEnd={`url(#${markerId(color)})`} />
            </g>
          )
        })}

        {machine.states.map((s) => {
          const p = nodePos(s)
          const color = colorOf(s.hidden_states)
          const selected = selection?.kind === 'state' && selection.id === s.id
          const visible = !onPath || onPath.has(s.id)
          const [line1, line2] = wrapTwoLines(stateItems(s.summary, machine.granularity), TW - 22, 11)
          const visit = firstVisit(s)
          return (
            <g key={s.id} style={{ cursor: 'pointer' }} opacity={dim(visible)}
              onClick={() => onSelect({ kind: 'state', id: s.id, episodeId: visit.episode_id, version: visit.version })}>
              <title>{`${s.id}（${s.episodes.length} 條演變過程經過）：${stateItems(s.summary, machine.granularity).join(' → ')}`}</title>
              {s.is_final && <rect x={p.x - 5} y={p.y - 5} width={TW + 10} height={TH + 10} rx={13} fill="none" stroke={color} strokeWidth={1.5} />}
              <rect x={p.x} y={p.y} width={TW} height={TH} rx={9} fill={theme.palette.background.paper} />
              <rect x={p.x} y={p.y} width={TW} height={TH} rx={9} fill={color} fillOpacity={color === neutral ? 0.04 : 0.13}
                stroke={color} strokeWidth={selected ? 3 : s.episodes.length > 1 ? 2.2 : 1.3} />
              <text x={p.x + 10} y={p.y + 18} fontSize={13} fontWeight={700} fill={theme.palette.text.primary}>{s.id}</text>
              <text x={p.x + TW - 10} y={p.y + 18} fontSize={10.5} textAnchor="end" fill={s.episodes.length > 1 ? theme.palette.primary.main : neutral}
                fontWeight={s.episodes.length > 1 ? 700 : 400}>
                {s.episodes.length > 1 ? `${s.episodes.length} 條共用` : `第 ${s.column} 版`}
              </text>
              <text x={p.x + 10} y={p.y + 36} fontSize={11} fill={theme.palette.text.primary}>{line1}</text>
              <text x={p.x + 10} y={p.y + 51} fontSize={11} fill={theme.palette.text.primary}>{line2}</text>
            </g>
          )
        })}

        {machine.transitions.map((t) => {
          const visible = !highlight || t.episodes.includes(highlight)
          if (highlight && !visible) return null
          const { label } = edgeGeometry(t, byId, machine.transitions)
          const [first, second] = transitionText(t, showProb)
          const w = Math.max(textWidth(first, 10), textWidth(second, 10)) + 12
          const event = firstEvent(t)
          return (
            <g key={`${t.id}-label`} style={{ cursor: 'pointer' }}
              onClick={() => onSelect({ kind: 'transition', id: t.id, episodeId: event.episode_id, step: event.step })}>
              <rect x={label.x - w / 2} y={label.y - (second ? 15 : 9)} width={w} height={second ? 30 : 18} rx={5}
                fill={theme.palette.background.paper} fillOpacity={0.92} stroke={theme.palette.divider} />
              <text x={label.x} y={label.y + (second ? -3 : 4)} fontSize={10} textAnchor="middle" fill={theme.palette.text.primary}>{first}</text>
              {second && <text x={label.x} y={label.y + 10} fontSize={10} fontWeight={700} textAnchor="middle" fill={neutral}>{second}</text>}
            </g>
          )
        })}
      </svg>
    </Box>
  )
}

function LanePaths({ machine, showHidden, selection, onSelect }: {
  machine: TaskStateMachine; showHidden: boolean; selection: Selection | null; onSelect: (s: Selection) => void
}) {
  const eventsOf = (lane: TaskFsmLane) => machine.transitions
    .flatMap((t) => t.events.filter((e) => e.episode_id === lane.episode_id).map((e) => ({ ...e, transition: t.id })))
    .sort((a, b) => a.step - b.step)
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>各解題者的演變路徑</Typography>
      <Stack spacing={1.25}>
        {machine.episodes.map((lane) => {
          const events = eventsOf(lane)
          return (
            <Stack key={lane.episode_id} direction="row" gap={0.75} flexWrap="wrap" alignItems="center">
              <Typography variant="body2" fontWeight={700} sx={{ minWidth: 120 }}>
                {laneLabel(lane.teacher, lane.scope, lane.episode_id)}
              </Typography>
              {lane.path.map((stateId, version) => {
                const event = version > 0 ? events[version - 1] : null
                const color = showHidden && event?.hidden_state ? stateColor(event.hidden_state) : undefined
                const selected = selection?.kind === 'state' && selection.id === stateId
                  && selection.episodeId === lane.episode_id && selection.version === version
                return (
                  <Stack key={version} direction="row" gap={0.75} alignItems="center">
                    {event && (
                      <Chip size="small" variant="outlined" clickable label={`→ ${operationLabel(event.operation_type)}`}
                        onClick={() => onSelect({ kind: 'transition', id: event.transition, episodeId: lane.episode_id, step: event.step })}
                        sx={{ borderStyle: 'dashed', ...(color ? { color, borderColor: color } : {}) }} />
                    )}
                    <Chip size="small" clickable color={selected ? 'primary' : 'default'} label={`第 ${version} 版・${stateId}`}
                      onClick={() => onSelect({ kind: 'state', id: stateId, episodeId: lane.episode_id, version })} />
                  </Stack>
                )
              })}
            </Stack>
          )
        })}
      </Stack>
    </Paper>
  )
}

function TaskDetail({ machine, selection, onSelect, showHidden }: {
  machine: TaskStateMachine; selection: Selection; onSelect: (s: Selection) => void; showHidden: boolean
}) {
  const laneOf = (episodeId: string) => machine.episodes.find((e) => e.episode_id === episodeId)
  const nameOf = (episodeId: string) => {
    const lane = laneOf(episodeId)
    return lane ? laneLabel(lane.teacher, lane.scope, lane.episode_id) : episodeId
  }

  if (selection.kind === 'state') {
    const state = machine.states.find((s) => s.id === selection.id)
    if (!state) return null
    const lane = laneOf(selection.episodeId)
    const version = lane?.versions[selection.version]
    return (
      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>
          狀態 {state.id}（{state.episodes.length} 條演變過程經過）
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {machine.granularity === 'category' ? '用到的節點類別：' : '主要節點（依流程順序）：'}
          {stateItems(state.summary, machine.granularity).join(' → ')}
        </Typography>
        {(state.starts.length > 0 || state.finals.length > 0) && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {state.starts.length > 0 && `起點：${state.starts.map(nameOf).join('、')}。`}
            {state.finals.length > 0 && `最後一版：${state.finals.map(nameOf).join('、')}。`}
          </Typography>
        )}
        <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center" sx={{ my: 1.5 }}>
          <Typography variant="body2" color="text.secondary">經過的版本：</Typography>
          {state.visits.map((v) => (
            <Chip key={`${v.episode_id}-${v.version}`} size="small" clickable
              color={v.episode_id === selection.episodeId && v.version === selection.version ? 'primary' : 'default'}
              label={`${nameOf(v.episode_id)}・第 ${v.version} 版`}
              onClick={() => onSelect({ kind: 'state', id: state.id, episodeId: v.episode_id, version: v.version })} />
          ))}
          {showHidden && Object.entries(state.hidden_states).map(([h, n]) => (
            <Chip key={h} size="small" variant="outlined" sx={{ borderColor: stateColor(h), color: stateColor(h) }} label={`${h}：${n} 次`} />
          ))}
        </Stack>
        {machine.granularity !== 'full' && state.visits.length > 1 && (
          <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
            這些版本在「{GRANULARITY_LABEL[machine.granularity]}」下算同一個狀態，參數或節點細節可能不同；
            下圖是{nameOf(selection.episodeId)}的第 {selection.version} 版。
          </Typography>
        )}
        {version && <AoVGraph nodes={version.nodes} edges={version.edges} />}
      </Paper>
    )
  }

  const transition = machine.transitions.find((t) => t.id === selection.id)
  if (!transition) return null
  const event = transition.events.find((e) => e.episode_id === selection.episodeId && e.step === selection.step) ?? transition.events[0]
  const lane = laneOf(event.episode_id)
  const before = lane?.versions[event.from_version]
  const after = lane?.versions[event.to_version]
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="subtitle1" fontWeight={700}>
        轉移 {transition.source} → {transition.target}{transition.self_loop ? '（停在原地）' : ''}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        {transition.count} 次修改、{transition.episodes.length} 條演變過程（{transition.episodes.map(nameOf).join('、')}），
        佔 {transition.source} 出去的修改 {pct(transition.probability)}。
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>解題者</TableCell>
              <TableCell>第幾次修改</TableCell>
              <TableCell>修改</TableCell>
              {showHidden && <TableCell>隱藏狀態（後驗）</TableCell>}
              {showHidden && <TableCell>隱藏狀態轉移</TableCell>}
              {machine.hmm && <TableCell align="right">模型當時的預測</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {transition.events.map((e) => (
              <TableRow key={`${e.episode_id}-${e.step}`} hover sx={{ cursor: 'pointer' }}
                selected={e.episode_id === event.episode_id && e.step === event.step}
                onClick={() => onSelect({ kind: 'transition', id: transition.id, episodeId: e.episode_id, step: e.step })}>
                <TableCell>{nameOf(e.episode_id)}</TableCell>
                <TableCell>第 {e.step} 次（第 {e.from_version} → {e.to_version} 版）</TableCell>
                <TableCell>{operationLabel(e.operation_type)}{e.activity_type ? `・${shortType(e.activity_type)}` : ''}</TableCell>
                {showHidden && (
                  <TableCell>
                    {e.hidden_state && (
                      <Chip size="small" variant="outlined" sx={{ borderColor: stateColor(e.hidden_state), color: stateColor(e.hidden_state) }}
                        label={`${e.hidden_state}（${pct(e.posterior?.[e.hidden_state])}）`} />
                    )}
                  </TableCell>
                )}
                {showHidden && (
                  <TableCell>
                    {e.hidden_transition && (e.hidden_transition.from
                      ? `${e.hidden_transition.from} → ${e.hidden_transition.to}：${e.hidden_transition.probability.toFixed(3)}`
                      : `初始 ${e.hidden_transition.to}：${e.hidden_transition.probability.toFixed(3)}`)}
                  </TableCell>
                )}
                {machine.hmm && (
                  <TableCell align="right">
                    {e.model_probability !== undefined ? `${(e.model_probability * 100).toFixed(1)}%・第 ${e.model_rank} 名` : '—'}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      {event.reason && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" fontWeight={700}>修改理由（{nameOf(event.episode_id)}，第 {event.step} 次）</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', maxHeight: 140, overflow: 'auto' }}>
            {event.reason}
          </Typography>
        </Box>
      )}
      {after && before && (
        <>
          <Typography variant="body2" fontWeight={700} sx={{ mt: 2, mb: 1 }}>
            {nameOf(event.episode_id)}的第 {after.index} 版（綠色＝新增、橘色＝更新，刪除的部分列在圖下方）
          </Typography>
          <AoVGraph nodes={after.nodes} edges={after.edges} change={event.change} beforeNodes={before.nodes} />
        </>
      )}
    </Paper>
  )
}
