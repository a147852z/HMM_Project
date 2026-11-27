import {
  Alert, Autocomplete, Box, Chip, CircularProgress, FormControlLabel, Paper, Stack, Switch, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import {
  GRANULARITY_HELP, GRANULARITY_LABEL, majority, pct, shortType, stateItems, textWidth, transitionText, wrapTwoLines,
} from '../components/fsmShared'
import { operationLabel, stateColor, tokenLabel } from '../labels'
import { TaskStateMachineView } from './TaskStateMachineView'
import { TeacherStateMachineView } from './TeacherStateMachineView'
import type { EpisodeSummary, FsmState, FsmTransition, Granularity, StateMachine } from '../types'

const STORE_KEY = (caseId: string) => `kcore.fsm.${caseId}`
const SCOPE_KEY = (caseId: string) => `kcore.fsm.scope.${caseId}`
const TEACHER_KEY = (caseId: string) => `kcore.fsm.teacher.${caseId}`
type Scope = 'episode' | 'task' | 'teacher'

const SCOPE_TITLE: Record<Scope, string> = {
  episode: '有限狀態機：一條演變過程',
  task: '有限狀態機：同一題所有解題者',
  teacher: '有限狀態機：同一個 AI 的所有題目',
}
const SCOPE_HELP: Record<Scope, string> = {
  episode: '一條演變過程是一條馬可夫鏈：每個狀態是一張流程圖（AoV），每次修改是一次轉移。轉移機率是這條演變過程裡，從同一個狀態出去的修改中走這條轉移的比例。',
  task: '把同一題所有解題者的演變過程合併成一個狀態機：每位解題者一列，相同的流程圖算同一個狀態，別人走到同一個狀態時就是路線交會。轉移機率用所有演變過程累計。',
  teacher: '把同一個解題 AI 在所有題目的演變過程合併成一個狀態機：看它習慣從哪種流程開始、往哪裡改、停在哪裡，以及哪些狀態比較容易通過。不同題目的完整流程圖幾乎不會相同，所以預設用「節點類別」分狀態。',
}

function storedScope(caseId: string): Scope {
  const value = readStored(SCOPE_KEY(caseId))
  return value === 'task' || value === 'teacher' ? value : 'episode'
}

type Selection = { kind: 'state'; id: string; visit: number } | { kind: 'transition'; id: string; step: number }

function readStored(key: string) {
  try { return localStorage.getItem(key) ?? '' } catch { return '' }
}
function writeStored(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* 瀏覽器不允許儲存時忽略 */ }
}

/** 狀態摘要依寬度拆成兩行，用於狀態框內。 */
function summaryLines(state: FsmState, granularity: Granularity): [string, string] {
  return wrapTwoLines(stateItems(state.summary, granularity), W - 26, 12)
}

export function StateMachinePage({ caseId }: { caseId: string }) {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [allEpisodes, setAllEpisodes] = useState<EpisodeSummary[]>([])
  const [scope, setScopeState] = useState<Scope>(() => storedScope(caseId))
  const [episodeId, setEpisodeId] = useState('')
  const [taskId, setTaskId] = useState('')
  const [teacher, setTeacher] = useState('')
  const [granularity, setGranularity] = useState<Granularity>(() => (storedScope(caseId) === 'teacher' ? 'category' : 'structure'))
  const [showProb, setShowProb] = useState(true)
  const [showHidden, setShowHidden] = useState(true)
  const [machine, setMachine] = useState<StateMachine | null>(null)
  const [taskHasModel, setTaskHasModel] = useState(true)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setEpisodes([])
    setAllEpisodes([])
    setMachine(null)
    api.listEpisodes(caseId)
      .then((list) => {
        const usable = list.filter((e) => e.event_count > 0)
        setEpisodes(usable)
        setAllEpisodes(list)
        const stored = readStored(STORE_KEY(caseId))
        const initial = usable.find((e) => e.episode_id === stored) ?? usable.find((e) => e.event_count >= 8) ?? usable[0]
        setEpisodeId(initial?.episode_id ?? '')
        setTaskId(initial?.work_order_id ?? '')
      })
      .catch((e) => setError(errorText(e)))
  }, [caseId])

  // 同一題（work_order_id 相同）的演變過程分組；演變過程多的題目排前面
  const tasks = useMemo(() => {
    const groups = new Map<string, EpisodeSummary[]>()
    for (const e of episodes) {
      if (!e.work_order_id) continue
      groups.set(e.work_order_id, [...(groups.get(e.work_order_id) ?? []), e])
    }
    return [...groups.entries()]
      .map(([id, list]) => ({ id, count: list.length, teachers: Array.from(new Set(list.map((e) => String(e.provenance?.teacher ?? '—')))) }))
      .sort((a, b) => b.count - a.count || a.id.localeCompare(b.id))
  }, [episodes])

  // 解題 AI（provenance.teacher）與各自的解題次數、題數；解題多的排前面
  const teachers = useMemo(() => {
    const groups = new Map<string, { count: number; tasks: Set<string> }>()
    for (const e of allEpisodes) {
      const name = e.provenance?.teacher
      if (!name) continue
      const group = groups.get(name) ?? { count: 0, tasks: new Set<string>() }
      group.count += 1
      if (e.work_order_id) group.tasks.add(e.work_order_id)
      groups.set(name, group)
    }
    return [...groups.entries()].map(([id, g]) => ({ id, count: g.count, tasks: g.tasks.size }))
      .sort((a, b) => b.count - a.count || a.id.localeCompare(b.id))
  }, [allEpisodes])

  useEffect(() => {
    if (teachers.length === 0 || teachers.some((t) => t.id === teacher)) return
    // 先用上次選的；沒有時預設 opus（先看 Opus 的狀態機），再沒有就用解題最多的
    const stored = readStored(TEACHER_KEY(caseId))
    setTeacher((teachers.find((t) => t.id === stored) ?? teachers.find((t) => t.id === 'opus') ?? teachers[0]).id)
  }, [teachers, teacher, caseId])

  useEffect(() => {
    if (teacher) writeStored(TEACHER_KEY(caseId), teacher)
  }, [teacher, caseId])

  useEffect(() => {
    if (scope !== 'episode' || !episodeId) return
    writeStored(STORE_KEY(caseId), episodeId)
    setLoading(true)
    setError('')
    api.stateMachine(caseId, episodeId, granularity)
      .then((result) => {
        setMachine(result)
        const start = result.states.find((s) => s.is_start)
        setSelection(start ? { kind: 'state', id: start.id, visit: start.visits[0] } : null)
      })
      .catch((e) => { setMachine(null); setError(errorText(e)) })
      .finally(() => setLoading(false))
  }, [caseId, episodeId, granularity, scope])

  const selectedEpisode = episodes.find((e) => e.episode_id === episodeId) ?? null
  const selectedTask = tasks.find((t) => t.id === taskId) ?? null
  const selectedTeacher = teachers.find((t) => t.id === teacher) ?? null
  const setScope = (value: Scope) => {
    setScopeState(value)
    writeStored(SCOPE_KEY(caseId), value)
    // 切到「同一題」時，預設用目前這條演變過程所屬的題目
    if (value === 'task' && selectedEpisode?.work_order_id) setTaskId(selectedEpisode.work_order_id)
    // 跨題目合併時，完整流程圖與結構幾乎不會重複，預設改用節點類別
    if (value === 'teacher') setGranularity('category')
  }
  const openEpisode = useCallback((id: string) => {
    setEpisodeId(id)
    setScopeState('episode')
    writeStored(SCOPE_KEY(caseId), 'episode')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [caseId])
  const hasModel = scope === 'episode' ? !!machine?.hmm : taskHasModel
  const hiddenOn = showHidden && !!machine?.hmm

  return (
    <Stack spacing={2.5}>
      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>{SCOPE_TITLE[scope]}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{SCOPE_HELP[scope]}</Typography>
        <Stack direction={{ xs: 'column', lg: 'row' }} gap={2} alignItems={{ lg: 'center' }} flexWrap="wrap">
          <ToggleButtonGroup size="small" exclusive value={scope}
            onChange={(_, value: Scope | null) => value && setScope(value)} aria-label="範圍">
            <ToggleButton value="episode" sx={{ px: 1.5 }}>單一演變過程</ToggleButton>
            <ToggleButton value="task" sx={{ px: 1.5 }}>同一題所有解題者</ToggleButton>
            <ToggleButton value="teacher" sx={{ px: 1.5 }}>同一個 AI 的所有題目</ToggleButton>
          </ToggleButtonGroup>
          {scope === 'teacher' ? (
            <Autocomplete
              id="fsm-teacher" size="small" sx={{ flex: 1, minWidth: 260 }}
              options={teachers} value={selectedTeacher}
              onChange={(_, value) => value && setTeacher(value.id)}
              getOptionLabel={(t) => `${t.id}（${t.count} 次解題・${t.tasks} 題）`}
              isOptionEqualToValue={(a, b) => a.id === b.id}
              renderInput={(params) => <TextField {...params} label={`解題 AI（${teachers.length} 個）`} />}
            />
          ) : scope === 'episode' ? (
            <Autocomplete
              id="fsm-episode" size="small" sx={{ flex: 1, minWidth: 320 }}
              options={episodes} value={selectedEpisode}
              onChange={(_, value) => value && setEpisodeId(value.episode_id)}
              getOptionLabel={(e) => `${e.work_order_id ?? e.episode_id}（${String(e.provenance?.teacher ?? '—')}・${e.event_count} 次修改）`}
              isOptionEqualToValue={(a, b) => a.episode_id === b.episode_id}
              renderInput={(params) => <TextField {...params} label={`演變過程（${episodes.length} 條）`} />}
            />
          ) : (
            <Autocomplete
              id="fsm-task" size="small" sx={{ flex: 1, minWidth: 320 }}
              options={tasks} value={selectedTask}
              onChange={(_, value) => value && setTaskId(value.id)}
              getOptionLabel={(t) => `${t.id}（${t.count} 條演變過程：${t.teachers.join('、')}）`}
              isOptionEqualToValue={(a, b) => a.id === b.id}
              renderInput={(params) => <TextField {...params} label={`題目（${tasks.length} 題）`} />}
            />
          )}
          <ToggleButtonGroup size="small" exclusive value={granularity}
            onChange={(_, value: Granularity | null) => value && setGranularity(value)} aria-label="狀態的粒度">
            {(Object.keys(GRANULARITY_LABEL) as Granularity[]).map((g) => (
              <ToggleButton key={g} value={g} sx={{ px: 1.5 }}>{GRANULARITY_LABEL[g]}</ToggleButton>
            ))}
          </ToggleButtonGroup>
          <FormControlLabel control={<Switch id="fsm-prob" checked={showProb} onChange={(e) => setShowProb(e.target.checked)} />}
            label="轉移機率" />
          <FormControlLabel
            control={<Switch id="fsm-hidden" checked={showHidden} disabled={!hasModel}
              onChange={(e) => setShowHidden(e.target.checked)} />}
            label="HMM 隱藏狀態" />
        </Stack>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1 }}>
          {GRANULARITY_HELP[granularity]}
        </Typography>
      </Paper>

      {scope === 'task' && taskId && (
        <TaskStateMachineView caseId={caseId} taskId={taskId} granularity={granularity} showProb={showProb}
          showHidden={showHidden} onModel={setTaskHasModel} />
      )}

      {scope === 'teacher' && teacher && (
        <TeacherStateMachineView caseId={caseId} teacher={teacher} granularity={granularity} showProb={showProb}
          showHidden={showHidden} onModel={setTaskHasModel} onOpenEpisode={openEpisode} />
      )}

      {scope === 'episode' && error && <Alert severity="error">{error}</Alert>}
      {scope === 'episode' && loading && !machine && <CircularProgress size={24} />}
      {scope === 'episode' && machine && (
        <>
          <Paper sx={{ p: 2.5 }}>
            <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mb: 1.5 }} alignItems="center">
              <Chip size="small" label={`${machine.stats.versions} 個版本 → ${machine.stats.states} 個狀態`} color="primary" />
              <Chip size="small" variant="outlined" label={`${machine.stats.transitions} 種轉移`} />
              <Chip size="small" variant="outlined" label={`停在原地 ${machine.stats.self_loop_steps} 次`} />
              <Chip size="small" variant="outlined" label={`回到先前的狀態 ${machine.stats.revisits} 次`} />
              {loading && <CircularProgress size={16} />}
              {!machine.hmm && <Typography variant="caption" color="text.secondary">這個案例沒有使用中的模型，不顯示隱藏狀態。</Typography>}
            </Stack>
            <FsmDiagram machine={machine} showProb={showProb} showHidden={hiddenOn} selection={selection} onSelect={setSelection} />
            <Legend machine={machine} showHidden={hiddenOn} />
          </Paper>
          <PathStrip machine={machine} showHidden={hiddenOn} selection={selection} onSelect={setSelection} />
          {selection && <Detail machine={machine} selection={selection} onSelect={setSelection} showHidden={hiddenOn} />}
        </>
      )}
    </Stack>
  )
}

// ------------------------------------------------------------------ 狀態圖（SVG）
const W = 176
const H = 68
const GX = 96
const GY = 112
const LEFT = 64
const TOP = 76

interface Point { x: number; y: number }

function layoutStates(states: FsmState[]) {
  const perRow = Math.min(5, Math.max(1, states.length))
  const pos = new Map<string, Point & { row: number; col: number }>()
  states.forEach((s, i) => {
    // 蛇形排列：偶數列由左往右、奇數列由右往左，換列時下一個狀態就在正下方
    const row = Math.floor(i / perRow)
    const col = row % 2 === 0 ? i % perRow : perRow - 1 - (i % perRow)
    pos.set(s.id, { x: LEFT + col * (W + GX), y: TOP + row * (H + GY), row, col })
  })
  const rows = Math.ceil(states.length / perRow)
  return { pos, width: LEFT + perRow * W + (perRow - 1) * GX + 40, height: TOP + rows * H + (rows - 1) * GY + 56 }
}

/** 從狀態框中心往 (tx, ty) 的方向，與框邊緣的交點。 */
function clip(center: Point, toward: Point, pad = 0): Point {
  const dx = toward.x - center.x
  const dy = toward.y - center.y
  if (dx === 0 && dy === 0) return center
  const scale = Math.min((W / 2 + pad) / Math.max(Math.abs(dx), 1e-9), (H / 2 + pad) / Math.max(Math.abs(dy), 1e-9))
  return { x: center.x + dx * scale, y: center.y + dy * scale }
}

function edgeGeometry(t: FsmTransition, machine: StateMachine, pos: ReturnType<typeof layoutStates>['pos']) {
  const a = pos.get(t.source)!
  const b = pos.get(t.target)!
  if (t.self_loop) {
    const x0 = a.x + W * 0.56
    const x1 = a.x + W * 0.86
    return { d: `M ${x0} ${a.y} C ${x0 - 12} ${a.y - 62}, ${x1 + 12} ${a.y - 62}, ${x1} ${a.y - 3}`, label: { x: a.x + W * 0.71, y: a.y - 50 } }
  }
  const ca = { x: a.x + W / 2, y: a.y + H / 2 }
  const cb = { x: b.x + W / 2, y: b.y + H / 2 }
  const order = (id: string) => machine.states.find((s) => s.id === id)!.order
  const forward = order(t.source) < order(t.target)
  // 法向量固定用「編號小 → 編號大」的方向：往後跳的弧線在上方，往回走的在下方，來回兩條不會重疊
  const lo = forward ? ca : cb
  const hi = forward ? cb : ca
  const dist = Math.hypot(hi.x - lo.x, hi.y - lo.y) || 1
  const normal = { x: -(hi.y - lo.y) / dist, y: (hi.x - lo.x) / dist }
  const adjacent = (a.row === b.row && Math.abs(a.col - b.col) === 1) || (a.col === b.col && Math.abs(a.row - b.row) === 1)
  const reverse = machine.transitions.some((o) => o.source === t.target && o.target === t.source)
  if (a.row !== b.row && !adjacent) {
    // 跨列：從框的上緣或下緣出發，經過兩列之間的空白再接到目標，不穿過其他狀態；來回兩條左右錯開
    const down = b.row > a.row
    const shift = reverse ? (forward ? -18 : 18) : 0
    const p0 = { x: ca.x + shift, y: down ? a.y + H : a.y }
    const p3 = { x: cb.x + shift, y: down ? b.y - 7 : b.y + H + 7 }
    const lift = down ? 1 : -1
    const c1 = { x: p0.x, y: p0.y + lift * GY * 0.75 }
    const c2 = { x: p3.x, y: p3.y - lift * GY * 0.75 }
    const label = { x: (p0.x + 3 * c1.x + 3 * c2.x + p3.x) / 8, y: (p0.y + 3 * c1.y + 3 * c2.y + p3.y) / 8 }
    return { d: `M ${p0.x} ${p0.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${p3.x} ${p3.y}`, label }
  }
  // 同一列跳過中間的狀態：弧線拉高到越過狀態框；相鄰兩個狀態之間有來回兩條時，各自稍微彎開
  const bend = adjacent
    ? (reverse ? (forward ? -1 : 1) * Math.min(78, 30 + dist * 0.14) : 0)
    : (forward ? -1 : 1) * (H + 48)
  const control = { x: (ca.x + cb.x) / 2 + normal.x * bend, y: (ca.y + cb.y) / 2 + normal.y * bend }
  const p0 = clip(ca, control, 2)
  const p2 = clip(cb, control, 7)
  const label = { x: 0.25 * p0.x + 0.5 * control.x + 0.25 * p2.x, y: 0.25 * p0.y + 0.5 * control.y + 0.25 * p2.y }
  return { d: `M ${p0.x} ${p0.y} Q ${control.x} ${control.y} ${p2.x} ${p2.y}`, label }
}

function FsmDiagram({ machine, showProb, showHidden, selection, onSelect }: {
  machine: StateMachine
  showProb: boolean
  showHidden: boolean
  selection: Selection | null
  onSelect: (s: Selection) => void
}) {
  const theme = useTheme()
  const neutral = theme.palette.text.secondary
  const { pos, width, height } = useMemo(() => layoutStates(machine.states), [machine])
  const colorOf = (counts: Record<string, number>) => {
    const state = showHidden ? majority(counts) : null
    return state ? stateColor(state) : neutral
  }
  const markerColors = Array.from(new Set([neutral, ...(machine.hmm?.states ?? []).map(stateColor)]))
  const markerId = (color: string) => `fsm-arrow-${color.replace(/[^a-z0-9]/gi, '')}`
  const start = machine.states.find((s) => s.is_start)

  return (
    <Box sx={{ overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.default' }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ minWidth: Math.min(width, 980), display: 'block' }}
        role="img" aria-label="演變過程的有限狀態機">
        <defs>
          {markerColors.map((color) => (
            <marker key={color} id={markerId(color)} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11"
              markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
            </marker>
          ))}
        </defs>

        {start && (() => {
          const p = pos.get(start.id)!
          return (
            <g>
              <line x1={p.x - 46} y1={p.y + H / 2} x2={p.x - 4} y2={p.y + H / 2} stroke={neutral} strokeWidth={2}
                markerEnd={`url(#${markerId(neutral)})`} />
              <text x={p.x - 46} y={p.y + H / 2 - 8} fontSize={11} fill={neutral}>開始</text>
            </g>
          )
        })()}

        {machine.transitions.map((t) => {
          const { d } = edgeGeometry(t, machine, pos)
          const color = colorOf(t.hidden_states)
          const selected = selection?.kind === 'transition' && selection.id === t.id
          return (
            <g key={t.id} style={{ cursor: 'pointer' }}
              onClick={() => onSelect({ kind: 'transition', id: t.id, step: t.events[0].step })}>
              <title>{`${t.source} → ${t.target}：${Object.entries(t.operations).map(([op, n]) => `${operationLabel(op)} ${n} 次`).join('、')}（${pct(t.probability)}）`}</title>
              <path d={d} fill="none" stroke="transparent" strokeWidth={16} />
              <path d={d} fill="none" stroke={color} strokeOpacity={selected ? 1 : 0.85}
                strokeWidth={Math.min(6, 1.4 + 1.2 * t.count) + (selected ? 1.5 : 0)} markerEnd={`url(#${markerId(color)})`} />
            </g>
          )
        })}

        {machine.states.map((s) => {
          const p = pos.get(s.id)!
          const color = colorOf(s.hidden_states)
          const selected = selection?.kind === 'state' && selection.id === s.id
          const [line1, line2] = summaryLines(s, machine.granularity)
          return (
            <g key={s.id} style={{ cursor: 'pointer' }} onClick={() => onSelect({ kind: 'state', id: s.id, visit: s.visits[0] })}>
              <title>{`${s.id}（第 ${s.visits.join('、')} 版）：${stateItems(s.summary, machine.granularity).join(' → ')}`}</title>
              {s.is_final && <rect x={p.x - 5} y={p.y - 5} width={W + 10} height={H + 10} rx={14} fill="none" stroke={color} strokeWidth={1.5} />}
              <rect x={p.x} y={p.y} width={W} height={H} rx={10} fill={theme.palette.background.paper} />
              <rect x={p.x} y={p.y} width={W} height={H} rx={10} fill={color} fillOpacity={color === neutral ? 0.04 : 0.13}
                stroke={color} strokeWidth={selected ? 3 : 1.5} />
              <text x={p.x + 12} y={p.y + 21} fontSize={14} fontWeight={700} fill={theme.palette.text.primary}>{s.id}</text>
              <text x={p.x + W - 12} y={p.y + 21} fontSize={11} textAnchor="end" fill={neutral}>
                {s.visits.length > 1 ? `經過 ${s.visits.length} 次` : `第 ${s.visits[0]} 版`}
              </text>
              <text x={p.x + 12} y={p.y + 41} fontSize={12} fill={theme.palette.text.primary}>{line1}</text>
              <text x={p.x + 12} y={p.y + 58} fontSize={12} fill={theme.palette.text.primary}>{line2}</text>
            </g>
          )
        })}

        {machine.transitions.map((t) => {
          const { label } = edgeGeometry(t, machine, pos)
          const [first, second] = transitionText(t, showProb)
          const width = Math.max(textWidth(first, 11), textWidth(second, 11)) + 16
          return (
            <g key={`${t.id}-label`} style={{ cursor: 'pointer' }} pointerEvents="all"
              onClick={() => onSelect({ kind: 'transition', id: t.id, step: t.events[0].step })}>
              <rect x={label.x - width / 2} y={label.y - (second ? 17 : 10)} width={width} height={second ? 34 : 20} rx={6}
                fill={theme.palette.background.paper} fillOpacity={0.92} stroke={theme.palette.divider} />
              <text x={label.x} y={label.y + (second ? -3 : 4)} fontSize={11} textAnchor="middle" fill={theme.palette.text.primary}>{first}</text>
              {second && <text x={label.x} y={label.y + 12} fontSize={11} fontWeight={700} textAnchor="middle" fill={theme.palette.text.secondary}>{second}</text>}
            </g>
          )
        })}
      </svg>
    </Box>
  )
}

function Legend({ machine, showHidden }: { machine: StateMachine; showHidden: boolean }) {
  return (
    <Stack direction="row" gap={1.5} flexWrap="wrap" alignItems="center" sx={{ mt: 1.5 }}>
      <Typography variant="caption" color="text.secondary">
        框＝狀態（流程圖）；箭頭＝修改；線越粗次數越多；雙框＝最後一版；點選狀態或轉移可看細節。
      </Typography>
      {showHidden && machine.hmm && machine.hmm.states.map((state) => (
        <Chip key={state} size="small" variant="outlined"
          sx={{ borderColor: stateColor(state), color: stateColor(state) }}
          label={`${state}：${machine.hmm!.top_observations[state].slice(0, 2).map((o) => tokenLabel(o.observation)).join('、')}`} />
      ))}
      {showHidden && machine.hmm && (
        <Typography variant="caption" color="text.secondary">
          顏色＝產生這一版的修改最可能來自哪個隱藏狀態（模型 {machine.hmm.model_version_id}，Viterbi）。
        </Typography>
      )}
    </Stack>
  )
}

// ------------------------------------------------------------------ 演變路徑
function PathStrip({ machine, showHidden, selection, onSelect }: {
  machine: StateMachine; showHidden: boolean; selection: Selection | null; onSelect: (s: Selection) => void
}) {
  const events = machine.transitions.flatMap((t) => t.events.map((e) => ({ ...e, transition: t.id })))
    .sort((a, b) => a.step - b.step)
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>演變路徑（依版本順序）</Typography>
      <Stack direction="row" gap={0.75} flexWrap="wrap" alignItems="center">
        {machine.path.map((stateId, index) => {
          const event = index > 0 ? events[index - 1] : null
          const color = showHidden && event?.hidden_state ? stateColor(event.hidden_state) : undefined
          const selected = selection?.kind === 'state' && selection.id === stateId && selection.visit === index
          return (
            <Stack key={index} direction="row" gap={0.75} alignItems="center">
              {event && (
                <Chip size="small" variant="outlined" clickable
                  onClick={() => onSelect({ kind: 'transition', id: event.transition, step: event.step })}
                  label={`→ ${operationLabel(event.operation_type)}`}
                  sx={{ borderStyle: 'dashed', ...(color ? { color, borderColor: color } : {}) }} />
              )}
              <Chip size="small" clickable color={selected ? 'primary' : 'default'}
                onClick={() => onSelect({ kind: 'state', id: stateId, visit: index })}
                label={`第 ${index} 版・${stateId}`} />
            </Stack>
          )
        })}
      </Stack>
    </Paper>
  )
}

// ------------------------------------------------------------------ 細節
function Detail({ machine, selection, onSelect, showHidden }: {
  machine: StateMachine; selection: Selection; onSelect: (s: Selection) => void; showHidden: boolean
}) {
  if (selection.kind === 'state') {
    const state = machine.states.find((s) => s.id === selection.id)
    if (!state) return null
    const version = machine.versions[selection.visit] ?? machine.versions[state.visits[0]]
    const items = stateItems(state.summary, machine.granularity)
    return (
      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>
          狀態 {state.id}{state.is_start ? '（開始）' : ''}{state.is_final ? '（最後一版）' : ''}
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {machine.granularity === 'category' ? '用到的節點類別：' : '主要節點（依流程順序）：'}{items.join(' → ')}
        </Typography>
        <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center" sx={{ my: 1.5 }}>
          <Typography variant="body2" color="text.secondary">經過的版本：</Typography>
          {state.visits.map((v) => (
            <Chip key={v} size="small" clickable color={v === version.index ? 'primary' : 'default'}
              label={`第 ${v} 版`} onClick={() => onSelect({ kind: 'state', id: state.id, visit: v })} />
          ))}
          {showHidden && Object.entries(state.hidden_states).map(([h, n]) => (
            <Chip key={h} size="small" variant="outlined" sx={{ borderColor: stateColor(h), color: stateColor(h) }}
              label={`${h}：${n} 次`} />
          ))}
        </Stack>
        {machine.granularity !== 'full' && state.visits.length > 1 && (
          <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
            這幾個版本在「{GRANULARITY_LABEL[machine.granularity]}」下算同一個狀態，參數或節點細節可能不同；下圖是第 {version.index} 版。
          </Typography>
        )}
        <AoVGraph nodes={version.nodes} edges={version.edges} />
      </Paper>
    )
  }

  const transition = machine.transitions.find((t) => t.id === selection.id)
  if (!transition) return null
  const event = transition.events.find((e) => e.step === selection.step) ?? transition.events[0]
  const before = machine.versions[event.from_version]
  const after = machine.versions[event.to_version]
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="subtitle1" fontWeight={700}>
        轉移 {transition.source} → {transition.target}{transition.self_loop ? '（停在原地）' : ''}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        {transition.count} 次修改，佔 {transition.source} 出去的修改 {pct(transition.probability)}。
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>第幾次修改</TableCell>
              <TableCell>修改</TableCell>
              {showHidden && <TableCell>隱藏狀態（後驗）</TableCell>}
              {showHidden && <TableCell>隱藏狀態轉移</TableCell>}
              {machine.hmm && <TableCell align="right">模型當時的預測</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {transition.events.map((e) => (
              <TableRow key={e.step} hover selected={e.step === event.step} sx={{ cursor: 'pointer' }}
                onClick={() => onSelect({ kind: 'transition', id: transition.id, step: e.step })}>
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
      {machine.hmm && (
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1 }}>
          模型當時的預測：只用這次之前的修改，模型給「{event.token ? tokenLabel(event.token) : '這次的修改類型'}」的機率與排名。
          隱藏狀態轉移：第一次修改用初始機率，之後用模型的轉移矩陣。
        </Typography>
      )}
      {event.reason && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" fontWeight={700}>修改理由（第 {event.step} 次）</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', maxHeight: 140, overflow: 'auto' }}>
            {event.reason}
          </Typography>
        </Box>
      )}
      <Typography variant="body2" fontWeight={700} sx={{ mt: 2, mb: 1 }}>
        第 {after.index} 版（綠色＝新增、橘色＝更新，刪除的部分列在圖下方）
      </Typography>
      <AoVGraph nodes={after.nodes} edges={after.edges} change={event.change} beforeNodes={before.nodes} />
    </Paper>
  )
}
