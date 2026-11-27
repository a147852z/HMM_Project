import {
  Alert, Box, Button, Chip, CircularProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { useEffect, useMemo, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import { stepLayout } from '../components/fsmLayout'
import { GRANULARITY_LABEL, majority, pct, shortType, stateItems, textWidth, truncateTo, wrapTwoLines } from '../components/fsmShared'
import { operationLabel, stateColor, tokenLabel } from '../labels'
import type {
  FsmResult, Granularity, TeacherFsmEpisode, TeacherFsmState, TeacherFsmTransition, TeacherStateMachine,
} from '../types'

type Selection = { kind: 'state'; id: string } | { kind: 'transition'; id: string; index: number }
type ColorBy = 'state' | 'reached'

// 圖的尺寸：由左到右是「平均約第幾版走到」，欄內由上到下
const W = 184
const H = 84
const GX = 108
const GY = 54 // 留給自迴圈與它的標籤
const LEFT = 36
const TOP = 76 // 欄標題＋第一列的自迴圈
const BOTTOM = 72 // 最下面一列往回走的弧線

const RESULT_LABEL: Record<FsmResult, string> = { pass: '通過', fail: '未通過', error: '無法執行', unjudged: '沒有判分' }
const RESULT_ORDER: FsmResult[] = ['pass', 'fail', 'error', 'unjudged']
const MIN_VISITS = [1, 2, 3, 5, 10]
const defaultMinVisits = (g: Granularity) => (g === 'category' ? 1 : 2)
const MAX_CHIPS = 80
const MAX_ROWS = 150

/** 通過率分四級：0%、未滿 20%、20～40%、40% 以上；沒有判分時為 null。 */
function passLevel(rate: number | null): number | null {
  if (rate === null) return null
  if (rate === 0) return 0
  if (rate < 0.2) return 1
  if (rate < 0.4) return 2
  return 3
}
const LEVEL_LABEL = ['0%', '未滿 20%', '20～40%', '40% 以上']

const resultSummary = (results: Partial<Record<FsmResult, number>>) =>
  RESULT_ORDER.filter((r) => results[r]).map((r) => `${RESULT_LABEL[r]} ${results[r]}`).join('、')

/** 題目代號可能很長（11/17 包），畫面上截短，完整代號放在提示。 */
const taskName = (task: string | null, episodeId: string) => truncateTo(task ?? episodeId, 260, 12)

export function TeacherStateMachineView({ caseId, teacher, granularity, showProb, showHidden, onModel, onOpenEpisode }: {
  caseId: string
  teacher: string
  granularity: Granularity
  showProb: boolean
  showHidden: boolean
  onModel: (available: boolean) => void
  onOpenEpisode: (episodeId: string) => void
}) {
  const [machine, setMachine] = useState<TeacherStateMachine | null>(null)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [colorBy, setColorBy] = useState<ColorBy>('state')
  const [minVisits, setMinVisits] = useState(defaultMinVisits(granularity))
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => setMinVisits(defaultMinVisits(granularity)), [granularity])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    api.teacherStateMachine(caseId, teacher, granularity)
      .then((result) => {
        if (cancelled) return
        setMachine(result)
        onModel(!!result.hmm)
        // 預設選起點次數最多的狀態
        const start = [...result.states].sort((a, b) => b.starts - a.starts || a.order - b.order)[0]
        setSelection(start ? { kind: 'state', id: start.id } : null)
      })
      .catch((e) => { if (!cancelled) { setMachine(null); setError(errorText(e)) } })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [caseId, teacher, granularity, onModel])

  const visible = useMemo(() => machine?.states.filter((s) => s.visits >= minVisits) ?? [], [machine, minVisits])

  if (error) return <Alert severity="error">{error}</Alert>
  if (!machine) return loading ? <CircularProgress size={24} /> : null
  const hiddenOn = showHidden && !!machine.hmm
  const { stats } = machine
  const hiddenCount = machine.states.length - visible.length

  return (
    <Stack spacing={2.5}>
      <Paper sx={{ p: 2.5 }}>
        <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mb: 1.5 }} alignItems="center">
          <Chip size="small" color="primary"
            label={`${stats.episodes} 次解題・${stats.tasks} 題・${stats.steps} 次修改 → ${stats.states} 個狀態`} />
          <Chip size="small" variant="outlined" label={`${stats.transitions} 種轉移`} />
          <Chip size="small" variant="outlined" label={`跨題共用的狀態 ${stats.shared_states} 個`} />
          <Chip size="small" variant="outlined" label={stats.judged_episodes
            ? `有判分的解題 ${stats.judged_episodes} 次，最後通過 ${stats.passed_episodes} 次（${pct(stats.passed_episodes / stats.judged_episodes)}）`
            : '沒有判分的解題'} />
          <Chip size="small" variant="outlined" label={`停在原地 ${stats.self_loop_steps} 次`} />
          {loading && <CircularProgress size={16} />}
          {!machine.hmm && <Typography variant="caption" color="text.secondary">這個案例沒有使用中的模型，不顯示隱藏狀態。</Typography>}
        </Stack>
        <Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }} sx={{ mb: 1.5 }}>
          <ToggleButtonGroup size="small" exclusive value={colorBy} aria-label="狀態的顏色"
            onChange={(_, value: ColorBy | null) => value && setColorBy(value)}>
            <ToggleButton value="state" sx={{ px: 1.5 }}>顏色：這個狀態的通過率</ToggleButton>
            <ToggleButton value="reached" sx={{ px: 1.5 }}>顏色：走到這裡的解題最後通過的比例</ToggleButton>
          </ToggleButtonGroup>
          <TextField select size="small" label="最少經過次數" value={minVisits} sx={{ width: 140 }}
            onChange={(e) => setMinVisits(Number(e.target.value))}>
            {MIN_VISITS.map((n) => <MenuItem key={n} value={n}>{n} 次</MenuItem>)}
          </TextField>
          {hiddenCount > 0 && (
            <Typography variant="caption" color="text.secondary">
              隱藏了 {hiddenCount} 個經過不到 {minVisits} 次的狀態（大多是只在一題出現的流程圖）。
            </Typography>
          )}
        </Stack>
        {visible.length > 0
          ? <TeacherDiagram machine={machine} states={visible} colorBy={colorBy} showProb={showProb} showHidden={hiddenOn}
              selection={selection} onSelect={setSelection} />
          : <Alert severity="info">沒有經過 {minVisits} 次以上的狀態，請調低「最少經過次數」。</Alert>}
        <Legend machine={machine} colorBy={colorBy} showHidden={hiddenOn} />
      </Paper>
      {selection && <TeacherDetail machine={machine} selection={selection} onSelect={setSelection} showHidden={hiddenOn}
        onOpenEpisode={onOpenEpisode} shown={new Set(visible.map((s) => s.id))} />}
    </Stack>
  )
}

function usePassColors() {
  const theme = useTheme()
  const levels = [theme.palette.error.main, theme.palette.warning.main, theme.palette.success.light, theme.palette.success.dark]
  return (rate: number | null) => {
    const level = passLevel(rate)
    return level === null ? theme.palette.text.secondary : levels[level]
  }
}

interface Point { x: number; y: number }
/** 一條轉移的線，以及標籤位置：out＝靠近起點那一端，in＝靠近終點那一端，mid＝中間。 */
interface EdgeShape { d: string; out: Point; in: Point; mid: Point }

function cubicAt(p0: Point, c1: Point, c2: Point, p3: Point, t: number): Point {
  const m = 1 - t
  return {
    x: m * m * m * p0.x + 3 * m * m * t * c1.x + 3 * m * t * t * c2.x + t * t * t * p3.x,
    y: m * m * m * p0.y + 3 * m * m * t * c1.y + 3 * m * t * t * c2.y + t * t * t * p3.y,
  }
}

/** 曲線上 x 最接近指定值的點：標籤放在欄與欄之間的空白，不會壓到狀態框。 */
function pointAtX(p0: Point, c1: Point, c2: Point, p3: Point, x: number): Point {
  let best = p0
  for (let i = 0; i <= 48; i += 1) {
    const point = cubicAt(p0, c1, c2, p3, i / 48)
    if (Math.abs(point.x - x) < Math.abs(best.x - x)) best = point
  }
  return best
}

/** 曲線在 t∈[from, to] 之間、y 最接近指定值的點：往回走的線把標籤放在框正下方的空隙。 */
function pointAtY(p0: Point, c1: Point, c2: Point, p3: Point, y: number, from: number, to: number): Point {
  let best = cubicAt(p0, c1, c2, p3, from)
  for (let i = 0; i <= 32; i += 1) {
    const point = cubicAt(p0, c1, c2, p3, from + ((to - from) * i) / 32)
    if (Math.abs(point.y - y) < Math.abs(best.y - y)) best = point
  }
  return best
}

const cubic = (p0: Point, c1: Point, c2: Point, p3: Point) =>
  `M ${p0.x} ${p0.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${p3.x} ${p3.y}`

function TeacherDiagram({ machine, states, colorBy, showProb, showHidden, selection, onSelect }: {
  machine: TeacherStateMachine
  states: TeacherFsmState[]
  colorBy: ColorBy
  showProb: boolean
  showHidden: boolean
  selection: Selection | null
  onSelect: (s: Selection) => void
}) {
  const theme = useTheme()
  const neutral = theme.palette.text.secondary
  const passColor = usePassColors()
  const ids = useMemo(() => new Set(states.map((s) => s.id)), [states])
  const transitions = useMemo(
    () => machine.transitions.filter((t) => ids.has(t.source) && ids.has(t.target)), [machine, ids])
  const layout = useMemo(() => stepLayout(
    states.map((s) => ({ id: s.id, order: s.order, visits: s.visits, firstReach: s.first_reach })), transitions,
  ), [states, transitions])
  const rows = Math.max(1, ...layout.sizes)
  const width = LEFT * 2 + layout.sizes.length * (W + GX) - GX
  const height = TOP + rows * (H + GY) - GY + BOTTOM

  const { shapes, pos } = useMemo(() => {
    const pos = (id: string): Point => {
      const column = layout.column.get(id)!
      const offset = ((rows - layout.sizes[column]) * (H + GY)) / 2
      return { x: LEFT + column * (W + GX), y: TOP + offset + layout.row.get(id)! * (H + GY) }
    }
    const col = (id: string) => layout.column.get(id)!
    // 同一個框的多條線分散接在框邊（依另一端的位置排序），線與標籤比較不會疊在一起
    const ports = (list: TeacherFsmTransition[], own: (t: TeacherFsmTransition) => string,
      other: (t: TeacherFsmTransition) => string, along: (p: Point) => number) => {
      const groups = new Map<string, TeacherFsmTransition[]>()
      for (const t of list) groups.set(own(t), [...(groups.get(own(t)) ?? []), t])
      const share = new Map<string, number>()
      const index = new Map<string, number>()
      for (const group of groups.values()) {
        group.sort((a, b) => along(pos(other(a))) - along(pos(other(b))))
        group.forEach((t, i) => { share.set(t.id, (i + 1) / (group.length + 1)); index.set(t.id, i) })
      }
      return { share, index }
    }
    const forward = transitions.filter((t) => !t.self_loop && col(t.target) > col(t.source))
    const back = transitions.filter((t) => !t.self_loop && col(t.target) < col(t.source))
    const outF = ports(forward, (t) => t.source, (t) => t.target, (p) => p.y)
    const inF = ports(forward, (t) => t.target, (t) => t.source, (p) => p.y)
    const outB = ports(back, (t) => t.source, (t) => t.target, (p) => p.x)
    const inB = ports(back, (t) => t.target, (t) => t.source, (p) => p.x)

    const shapes = new Map<string, EdgeShape>()
    transitions.forEach((t, k) => {
      const a = pos(t.source)
      const b = pos(t.target)
      if (t.self_loop) {
        const x0 = a.x + W - 50
        const x1 = a.x + W - 18
        const label = { x: (x0 + x1) / 2, y: a.y - 36 }
        shapes.set(t.id, {
          d: `M ${x0} ${a.y} C ${x0 - 10} ${a.y - 42}, ${x1 + 10} ${a.y - 42}, ${x1} ${a.y - 3}`, out: label, in: label, mid: label,
        })
      } else if (col(t.target) > col(t.source)) {
        // 往右：從右側出發，接到目標的左側；標籤放在欄與欄之間的空白
        const p0 = { x: a.x + W, y: a.y + H * outF.share.get(t.id)! }
        const p3 = { x: b.x - 7, y: b.y + H * inF.share.get(t.id)! }
        const dx = Math.max(36, (p3.x - p0.x) * 0.45)
        const c1 = { x: p0.x + dx, y: p0.y }
        const c2 = { x: p3.x - dx, y: p3.y }
        shapes.set(t.id, {
          d: cubic(p0, c1, c2, p3), mid: cubicAt(p0, c1, c2, p3, 0.5),
          out: pointAtX(p0, c1, c2, p3, p0.x + GX * (outF.index.get(t.id)! % 2 ? 0.66 : 0.3)),
          in: pointAtX(p0, c1, c2, p3, p3.x - GX * (inF.index.get(t.id)! % 2 ? 0.66 : 0.3)),
        })
      } else if (col(t.target) < col(t.source)) {
        // 往回走：從下緣出發，在兩個框下方的空隙繞回，接到目標的下緣；標籤放在靠近選取狀態的那一端（框下方的空隙）
        const p0 = { x: a.x + W * (0.15 + 0.7 * outB.share.get(t.id)!), y: a.y + H }
        const p3 = { x: b.x + W * (0.15 + 0.7 * inB.share.get(t.id)!), y: b.y + H + 7 }
        const low = Math.max(p0.y, p3.y) + 30 + 8 * (k % 3)
        const cy = (low - (p0.y + p3.y) / 8) / 0.75 // 讓曲線最低處約在 low
        const c1 = { x: p0.x, y: cy }
        const c2 = { x: p3.x, y: cy }
        shapes.set(t.id, {
          d: cubic(p0, c1, c2, p3), mid: cubicAt(p0, c1, c2, p3, 0.5),
          out: pointAtY(p0, c1, c2, p3, p0.y + 18, 0, 0.5), in: pointAtY(p0, c1, c2, p3, p3.y + 11, 0.5, 1),
        })
      } else {
        // 同一欄：從右側出發，在欄右邊的空白鼓起，接回目標的右側
        const p0 = { x: a.x + W, y: a.y + H * 0.68 }
        const p3 = { x: b.x + W + 7, y: b.y + H * 0.32 }
        const bulge = Math.min(GX * 0.45, 30 + Math.abs(p3.y - p0.y) * 0.08)
        const c1 = { x: p0.x + bulge, y: p0.y }
        const c2 = { x: p3.x + bulge, y: p3.y }
        const label = cubicAt(p0, c1, c2, p3, 0.5)
        shapes.set(t.id, { d: cubic(p0, c1, c2, p3), out: label, in: label, mid: label })
      }
    })
    return { shapes, pos }
  }, [transitions, layout, rows])
  const maxCount = Math.max(1, ...transitions.map((t) => t.count))

  const selectedState = selection?.kind === 'state' ? selection.id : null
  const selectedTransition = selection?.kind === 'transition' ? transitions.find((t) => t.id === selection.id) : undefined
  const related = (t: TeacherFsmTransition) =>
    (selectedState ? t.source === selectedState || t.target === selectedState : t.id === selectedTransition?.id)
  const focus = new Set<string>(selectedState
    ? [selectedState, ...transitions.filter(related).flatMap((t) => [t.source, t.target])]
    : selectedTransition ? [selectedTransition.source, selectedTransition.target] : [])
  const rateOf = (s: TeacherFsmState) => (colorBy === 'state' ? s.pass_rate : s.reached_pass_rate)
  const edgeColor = (t: TeacherFsmTransition) => {
    const hidden = showHidden ? majority(t.hidden_states) : null
    return hidden ? stateColor(hidden) : neutral
  }
  const markerColors = Array.from(new Set([neutral, ...(machine.hmm?.states ?? []).map(stateColor)]))
  const markerId = (color: string) => `teacher-arrow-${color.replace(/[^a-z0-9]/gi, '')}`
  const labelText = (t: TeacherFsmTransition) => (showProb ? `${pct(t.probability)}・×${t.count}` : `×${t.count}`)

  return (
    <Box sx={{ overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.default' }}>
      {/* 維持原始大小、橫向捲動：欄數多時寬度會超過畫面，縮小會讓文字無法閱讀 */}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}
        role="img" aria-label={`${machine.teacher} 所有題目合併的有限狀態機`}>
        <defs>
          {markerColors.map((color) => (
            <marker key={color} id={markerId(color)} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10"
              markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
            </marker>
          ))}
        </defs>

        {layout.steps.map((step, c) => (
          <text key={c} x={LEFT + c * (W + GX) + W / 2} y={24} fontSize={12} fontWeight={700} textAnchor="middle" fill={neutral}>
            {step === 0 ? '起點（第 0 版）' : `平均約第 ${step} 版走到`}
          </text>
        ))}

        {transitions.map((t) => {
          const color = edgeColor(t)
          const on = !selection || related(t)
          const d = shapes.get(t.id)!.d
          return (
            <g key={t.id} style={{ cursor: 'pointer' }} opacity={on ? 1 : 0.14}
              onClick={() => onSelect({ kind: 'transition', id: t.id, index: 0 })}>
              <title>{`${t.source} → ${t.target}：${t.count} 次（${pct(t.probability)}），${t.tasks} 題`}</title>
              <path d={d} fill="none" stroke="transparent" strokeWidth={14} />
              <path d={d} fill="none" stroke={color} strokeOpacity={0.85}
                strokeWidth={1.2 + 4.8 * Math.sqrt(t.count / maxCount) + (selectedTransition?.id === t.id ? 1.5 : 0)}
                markerEnd={`url(#${markerId(color)})`} />
            </g>
          )
        })}

        {states.map((s) => {
          const p = pos(s.id)
          const rate = rateOf(s)
          const color = passColor(rate)
          const selected = selectedState === s.id
          const dimmed = focus.size > 0 && !focus.has(s.id)
          const [line1, line2] = wrapTwoLines(stateItems(s.summary, machine.granularity), W - 22, 11)
          const hidden = showHidden ? majority(s.hidden_states) : null
          const marks = [s.starts ? `起點 ${s.starts}` : '', s.finals ? `終點 ${s.finals}` : ''].filter(Boolean).join('・')
          return (
            <g key={s.id} style={{ cursor: 'pointer' }} opacity={dimmed ? 0.35 : 1} onClick={() => onSelect({ kind: 'state', id: s.id })}>
              <title>{`${s.id}：${stateItems(s.summary, machine.granularity).join(' → ')}（經過 ${s.visits} 次、${s.tasks.length} 題）`}</title>
              <rect x={p.x} y={p.y} width={W} height={H} rx={9} fill={theme.palette.background.paper} />
              <rect x={p.x} y={p.y} width={W} height={H} rx={9} fill={color} fillOpacity={rate === null ? 0.05 : 0.16}
                stroke={selected ? theme.palette.primary.main : color} strokeWidth={selected ? 3 : 1.5} />
              {hidden && <rect x={p.x + 1} y={p.y + 10} width={5} height={H - 20} rx={2} fill={stateColor(hidden)} />}
              <text x={p.x + 12} y={p.y + 18} fontSize={13} fontWeight={700} fill={theme.palette.text.primary}>{s.id}</text>
              <text x={p.x + W - 10} y={p.y + 18} fontSize={10.5} textAnchor="end" fill={neutral}>
                {`經過 ${s.visits}・${s.tasks.length} 題`}
              </text>
              <text x={p.x + 12} y={p.y + 36} fontSize={11} fill={theme.palette.text.primary}>{line1}</text>
              <text x={p.x + 12} y={p.y + 51} fontSize={11} fill={theme.palette.text.primary}>{line2}</text>
              <text x={p.x + 12} y={p.y + 72} fontSize={11} fontWeight={700} fill={color}>
                {rate === null ? '沒有判分' : `通過 ${pct(rate)}`}
              </text>
              <text x={p.x + W - 10} y={p.y + 72} fontSize={10} textAnchor="end" fill={neutral}>{marks}</text>
            </g>
          )
        })}

        {transitions.filter((t) => selection && related(t)).map((t) => {
          const shape = shapes.get(t.id)!
          // 選了狀態時，標籤放在靠近這個狀態的那一端；選了轉移時放在中間
          const label = !selectedState ? shape.mid
            : t.target === selectedState && t.source !== selectedState ? shape.in : shape.out
          const text = labelText(t)
          const w = textWidth(text, 10) + 12
          return (
            <g key={`${t.id}-label`} style={{ cursor: 'pointer' }} onClick={() => onSelect({ kind: 'transition', id: t.id, index: 0 })}>
              <rect x={label.x - w / 2} y={label.y - 9} width={w} height={18} rx={5}
                fill={theme.palette.background.paper} fillOpacity={0.94} stroke={theme.palette.divider} />
              <text x={label.x} y={label.y + 4} fontSize={10} fontWeight={700} textAnchor="middle" fill={theme.palette.text.primary}>{text}</text>
            </g>
          )
        })}
      </svg>
    </Box>
  )
}

function Legend({ machine, colorBy, showHidden }: { machine: TeacherStateMachine; colorBy: ColorBy; showHidden: boolean }) {
  const passColor = usePassColors()
  const samples: (number | null)[] = [0, 0.1, 0.3, 0.5, null]
  return (
    <Stack spacing={1} sx={{ mt: 1.5 }}>
      <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center">
        <Typography variant="caption" color="text.secondary">
          {colorBy === 'state' ? '這個狀態的版本通過判分的比例：' : '走到這個狀態的解題，最後通過的比例：'}
        </Typography>
        {samples.map((rate, i) => (
          <Chip key={i} size="small" variant="outlined" sx={{ borderColor: passColor(rate), color: passColor(rate) }}
            label={rate === null ? '沒有判分' : LEVEL_LABEL[passLevel(rate)!]} />
        ))}
        {showHidden && machine.hmm && machine.hmm.states.map((state) => (
          <Chip key={state} size="small" variant="outlined" sx={{ borderColor: stateColor(state), color: stateColor(state) }}
            label={`${state}：${machine.hmm!.top_observations[state].slice(0, 2).map((o) => tokenLabel(o.observation)).join('、')}`} />
        ))}
      </Stack>
      <Typography variant="caption" color="text.secondary">
        由左到右是平均在第幾版第一次走到這個狀態（起點在最左欄）；往回走的轉移畫成下方的弧線、同一欄之間的畫在右側。線越粗代表次數越多；
        點選狀態會標出它的進出轉移與機率。通過率只用有標準答案的解題計算（照片題沒有標準答案）。
        {showHidden && '框左側的色條與線的顏色是產生這一步的 HMM 隱藏狀態（次數最多的）。'}
      </Typography>
    </Stack>
  )
}

function TeacherDetail({ machine, selection, onSelect, showHidden, onOpenEpisode, shown }: {
  machine: TeacherStateMachine
  selection: Selection
  onSelect: (s: Selection) => void
  showHidden: boolean
  onOpenEpisode: (episodeId: string) => void
  /** 圖上顯示的狀態（經過次數達到門檻的） */
  shown: Set<string>
}) {
  const name = (id: string) => (shown.has(id) ? id : `${id}（圖中隱藏）`)
  const episodes = useMemo(() => new Map(machine.episodes.map((e) => [e.episode_id, e])), [machine])
  const episodeChip = (episode: TeacherFsmEpisode) => {
    const name = taskName(episode.work_order_id, episode.episode_id)
    const suffix = episode.final_ok === true ? '・通過' : episode.final_ok === false ? '・未通過' : ''
    return (
      <Chip key={episode.episode_id} size="small" clickable variant={episode.final_ok === null ? 'outlined' : 'filled'}
        color={episode.final_ok === true ? 'success' : 'default'} label={`${name}${suffix}`}
        title={`${episode.work_order_id ?? ''}（${episode.events} 次修改）：點選開啟這次解題`}
        onClick={() => onOpenEpisode(episode.episode_id)} />
    )
  }

  if (selection.kind === 'state') {
    const state = machine.states.find((s) => s.id === selection.id)
    if (!state) return null
    const outgoing = machine.transitions.filter((t) => t.source === state.id).sort((a, b) => b.count - a.count)
    const incoming = machine.transitions.filter((t) => t.target === state.id && !t.self_loop).sort((a, b) => b.count - a.count)
    const example = episodes.get(state.example.episode_id)
    const reached = state.episodes.map((id) => episodes.get(id)).filter((e): e is TeacherFsmEpisode => !!e)
    return (
      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>
          狀態 {state.id}（經過 {state.visits} 次、{state.episodes.length} 次解題、{state.tasks.length} 題）
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {machine.granularity === 'category' ? '用到的節點類別：' : '主要節點（依流程順序）：'}
          {stateItems(state.summary, machine.granularity).join(' → ')}
        </Typography>
        <Stack spacing={0.5} sx={{ mt: 1.5 }}>
          <Typography variant="body2">
            這個狀態的版本：{resultSummary(state.results)}
            {state.pass_rate !== null && `；有判分的 ${state.judged_versions} 版中通過 ${state.passed_versions} 版（${pct(state.pass_rate)}）`}
          </Typography>
          <Typography variant="body2">
            走到這裡的解題：{state.reached_judged
              ? `有判分的 ${state.reached_judged} 次中，最後通過 ${state.reached_passed} 次（${pct(state.reached_pass_rate)}）`
              : '都沒有判分'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            起點 {state.starts} 次、停在這裡（最後一版）{state.finals} 次
            {state.finals > 0 && `：${resultSummary(state.final_results)}`}
          </Typography>
        </Stack>
        {showHidden && Object.keys(state.hidden_states).length > 0 && (
          <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mt: 1 }}>
            {Object.entries(state.hidden_states).map(([h, n]) => (
              <Chip key={h} size="small" variant="outlined" sx={{ borderColor: stateColor(h), color: stateColor(h) }} label={`${h}：${n} 次`} />
            ))}
          </Stack>
        )}

        {outgoing.length > 0 && (
          <>
            <Typography variant="subtitle2" fontWeight={700} sx={{ mt: 2, mb: 0.5 }}>從這裡出去的修改</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>去向</TableCell>
                    <TableCell align="right">次數</TableCell>
                    <TableCell align="right">機率</TableCell>
                    <TableCell>修改</TableCell>
                    <TableCell align="right">修改後通過</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {outgoing.map((t) => (
                    <TableRow key={t.id} hover sx={{ cursor: 'pointer' }} onClick={() => onSelect({ kind: 'transition', id: t.id, index: 0 })}>
                      <TableCell>{t.self_loop ? `${t.target}（停在原地）` : name(t.target)}</TableCell>
                      <TableCell align="right">{t.count}</TableCell>
                      <TableCell align="right">{pct(t.probability)}</TableCell>
                      <TableCell>{Object.entries(t.operations).sort((a, b) => b[1] - a[1]).map(([op, n]) => `${operationLabel(op)} ${n}`).join('、')}</TableCell>
                      <TableCell align="right">{pct(t.pass_rate)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
        {incoming.length > 0 && (
          <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center" sx={{ mt: 1.5 }}>
            <Typography variant="body2" color="text.secondary">從哪裡來：</Typography>
            {incoming.map((t) => (
              <Chip key={t.id} size="small" variant="outlined" clickable label={`${name(t.source)}（${t.count} 次）`}
                onClick={() => onSelect({ kind: 'transition', id: t.id, index: 0 })} />
            ))}
          </Stack>
        )}

        <Typography variant="subtitle2" fontWeight={700} sx={{ mt: 2, mb: 0.5 }}>
          走到這裡的解題（{reached.length} 次，點選可開啟那次解題的狀態機）
        </Typography>
        <Stack direction="row" gap={0.75} flexWrap="wrap">
          {reached.slice(0, MAX_CHIPS).map(episodeChip)}
          {reached.length > MAX_CHIPS && <Typography variant="caption" color="text.secondary">還有 {reached.length - MAX_CHIPS} 次</Typography>}
        </Stack>

        <Typography variant="body2" fontWeight={700} sx={{ mt: 2, mb: 1 }}>
          例子：{example ? taskName(example.work_order_id, example.episode_id) : state.example.episode_id} 的第 {state.example.version} 版
          {machine.granularity !== 'full' && state.visits > 1 && `（在「${GRANULARITY_LABEL[machine.granularity]}」下算同一個狀態，其他版本的細節可能不同）`}
        </Typography>
        <AoVGraph nodes={state.example.nodes} edges={state.example.edges} />
      </Paper>
    )
  }

  const transition = machine.transitions.find((t) => t.id === selection.id)
  if (!transition) return null
  const rows = transition.events.slice(0, MAX_ROWS)
  const event = transition.events[Math.min(selection.index, transition.events.length - 1)]
  const episode = episodes.get(event.episode_id)
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="subtitle1" fontWeight={700}>
        轉移 {transition.source} → {transition.target}{transition.self_loop ? '（停在原地）' : ''}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        {transition.count} 次修改、{transition.episodes.length} 次解題、{transition.tasks} 題，
        佔 {transition.source} 出去的修改 {pct(transition.probability)}。
        修改後的版本：{resultSummary(transition.results)}
        {transition.pass_rate !== null && `（有判分的通過 ${pct(transition.pass_rate)}）`}。
      </Typography>
      <TableContainer sx={{ maxHeight: 420 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>題目</TableCell>
              <TableCell>第幾次修改</TableCell>
              <TableCell>修改</TableCell>
              <TableCell>修改後</TableCell>
              {showHidden && <TableCell>隱藏狀態（後驗）</TableCell>}
              {machine.hmm && <TableCell align="right">模型當時的預測</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((e, index) => (
              <TableRow key={`${e.episode_id}-${e.step}`} hover sx={{ cursor: 'pointer' }} selected={e === event}
                onClick={() => onSelect({ kind: 'transition', id: transition.id, index })}>
                <TableCell title={e.work_order_id ?? e.episode_id}>{taskName(e.work_order_id, e.episode_id)}</TableCell>
                <TableCell>第 {e.step} 次</TableCell>
                <TableCell>{operationLabel(e.operation_type)}{e.activity_type ? `・${shortType(e.activity_type)}` : ''}</TableCell>
                <TableCell>{e.judged ? RESULT_LABEL[e.result ?? 'unjudged'] : '沒有判分'}</TableCell>
                {showHidden && (
                  <TableCell>
                    {e.hidden_state && (
                      <Chip size="small" variant="outlined" sx={{ borderColor: stateColor(e.hidden_state), color: stateColor(e.hidden_state) }}
                        label={`${e.hidden_state}（${pct(e.posterior?.[e.hidden_state])}）`} />
                    )}
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
      {transition.events.length > MAX_ROWS && (
        <Typography variant="caption" color="text.secondary">只列出前 {MAX_ROWS} 筆，共 {transition.events.length} 筆。</Typography>
      )}
      <Stack direction="row" gap={1.5} alignItems="center" sx={{ mt: 2 }}>
        <Typography variant="body2" fontWeight={700}>
          {taskName(event.work_order_id, event.episode_id)}，第 {event.step} 次修改
        </Typography>
        {episode && <Button size="small" variant="outlined" onClick={() => onOpenEpisode(episode.episode_id)}>開啟這次解題</Button>}
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', maxHeight: 160, overflow: 'auto', mt: 0.5 }}>
        {event.reason || '（沒有修改理由）'}
      </Typography>
    </Paper>
  )
}
