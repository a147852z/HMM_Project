import ChevronLeftRounded from '@mui/icons-material/ChevronLeftRounded'
import ChevronRightRounded from '@mui/icons-material/ChevronRightRounded'
import {
  Alert, Autocomplete, Box, Button, Chip, CircularProgress, IconButton, MenuItem, Paper, Slider, Stack, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Tooltip, Typography,
} from '@mui/material'
import { useEffect, useMemo, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import { formatTime, methodLabel, operationLabel, stateColor, tokenLabel } from '../labels'
import type { EpisodeDetail, EpisodeSummary, ModelAnalysis, ModelSummary, ReplayResult } from '../types'

const METHOD_ORDER = ['hmm_bigram_mix', 'hmm_route_bigram', 'hmm', 'bigram', 'unigram']
const pct = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`)

export function ResearchPage({ caseId }: { caseId: string }) {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [modelId, setModelId] = useState('')
  const [analysis, setAnalysis] = useState<ModelAnalysis | null>(null)
  const [groupBy, setGroupBy] = useState('teacher')
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    setAnalysis(null)
    setModelId('')
    Promise.all([api.listModels(caseId), api.listEpisodes(caseId)])
      .then(([m, e]) => {
        setModels(m)
        setEpisodes(e.filter((ep) => ep.status === 'finalized'))
        setModelId(m.find((x) => x.status === 'active')?.model_version_id ?? m.at(-1)?.model_version_id ?? '')
        setError('')
      })
      .catch((err) => setError(errorText(err)))
  }, [caseId])

  useEffect(() => {
    if (!modelId) return
    setAnalysis(null)
    api.modelAnalysis(caseId, modelId, groupBy || undefined).then(setAnalysis).catch((err) => setError(errorText(err)))
  }, [caseId, modelId, groupBy])

  if (error) return <Alert severity="error">{error}</Alert>
  if (models.length === 0) {
    return <Alert severity="info">這個案例還沒有訓練好的模型。請先匯入演變序列並建立訓練（目前可透過 API 或實驗腳本進行）。</Alert>
  }

  return (
    <Stack gap={3}>
      <Paper sx={{ p: 2.5 }}>
        <Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="subtitle1" fontWeight={700}>模型概況</Typography>
            <Typography variant="body2" color="text.secondary">
              指標來自訓練時保留的測試序列（依時間切分，沒有參與訓練）。困惑度越低代表機率估得越準；Top-1 是下一次修改第一名猜中的比例。
            </Typography>
          </Box>
          <TextField id="research-model" select size="small" label="模型版本" value={modelId}
            onChange={(e) => setModelId(e.target.value)} sx={{ minWidth: 280 }}>
            {models.map((m) => (
              <MenuItem key={m.model_version_id} value={m.model_version_id}>
                {m.model_version_id}（K={m.n_states}・{m.feature_schema.fields.join('＋')}）{m.status === 'active' ? '・使用中' : ''}
              </MenuItem>
            ))}
          </TextField>
        </Stack>
        {analysis ? <ModelOverview analysis={analysis} /> : <CircularProgress size={24} sx={{ mt: 2 }} />}
      </Paper>

      {analysis && (
        <Paper sx={{ p: 2.5 }}>
          <Stack direction={{ xs: 'column', md: 'row' }} gap={2} justifyContent="space-between" alignItems={{ md: 'center' }}>
            <Box>
              <Typography variant="subtitle1" fontWeight={700}>隱藏狀態</Typography>
              <Typography variant="body2" color="text.secondary">
                以 Viterbi 解碼全部 {analysis.episodes} 條序列、{analysis.events} 次修改。狀態名稱為中性編號，代表的意義需要人工解讀。
                有 {pct(analysis.episodes_with_state_switch, 1)} 的序列在過程中換過狀態。
              </Typography>
            </Box>
            <TextField id="research-group" size="small" label="分組屬性" value={groupBy}
              onChange={(e) => setGroupBy(e.target.value.trim())} helperText="例如 teacher（CVflow 的 AI 模型）" sx={{ width: 220 }} />
          </Stack>
          <StatesPanel analysis={analysis} />
        </Paper>
      )}

      {analysis && modelId && (
        <Paper sx={{ p: 2.5 }}>
          <Typography variant="subtitle1" fontWeight={700}>序列重播</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            選一條序列，逐步檢視：只看前面的修改時，各方法怎麼預測下一次修改，以及實際發生了什麼。路線判斷只使用當時已知的修改。
          </Typography>
          <EpisodeReplay caseId={caseId} modelId={modelId} episodes={episodes} groupBy={groupBy} states={analysis.states.map((s) => s.state)} />
        </Paper>
      )}
    </Stack>
  )
}

function ModelOverview({ analysis }: { analysis: ModelAnalysis }) {
  const metrics = analysis.test_metrics ?? {}
  const methods = METHOD_ORDER.filter((m) => metrics[m])
  const best = (key: 'perplexity' | 'next_top1' | 'next_top3' | 'next_mrr') => {
    const values = methods.map((m) => metrics[m][key])
    return key === 'perplexity' ? Math.min(...values) : Math.max(...values)
  }
  const bests = { perplexity: best('perplexity'), next_top1: best('next_top1'), next_top3: best('next_top3'), next_mrr: best('next_mrr') }
  const cell = (m: string, key: keyof typeof bests, render: (v: number) => string) => {
    const v = metrics[m][key]
    return (
      <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: v === bests[key] ? 700 : 400,
        color: v === bests[key] ? 'success.main' : undefined }}>{render(v)}</TableCell>
    )
  }
  const testEvents = methods.length ? metrics[methods[0]].events : 0
  return (
    <Stack gap={2} sx={{ mt: 2 }}>
      <Stack direction="row" gap={1} flexWrap="wrap">
        <Chip label={`隱藏狀態 K=${analysis.training.selected_k ?? analysis.states.length}`} color="primary" />
        <Chip variant="outlined" label={`編碼：${analysis.feature_schema.fields.join('＋')}（${analysis.feature_schema.version}）`} />
        <Chip variant="outlined" label={`測試資料 ${testEvents} 次修改`} />
        {analysis.training.candidates.length > 0 && (
          <Chip variant="outlined" label={`BIC：${analysis.training.candidates.map((c) => `K${c.k}=${c.bic.toFixed(0)}`).join('、')}（越低越好）`} />
        )}
      </Stack>
      {methods.length > 0 ? (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>方法</TableCell>
                <TableCell align="right">困惑度</TableCell>
                <TableCell align="right">Top-1</TableCell>
                <TableCell align="right">Top-3</TableCell>
                <TableCell align="right">MRR</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {methods.map((m) => (
                <TableRow key={m}>
                  <TableCell>{methodLabel(m)}{m === 'hmm_bigram_mix' && <Chip size="small" label="推論預設" sx={{ ml: 1 }} />}</TableCell>
                  {cell(m, 'perplexity', (v) => v.toFixed(2))}
                  {cell(m, 'next_top1', (v) => pct(v, 1))}
                  {cell(m, 'next_top3', (v) => pct(v, 1))}
                  {cell(m, 'next_mrr', (v) => v.toFixed(3))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : <Typography variant="body2" color="text.secondary">這個模型沒有保存測試指標。</Typography>}
    </Stack>
  )
}

function ShareBar({ shares, states }: { shares: Record<string, number>; states: string[] }) {
  return (
    <Box sx={{ display: 'flex', height: 18, borderRadius: 1, overflow: 'hidden', bgcolor: 'grey.200', flex: 1, minWidth: 120 }}>
      {states.map((s) => (shares[s] ?? 0) > 0 && (
        <Tooltip key={s} title={`${s}：${pct(shares[s], 1)}`}>
          <Box sx={{ width: `${(shares[s] ?? 0) * 100}%`, bgcolor: stateColor(s), color: '#fff', fontSize: 11,
            display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', whiteSpace: 'nowrap' }}>
            {(shares[s] ?? 0) >= 0.12 ? pct(shares[s]) : ''}
          </Box>
        </Tooltip>
      ))}
    </Box>
  )
}

function StatesPanel({ analysis }: { analysis: ModelAnalysis }) {
  const states = analysis.states.map((s) => s.state)
  const outcomeLabel: Record<string, string> = { final_ok: '最後成功', final_fail: '最後失敗' }
  return (
    <Stack gap={3} sx={{ mt: 2 }}>
      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: `repeat(${Math.min(states.length, 3)}, minmax(0, 1fr))` } }}>
        {analysis.states.map((s) => {
          const ops = Object.entries(s.operation_share).sort((a, b) => b[1] - a[1])
          return (
            <Box key={s.state} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 2, borderTop: 4, borderTopColor: stateColor(s.state) }}>
              <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                <Typography fontWeight={700} sx={{ color: stateColor(s.state) }}>{s.state}</Typography>
                <Typography variant="body2" color="text.secondary">{s.events} 次修改・停留 {pct(s.self_transition, 1)}</Typography>
              </Stack>
              <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1 }}>最可能產生的觀測</Typography>
              <Stack gap={0.5} sx={{ mt: 0.5 }}>
                {s.top_observations.slice(0, 5).map((o) => (
                  <Stack key={o.observation} direction="row" gap={1} alignItems="center">
                    <Typography variant="body2" sx={{ width: 130, flexShrink: 0 }} noWrap title={o.observation}>{tokenLabel(o.observation)}</Typography>
                    <Box sx={{ flex: 1, height: 8, bgcolor: 'grey.100', borderRadius: 1 }}>
                      <Box sx={{ width: `${Math.min(100, o.probability * 100 / (s.top_observations[0]?.probability || 1))}%`, height: '100%', bgcolor: stateColor(s.state), borderRadius: 1, opacity: 0.8 }} />
                    </Box>
                    <Typography variant="caption" sx={{ width: 40, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{pct(o.probability, 1)}</Typography>
                  </Stack>
                ))}
              </Stack>
              <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1.5 }}>實際修改組成</Typography>
              <Typography variant="body2">{ops.map(([op, v]) => `${operationLabel(op)} ${pct(v)}`).join('・')}</Typography>
            </Box>
          )
        })}
      </Box>

      {(Object.keys(analysis.state_share_by_group).length > 0 || Object.keys(analysis.state_share_by_outcome).length > 0) && (
        <Box>
          <Stack direction="row" gap={2} alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" fontWeight={700}>各組的狀態比例（以修改次數計）</Typography>
            <Stack direction="row" gap={1}>
              {states.map((s) => <Chip key={s} size="small" label={s} sx={{ bgcolor: stateColor(s), color: '#fff' }} />)}
            </Stack>
          </Stack>
          <Stack gap={1}>
            {Object.entries(analysis.state_share_by_group).map(([group, shares]) => (
              <Stack key={group} direction="row" gap={1.5} alignItems="center">
                <Typography variant="body2" sx={{ width: 110, flexShrink: 0 }} noWrap>{group}</Typography>
                <ShareBar shares={shares} states={states} />
              </Stack>
            ))}
            {Object.entries(analysis.state_share_by_outcome).map(([key, shares]) => (
              <Stack key={key} direction="row" gap={1.5} alignItems="center">
                <Typography variant="body2" fontWeight={700} sx={{ width: 110, flexShrink: 0 }}>{outcomeLabel[key] ?? key}</Typography>
                <ShareBar shares={shares} states={states} />
              </Stack>
            ))}
          </Stack>
        </Box>
      )}
    </Stack>
  )
}

function episodeGroup(ep: EpisodeSummary, groupBy: string): string {
  const value = (ep.provenance as Record<string, unknown>)?.[groupBy]
  return value === undefined || value === null ? '' : String(value)
}
const episodeOutcome = (ep: EpisodeSummary) => ep.provenance?.outcome?.final_ok

function EpisodeReplay({ caseId, modelId, episodes, groupBy, states }: {
  caseId: string; modelId: string; episodes: EpisodeSummary[]; groupBy: string; states: string[]
}) {
  const [group, setGroup] = useState('')
  const [outcome, setOutcome] = useState<'all' | 'ok' | 'fail'>('all')
  const [selected, setSelected] = useState<EpisodeSummary | null>(null)
  const [replay, setReplay] = useState<ReplayResult | null>(null)
  const [detail, setDetail] = useState<EpisodeDetail | null>(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')

  const groups = useMemo(() => [...new Set(episodes.map((e) => episodeGroup(e, groupBy)).filter(Boolean))].sort(), [episodes, groupBy])
  const filtered = episodes.filter((e) => (!group || episodeGroup(e, groupBy) === group)
    && (outcome === 'all' || (outcome === 'ok' ? episodeOutcome(e) === true : episodeOutcome(e) === false)))

  useEffect(() => {
    if (!selected && episodes.length) setSelected([...episodes].sort((a, b) => b.event_count - a.event_count)[0])
  }, [episodes, selected])

  useEffect(() => {
    if (!selected) return
    setReplay(null)
    setDetail(null)
    setStep(0)
    Promise.all([api.replay(caseId, modelId, selected.episode_id), api.getEpisode(caseId, selected.episode_id)])
      .then(([r, d]) => { setReplay(r); setDetail(d); setError('') })
      .catch((err) => setError(errorText(err)))
  }, [caseId, modelId, selected])

  return (
    <Stack gap={2}>
      <Stack direction={{ xs: 'column', md: 'row' }} gap={2}>
        {groups.length > 0 && (
          <TextField id="replay-group" select size="small" label={`篩選 ${groupBy}`} value={group}
            onChange={(e) => setGroup(e.target.value)} sx={{ minWidth: 160 }}>
            <MenuItem value="">全部</MenuItem>
            {groups.map((g) => <MenuItem key={g} value={g}>{g}</MenuItem>)}
          </TextField>
        )}
        <TextField id="replay-outcome" select size="small" label="最後成敗" value={outcome}
          onChange={(e) => setOutcome(e.target.value as typeof outcome)} sx={{ minWidth: 140 }}>
          <MenuItem value="all">全部</MenuItem>
          <MenuItem value="ok">最後成功</MenuItem>
          <MenuItem value="fail">最後失敗</MenuItem>
        </TextField>
        <Autocomplete
          id="replay-episode" size="small" sx={{ flex: 1, minWidth: 280 }}
          options={filtered} value={selected && filtered.includes(selected) ? selected : null}
          onChange={(_, value) => value && setSelected(value)}
          getOptionLabel={(e) => `${e.work_order_id ?? e.episode_id}（${episodeGroup(e, groupBy) || '—'}・${e.event_count} 次修改${episodeOutcome(e) === true ? '・成功' : episodeOutcome(e) === false ? '・失敗' : ''}）`}
          isOptionEqualToValue={(a, b) => a.episode_id === b.episode_id}
          renderInput={(params) => <TextField {...params} label={`序列（${filtered.length} 條）`} />}
        />
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}
      {selected && (!replay || !detail) && !error && <CircularProgress size={24} />}
      {replay && detail && replay.steps.length > 0 && (
        <ReplayView replay={replay} detail={detail} step={Math.min(step, replay.steps.length - 1)} onStep={setStep} states={states} />
      )}
    </Stack>
  )
}

function ReplayView({ replay, detail, step, onStep, states }: {
  replay: ReplayResult; detail: EpisodeDetail; step: number; onStep: (n: number) => void; states: string[]
}) {
  const current = replay.steps[step]
  const event = detail.events[step]
  const version = detail.snapshots[step + 1] ?? detail.snapshots.at(-1)!
  const before = detail.snapshots[step]
  const methods = METHOD_ORDER.filter((m) => replay.methods.includes(m))
  const def = replay.default_method
  const n = replay.steps.length
  const result = (event?.attributes?.result ?? {}) as { passed?: boolean | null; executable?: boolean | null }
  const argmax = (probs: Record<string, number>) => Object.entries(probs).sort((a, b) => b[1] - a[1])[0]?.[0] ?? ''

  return (
    <Stack gap={2}>
      {/* 時間軸：每一步的顏色是看到該修改後最可能的狀態；勾號代表預設方法第一名猜中 */}
      <Box sx={{ display: 'flex', gap: 0.5, overflowX: 'auto', pb: 0.5 }}>
        {replay.steps.map((s, i) => {
          const hit = s.predictions[def]?.actual_rank === 1
          return (
            <Tooltip key={s.event_id} title={`${i + 1}. ${tokenLabel(s.token)}｜${methodLabel(def)}：第 ${s.predictions[def]?.actual_rank} 名`}>
              <Box onClick={() => onStep(i)} sx={{
                cursor: 'pointer', minWidth: 64, px: 0.75, py: 0.5, borderRadius: 1, color: '#fff', fontSize: 12, textAlign: 'center',
                bgcolor: stateColor(argmax(s.belief_after)), outline: i === step ? '3px solid' : 'none', outlineColor: 'text.primary', outlineOffset: 1,
              }}>
                <div>{i + 1}. {operationLabel(s.operation_type)}</div>
                <div>{hit ? '✓' : '·'}</div>
              </Box>
            </Tooltip>
          )
        })}
      </Box>
      <Stack direction="row" gap={1} alignItems="center">
        <IconButton aria-label="上一步" onClick={() => onStep(Math.max(0, step - 1))} disabled={step === 0}><ChevronLeftRounded /></IconButton>
        <Slider aria-label="步驟" value={step + 1} min={1} max={n} step={1} marks onChange={(_, v) => onStep((v as number) - 1)} sx={{ flex: 1, mx: 1 }} />
        <IconButton aria-label="下一步" onClick={() => onStep(Math.min(n - 1, step + 1))} disabled={step === n - 1}><ChevronRightRounded /></IconButton>
        <Typography variant="body2" sx={{ minWidth: 90, fontVariantNumeric: 'tabular-nums' }}>第 {step + 1}／{n} 次修改</Typography>
      </Stack>

      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 3fr) minmax(0, 2fr)' } }}>
        <Stack gap={1.5} sx={{ minWidth: 0 }}>
          <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
            <Typography fontWeight={700}>實際修改：{tokenLabel(current.token)}</Typography>
            {result.executable === false && <Chip size="small" color="error" label="這一版無法執行" />}
            {result.passed === true && <Chip size="small" color="success" label="這一版通過" />}
            {result.passed === false && result.executable !== false && <Chip size="small" variant="outlined" label="這一版未通過" />}
          </Stack>
          <AoVGraph nodes={version.nodes} edges={version.edges} change={event?.change} beforeNodes={before?.nodes} />
          {event?.reason && (
            <Box sx={{ typography: 'body2', bgcolor: 'grey.50', border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}>
              <Typography variant="caption" color="text.secondary" component="div">
                修改理由（{event.reason_source === 'actor' ? '修改者自述' : event.reason_source === 'human' ? '原文記載' : 'LLM 推論'}）
              </Typography>
              {event.reason}
            </Box>
          )}
        </Stack>

        <Stack gap={2} sx={{ minWidth: 0 }}>
          <Box>
            <Typography variant="subtitle2" fontWeight={700}>此時判斷的路線（只看前 {step} 次修改）</Typography>
            <Stack gap={0.75} sx={{ mt: 1 }}>
              {states.map((s) => (
                <Stack key={s} direction="row" gap={1} alignItems="center">
                  <Typography variant="body2" sx={{ width: 32, color: stateColor(s), fontWeight: 700 }}>{s}</Typography>
                  <Box sx={{ flex: 1, height: 10, bgcolor: 'grey.100', borderRadius: 1 }}>
                    <Box sx={{ width: `${(current.route_before[s] ?? 0) * 100}%`, height: '100%', bgcolor: stateColor(s), borderRadius: 1 }} />
                  </Box>
                  <Typography variant="caption" sx={{ width: 44, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{pct(current.route_before[s], 1)}</Typography>
                </Stack>
              ))}
            </Stack>
          </Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700}>預測第 {step + 1} 次修改</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>方法</TableCell>
                    <TableCell>第一名</TableCell>
                    <TableCell align="right">實際排名</TableCell>
                    <TableCell align="right">實際機率</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {methods.map((m) => {
                    const p = current.predictions[m]
                    return (
                      <TableRow key={m} sx={{ bgcolor: m === def ? 'action.hover' : undefined }}>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{methodLabel(m)}</TableCell>
                        <TableCell>
                          <Box component="span" sx={{ color: p.top[0]?.observation === current.encoded_as ? 'success.main' : undefined, fontWeight: p.top[0]?.observation === current.encoded_as ? 700 : 400 }}>
                            {tokenLabel(p.top[0]?.observation ?? '')}
                          </Box>
                          <Box component="span" sx={{ color: 'text.secondary', ml: 0.5 }}>{pct(p.top[0]?.probability)}</Box>
                        </TableCell>
                        <TableCell align="right">
                          <Chip size="small" label={`第 ${p.actual_rank} 名`} color={p.actual_rank === 1 ? 'success' : p.actual_rank <= 3 ? 'warning' : 'default'} variant={p.actual_rank === 1 ? 'filled' : 'outlined'} />
                        </TableCell>
                        <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>{pct(p.actual_probability, 1)}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </TableContainer>
            <Typography variant="caption" color="text.secondary">
              {methodLabel(def)} 的前五名：{current.predictions[def]?.top.map((t) => `${tokenLabel(t.observation)} ${pct(t.probability)}`).join('、')}
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700}>這條序列的整體表現</Typography>
            <Stack gap={0.5} sx={{ mt: 0.5 }}>
              {methods.map((m) => (
                <Typography key={m} variant="body2">
                  {methodLabel(m)}：Top-1 {pct(replay.summary[m].top1)}・Top-3 {pct(replay.summary[m].top3)}
                </Typography>
              ))}
            </Stack>
          </Box>
          <Typography variant="caption" color="text.secondary">
            序列：{detail.work_order_id ?? detail.episode_id}・開始於 {formatTime(detail.started_at)}・模型 {replay.model_version_id}
          </Typography>
          <Button size="small" sx={{ alignSelf: 'flex-start' }} onClick={() => onStep(0)} disabled={step === 0}>回到第 1 步</Button>
        </Stack>
      </Box>
    </Stack>
  )
}
