import {
  Alert, Box, Chip, CircularProgress, FormControlLabel, MenuItem, Switch, Paper, Slider, Stack, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, TextField, Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import { operationLabel, tokenLabel } from '../labels'
import type { AssistCandidate, AssistEvaluation, AssistResult } from '../types'

const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)
const RESULT_LABEL: Record<string, string> = { pass: '通過', fail: '未通過', error: '無法執行', unknown: '沒有記錄' }
const RANKING_LABEL: Record<string, string> = {
  assist: 'HMM 評分（本系統）', assist_result: 'HMM 評分＋上一版執行結果（實驗）', bigram: '只看上一次修改', frequency: '只看樣式次數',
}
const shortType = (t: string) => t.replace(/^[a-z]+\./, '')

function describe(c: AssistCandidate): string {
  const d = c.detail as { suggested_parameters?: string[]; replaced_type?: string; template_types?: string[] }
  if (d.suggested_parameters) {
    return d.suggested_parameters.length ? `建議調整：${d.suggested_parameters.join('、')}` : '調整這個節點的參數'
  }
  if (d.replaced_type) return `取代 ${shortType(d.replaced_type)}`
  if (d.template_types) return `換成訓練資料中常見的流程（${d.template_types.length} 個節點）`
  return ''
}

export function AssistPage({ caseId }: { caseId: string }) {
  const [evaluation, setEvaluation] = useState<AssistEvaluation | null>(null)
  const [episodeId, setEpisodeId] = useState('')
  const [step, setStep] = useState(0)
  const [result, setResult] = useState<AssistResult | null>(null)
  const [selected, setSelected] = useState(0)
  const [useResult, setUseResult] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setEvaluation(null)
    setResult(null)
    setEpisodeId('')
    setStep(0)
    api.assistEvaluation(caseId)
      .then((e) => {
        setEvaluation(e)
        setError('')
        setEpisodeId([...e.episodes].sort((a, b) => b.events - a.events)[0]?.episode_id ?? '')
      })
      .catch((err) => setError(errorText(err)))
  }, [caseId])

  useEffect(() => {
    if (!episodeId) return
    let cancelled = false
    api.assist(caseId, episodeId, step, useResult)
      .then((r) => { if (!cancelled) { setResult(r); setSelected(0); setError('') } })
      .catch((err) => { if (!cancelled) setError(errorText(err)) })
    return () => { cancelled = true }
  }, [caseId, episodeId, step, useResult])

  if (error && !evaluation) return <Alert severity="error">{error}</Alert>
  if (!evaluation) return <CircularProgress />

  const episode = evaluation.episodes.find((e) => e.episode_id === episodeId)
  const gain = (c: { assist_rank: number | null; assist_result_rank: number | null }) =>
    (c.assist_rank ?? 99) - (c.assist_result_rank ?? 99)
  const changedSorted = [...evaluation.changed_steps].sort((a, b) => gain(b) - gain(a))
  const jumpTo = (id: string, prefix: number) => { setUseResult(true); setEpisodeId(id); setStep(prefix) }
  const candidate = result?.candidates[selected]
  const actual = result?.actual

  return (
    <Stack gap={3}>
      <Alert severity="info">
        這一頁是<b>離線測試</b>：尚未接上 AoV 平台，所以用模型訓練時保留、沒有參與訓練的 {evaluation.test_episodes} 條序列來檢驗。
        在序列的每一步，系統只看當時的流程圖與之前的修改，提出候選修改並排序，再和實際發生的下一次修改比對。分數是模型機率，不是成功率。
      </Alert>

      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>整體成績（{evaluation.steps} 個測試步驟・模型 {evaluation.model_version_id}）</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          命中率：實際的下一次修改（修改類型＋節點類型）排在候選的前 1、3、5 名的比例。三種排法使用相同的候選，只有排序方式不同。
        </Typography>
        <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mb: 1.5 }}>
          <Chip color="primary" label={`候選涵蓋率 ${pct(evaluation.coverage)}`} />
          <Chip variant="outlined" label={`平均每步 ${evaluation.mean_candidates.toFixed(0)} 個合法候選`} />
          <Chip variant="outlined" label={`樣式庫 ${evaluation.pattern_library.patterns} 種樣式（來自 ${evaluation.train_episodes} 條訓練序列）`} />
          <Chip variant="outlined" label={`未通過 DAG 驗證而剔除：${evaluation.invalid_candidates_dropped}`} />
        </Stack>
        <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' } }}>
          <TableContainer>
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>排序方式</TableCell><TableCell align="right">前 1 名</TableCell><TableCell align="right">前 3 名</TableCell>
                <TableCell align="right">前 5 名</TableCell><TableCell align="right">MRR</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {(['assist', 'assist_result', 'bigram', 'frequency'] as const).map((m) => (
                  <TableRow key={m} sx={{ bgcolor: m === 'assist' ? 'action.hover' : undefined }}>
                    <TableCell>{RANKING_LABEL[m]}</TableCell>
                    <TableCell align="right">{pct(evaluation.methods[m].hit1, 1)}</TableCell>
                    <TableCell align="right">{pct(evaluation.methods[m].hit3, 1)}</TableCell>
                    <TableCell align="right">{pct(evaluation.methods[m].hit5, 1)}</TableCell>
                    <TableCell align="right">{evaluation.methods[m].mrr.toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TableContainer>
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>實際修改類型</TableCell><TableCell align="right">步驟數</TableCell>
                <TableCell align="right">候選涵蓋率</TableCell><TableCell align="right">前 3 名命中</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {Object.entries(evaluation.by_operation).map(([op, v]) => (
                  <TableRow key={op}>
                    <TableCell>{operationLabel(op)}</TableCell>
                    <TableCell align="right">{v.steps}</TableCell>
                    <TableCell align="right">{pct(v.coverage)}</TableCell>
                    <TableCell align="right">{pct(v.assist_hit3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
        <Typography variant="caption" color="text.secondary">
          候選涵蓋率：實際修改有出現在候選清單中的比例，是命中率的上限。涵蓋率低代表候選產生器提不出那種修改。
        </Typography>
      </Paper>

      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700}>實驗功能：納入上一版執行結果</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          上一版流程的執行結果（通過、未通過、無法執行）在提出下一次修改前就已經知道。這個功能預設關閉，用下方「逐步測試」的開關切換。
          測試集裡多數步驟的上一版都是「未通過」，所以整體差異不大；下表與清單可以看出差異出現在哪裡。
        </Typography>
        <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 2fr) minmax(0, 3fr)' } }}>
          <TableContainer>
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>上一版結果</TableCell><TableCell align="right">步驟數</TableCell>
                <TableCell align="right">前 3 名（關）</TableCell><TableCell align="right">前 3 名（開）</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {Object.entries(evaluation.by_previous_result).map(([label, v]) => (
                  <TableRow key={label}>
                    <TableCell>{RESULT_LABEL[label] ?? label}</TableCell>
                    <TableCell align="right">{v.steps}</TableCell>
                    <TableCell align="right">{pct(v.assist_hit3)}</TableCell>
                    <TableCell align="right">{pct(v.assist_result_hit3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              開關前後，實際修改的排名不同的步驟（{evaluation.changed_steps.length} 個，點選可跳到該步驟並打開開關）：
            </Typography>
            <Stack direction="row" gap={0.75} flexWrap="wrap" sx={{ maxHeight: 132, overflow: 'auto' }}>
              {changedSorted.map((c) => (
                <Chip key={`${c.episode_id}-${c.prefix_length}`} size="small" variant="outlined"
                  color={(c.assist_result_rank ?? 99) < (c.assist_rank ?? 99) ? 'success' : 'warning'}
                  label={`${RESULT_LABEL[c.previous_result ?? 'unknown']}｜第 ${c.assist_rank} → ${c.assist_result_rank} 名`}
                  onClick={() => jumpTo(c.episode_id, c.prefix_length)} />
              ))}
            </Stack>
          </Box>
        </Box>
      </Paper>

      <Paper sx={{ p: 2.5 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
          <Typography variant="subtitle1" fontWeight={700}>逐步測試</Typography>
          <FormControlLabel
            control={<Switch id="assist-use-result" checked={useResult} onChange={(e) => setUseResult(e.target.checked)} />}
            label="納入上一版執行結果（實驗功能）"
          />
        </Stack>
        <Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }} sx={{ mt: 1.5 }}>
          <TextField id="assist-episode" select size="small" label="測試序列" value={episodeId}
            onChange={(e) => { setEpisodeId(e.target.value); setStep(0) }} sx={{ minWidth: 420 }}>
            {evaluation.episodes.map((e) => (
              <MenuItem key={e.episode_id} value={e.episode_id}>
                {e.episode_id}（{e.events} 次修改・前 3 名命中 {pct(e.assist_hit3)}）
              </MenuItem>
            ))}
          </TextField>
          {episode && (
            <Stack direction="row" gap={2} alignItems="center" sx={{ flex: 1 }}>
              <Slider aria-label="已知的修改次數" value={step} min={0} max={episode.events} step={1} marks
                onChange={(_, v) => setStep(v as number)} />
              <Typography variant="body2" sx={{ whiteSpace: 'nowrap' }}>已知前 {step}／{episode.events} 次修改</Typography>
            </Stack>
          )}
        </Stack>
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

        {result && (
          <Stack gap={2} sx={{ mt: 2 }}>
            <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
              <Typography variant="body2">上一版的執行結果：</Typography>
              <Chip size="small" label={RESULT_LABEL[result.previous_result ?? 'unknown']}
                color={result.previous_result === 'pass' ? 'success' : result.previous_result === 'error' ? 'error' : 'default'} />
              <Typography variant="body2" color="text.secondary">
                {useResult ? '目前的排序有使用這項資訊。' : '目前的排序沒有使用這項資訊。'}
              </Typography>
            </Stack>
            {actual ? (
              <Alert severity={actual.ranks.assist === null ? 'warning' : actual.ranks.assist <= 3 ? 'success' : 'info'}>
                實際的下一次修改：<b>{operationLabel(actual.operation_type)}・{shortType(actual.activity_type)}</b>。
                {actual.ranks.assist === null
                  ? ' 這個修改不在候選清單中（候選產生器沒有提出）。'
                  : ` 排名：HMM 評分第 ${actual.ranks.assist} 名、加上執行結果後第 ${actual.ranks.assist_result} 名、只看上一次修改第 ${actual.ranks.bigram} 名、只看樣式次數第 ${actual.ranks.frequency} 名。`}
              </Alert>
            ) : <Alert severity="info">這是序列的結尾，沒有實際的下一次修改可以比對。</Alert>}

            <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 5fr) minmax(0, 6fr)' } }}>
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="subtitle2" fontWeight={700} gutterBottom>
                  候選修改（{result.candidate_stats.valid_candidates} 個合法候選中的前 {result.candidates.length} 名）
                </Typography>
                <TableContainer sx={{ border: 1, borderColor: 'divider', borderRadius: 1 }}>
                  <Table size="small">
                    <TableHead><TableRow>
                      <TableCell>#</TableCell><TableCell>建議的修改</TableCell><TableCell align="right">分數</TableCell>
                    </TableRow></TableHead>
                    <TableBody>
                      {result.candidates.map((c, i) => {
                        const isActual = actual && c.operation_type === actual.operation_type && c.activity_type === actual.activity_type
                        return (
                          <TableRow key={`${c.operation_type}-${c.activity_type}`} hover selected={i === selected}
                            onClick={() => setSelected(i)} sx={{ cursor: 'pointer' }}>
                            <TableCell>{c.rank}</TableCell>
                            <TableCell>
                              <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
                                <b>{operationLabel(c.operation_type)}・{shortType(c.activity_type)}</b>
                                {isActual && <Chip size="small" color="success" label="實際發生" />}
                              </Stack>
                              <Typography variant="caption" color="text.secondary">{describe(c)}</Typography>
                            </TableCell>
                            <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>{pct(c.score, 1)}</TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>

              <Stack gap={1.5} sx={{ minWidth: 0 }}>
                <Typography variant="subtitle2" fontWeight={700}>目前的流程（已知前 {result.prefix_length} 次修改後）</Typography>
                <AoVGraph nodes={result.current_workflow.nodes} edges={result.current_workflow.edges} />
                {candidate && (
                  <>
                    <Typography variant="subtitle2" fontWeight={700}>
                      第 {candidate.rank} 名候選套用後：{operationLabel(candidate.operation_type)}・{shortType(candidate.activity_type)}
                    </Typography>
                    <AoVGraph nodes={candidate.proposed_workflow.nodes} edges={candidate.proposed_workflow.edges}
                      change={candidate.change} beforeNodes={result.current_workflow.nodes} />
                    <Typography variant="body2" color="text.secondary">
                      分數 {pct(candidate.score, 1)} ＝ 模型預測「{tokenLabel(candidate.token)}」的機率 {pct(candidate.token_probability, 1)}
                      × 這個樣式在同類可行候選中的比例 {pct(candidate.pattern_prior, 1)}（訓練資料出現 {candidate.training_count} 次）。
                      {describe(candidate) && ` ${describe(candidate)}。`} 已通過 DAG 驗證。
                    </Typography>
                  </>
                )}
              </Stack>
            </Box>
          </Stack>
        )}
      </Paper>
    </Stack>
  )
}
