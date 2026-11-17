import ArrowBackRounded from '@mui/icons-material/ArrowBackRounded'
import CheckRounded from '@mui/icons-material/CheckRounded'
import CloseRounded from '@mui/icons-material/CloseRounded'
import ErrorOutlineRounded from '@mui/icons-material/ErrorOutlineRounded'
import WarningAmberRounded from '@mui/icons-material/WarningAmberRounded'
import {
  Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, List,
  ListItemButton, ListItemText, Paper, Stack, Tab, Tabs, TextField, Tooltip, Typography,
} from '@mui/material'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, errorText } from '../api'
import { AoVGraph } from '../components/AoVGraph'
import { SourceDocument, type Highlight } from '../components/SourceDocument'
import { translateIssue } from '../issues'
import { STATUS_COLOR, STATUS_LABEL, formatTime, isActive, operationLabel } from '../labels'
import type { AoVNode, Change, ConversionDetail, DocumentInfo, PreviewEpisode, PreviewEvent } from '../types'

type Panel = 'steps' | 'checks' | 'draft' | 'raw'

const REVIEWER_KEY = 'kcore.reviewer'
function readReviewer() { try { return localStorage.getItem(REVIEWER_KEY) ?? '' } catch { return '' } }
function saveReviewer(name: string) { try { localStorage.setItem(REVIEWER_KEY, name) } catch { /* 忽略 */ } }

interface Issues { general: { errors: string[]; warnings: string[] }; byEvent: Map<number, { errors: string[]; warnings: string[] }> }

/** 驗證訊息以「event N:」開頭的歸到對應步驟，其餘歸為整體問題。 */
function splitIssues(errors: string[], warnings: string[]): Issues {
  const issues: Issues = { general: { errors: [], warnings: [] }, byEvent: new Map() }
  const put = (message: string, kind: 'errors' | 'warnings') => {
    const match = /^event (\d+): (.*)$/s.exec(message)
    if (!match) { issues.general[kind].push(message); return }
    const index = Number(match[1])
    const bucket = issues.byEvent.get(index) ?? { errors: [], warnings: [] }
    bucket[kind].push(match[2])
    issues.byEvent.set(index, bucket)
  }
  errors.forEach((m) => put(m, 'errors'))
  warnings.forEach((m) => put(m, 'warnings'))
  return issues
}

export function ReviewView({ caseId, conversionId, onBack }: { caseId: string; conversionId: string; onBack: () => void }) {
  const [detail, setDetail] = useState<ConversionDetail | null>(null)
  const [document, setDocument] = useState<DocumentInfo | null>(null)
  const [panel, setPanel] = useState<Panel>('steps')
  const [episodeIndex, setEpisodeIndex] = useState(0)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setDetail(await api.getConversion(caseId, conversionId))
      setError('')
    } catch (err) {
      setError(errorText(err))
    }
  }, [caseId, conversionId])

  useEffect(() => { void load() }, [load])
  const documentId = detail?.document_id
  useEffect(() => {
    if (!documentId) return
    api.getDocument(caseId, documentId).then(setDocument).catch((err) => setError(errorText(err)))
  }, [caseId, documentId])
  const active = detail ? isActive(detail.status) : false
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => { void load() }, 3000)
    return () => window.clearInterval(timer)
  }, [active, load])
  useEffect(() => {
    if (detail?.status === 'failed') setPanel((p) => (p === 'steps' ? 'raw' : p))
  }, [detail?.status])

  // 第一次載入草稿時，直接跳到第一個有錯誤（其次是提醒）的步驟
  const [jumped, setJumped] = useState(false)
  useEffect(() => {
    if (jumped || !detail || detail.preview.length === 0) return
    setJumped(true)
    const reports = detail.validation.episodes ?? []
    for (const kind of ['errors', 'warnings'] as const) {
      for (let i = 0; i < reports.length; i += 1) {
        const hit = reports[i][kind].map((m) => /^event (\d+):/.exec(m)).find(Boolean)
        if (hit) { setEpisodeIndex(i); setStep(Number(hit[1])); return }
      }
    }
  }, [detail, jumped])

  const episode: PreviewEpisode | undefined = detail?.preview[episodeIndex]
  const report = detail?.validation.episodes?.[episodeIndex]
  const issues = useMemo(() => splitIssues(report?.errors ?? [], report?.warnings ?? []), [report])
  const event: PreviewEvent | undefined = step > 0 ? episode?.events[step - 1] : undefined

  const highlights: Highlight[] = useMemo(() => {
    const list: Highlight[] = []
    if (event?.attributes.source_excerpt) list.push({ text: event.attributes.source_excerpt, kind: 'excerpt' })
    if (event?.attributes.reason_quote) list.push({ text: event.attributes.reason_quote, kind: 'quote' })
    return list
  }, [event])

  if (!detail) {
    return (
      <Stack gap={2}>
        <Button startIcon={<ArrowBackRounded />} onClick={onBack} sx={{ alignSelf: 'flex-start' }}>回到轉換紀錄</Button>
        {error ? <Alert severity="error">{error}</Alert> : <CircularProgress />}
      </Stack>
    )
  }

  const totalErrors = (detail.validation.schema_errors?.length ?? 0)
    + (detail.validation.episodes ?? []).reduce((n, e) => n + e.errors.length, 0)
    + (detail.validation.already_ingested?.length ? 1 : 0)
  const totalWarnings = (detail.validation.episodes ?? []).reduce((n, e) => n + e.warnings.length, 0)

  return (
    <Stack gap={2}>
      <Stack direction="row" alignItems="center" gap={1.5} flexWrap="wrap">
        <Button startIcon={<ArrowBackRounded />} onClick={onBack}>回到轉換紀錄</Button>
        <Typography variant="h6" fontWeight={700} sx={{ fontFamily: 'monospace' }}>{detail.conversion_id}</Typography>
        <Chip label={STATUS_LABEL[detail.status]} color={STATUS_COLOR[detail.status]} />
        {detail.draft_edited && <Chip variant="outlined" label="草稿已人工修正" />}
        <Typography variant="body2" color="text.secondary">
          {document?.filename}・設定 {detail.profile_id} 第 {detail.profile_version} 版・{formatTime(detail.created_at)}
        </Typography>
      </Stack>

      {error && <Alert severity="error">{error}</Alert>}
      {active && <Alert severity="info" icon={<CircularProgress size={18} />}>本地 LLM 正在整理這份文件，完成後會自動更新。</Alert>}
      {detail.status === 'failed' && <Alert severity="error">轉換失敗：{detail.error}。可以查看 LLM 原始回應，或直接在「草稿 JSON」貼上修正後的內容。</Alert>}
      {detail.validation.already_ingested?.length ? (
        <Alert severity="error">這份文件已經匯入過（{detail.validation.already_ingested.join('、')}），不能重複核准，以免重複計入訓練資料。</Alert>
      ) : null}
      {detail.status === 'approved' && (
        <Alert severity="success">已由 {detail.reviewer} 於 {formatTime(detail.reviewed_at)} 核准，匯入 {detail.ingested_episode_ids.join('、')}。</Alert>
      )}
      {detail.status === 'rejected' && (
        <Alert severity="warning">已由 {detail.reviewer} 於 {formatTime(detail.reviewed_at)} 退回。{detail.review_note && `備註：${detail.review_note}`}</Alert>
      )}

      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 5fr) minmax(0, 7fr)' }, alignItems: 'start' }}>
        <Paper sx={{ p: 2, position: { lg: 'sticky' }, top: { lg: 80 }, maxHeight: { lg: 'calc(100vh - 200px)' }, overflow: 'auto', minWidth: 0 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" fontWeight={700}>原始文件</Typography>
            <Stack direction="row" gap={1}>
              <Chip size="small" sx={{ bgcolor: 'warning.light' }} label="修改依據" />
              <Chip size="small" sx={{ bgcolor: 'success.light' }} label="原因引文" />
            </Stack>
          </Stack>
          {document?.content ? <SourceDocument content={document.content} highlights={highlights} /> : <CircularProgress size={20} />}
        </Paper>

        <Paper sx={{ minWidth: 0 }}>
          <Tabs value={panel} onChange={(_, value: Panel) => setPanel(value)} variant="scrollable" sx={{ borderBottom: 1, borderColor: 'divider', px: 1 }}>
            <Tab value="steps" label="修改步驟" />
            <Tab value="checks" label={`檢查結果（${totalErrors} 錯誤・${totalWarnings} 提醒）`} />
            <Tab value="draft" label="草稿 JSON" />
            <Tab value="raw" label="LLM 原始回應" />
          </Tabs>
          <Box sx={{ p: 2 }}>
            {panel === 'steps' && (
              <StepsPanel
                detail={detail} episodeIndex={episodeIndex} onEpisode={(i) => { setEpisodeIndex(i); setStep(0) }}
                step={step} onStep={setStep} issues={issues}
              />
            )}
            {panel === 'checks' && <ChecksPanel detail={detail} />}
            {panel === 'draft' && <DraftPanel caseId={caseId} detail={detail} onSaved={(next) => { setDetail(next); setStep(0) }} />}
            {panel === 'raw' && (
              <Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 13 }}>
                {detail.raw_response ?? '（尚無回應）'}
              </Box>
            )}
          </Box>
        </Paper>
      </Box>

      <ReviewBar caseId={caseId} detail={detail} onDone={load} />
    </Stack>
  )
}

function StepsPanel({ detail, episodeIndex, onEpisode, step, onStep, issues }: {
  detail: ConversionDetail
  episodeIndex: number
  onEpisode: (index: number) => void
  step: number
  onStep: (step: number) => void
  issues: Issues
}) {
  if (detail.preview.length === 0) {
    return <Typography color="text.secondary">還沒有可以預覽的草稿。{detail.status === 'failed' && '請查看 LLM 原始回應。'}</Typography>
  }
  const episode = detail.preview[episodeIndex]
  const version = episode.versions.find((v) => v.event_index === step)
  const before = episode.versions.find((v) => v.event_index === step - 1)
  const event = step > 0 ? episode.events[step - 1] : undefined
  const eventIssues = event ? issues.byEvent.get(event.sequence_index) : undefined

  return (
    <Stack gap={2}>
      {detail.preview.length > 1 && (
        <Tabs value={episodeIndex} onChange={(_, value: number) => onEpisode(value)} variant="scrollable">
          {detail.preview.map((ep, i) => <Tab key={ep.episode_id} value={i} label={`序列 ${i + 1}（${ep.events.length} 步）`} />)}
        </Tabs>
      )}
      {(issues.general.errors.length > 0 || issues.general.warnings.length > 0) && (
        <IssueList errors={issues.general.errors} warnings={issues.general.warnings} />
      )}
      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '220px minmax(0, 1fr)' } }}>
        <List dense disablePadding sx={{ border: 1, borderColor: 'divider', borderRadius: 1, alignSelf: 'start' }}>
          <ListItemButton selected={step === 0} onClick={() => onStep(0)}>
            <ListItemText primary="初始流程" secondary={`${episode.versions[0]?.nodes.length ?? 0} 個活動`} />
          </ListItemButton>
          {episode.events.map((ev) => {
            const issue = issues.byEvent.get(ev.sequence_index)
            const blocked = episode.stopped_at !== null && ev.sequence_index >= episode.stopped_at
            return (
              <ListItemButton key={ev.event_id} selected={step === ev.sequence_index} onClick={() => onStep(ev.sequence_index)}>
                <ListItemText
                  primary={<Stack direction="row" alignItems="center" gap={0.5}>
                    <span>{ev.sequence_index}. {operationLabel(ev.operation_type)}</span>
                    {issue?.errors.length ? <ErrorOutlineRounded color="error" fontSize="small" /> : null}
                    {!issue?.errors.length && issue?.warnings.length ? <WarningAmberRounded color="warning" fontSize="small" /> : null}
                  </Stack>}
                  secondary={blocked ? '無法預覽' : reasonSourceLabel(ev)}
                />
              </ListItemButton>
            )
          })}
        </List>

        <Stack gap={2} sx={{ minWidth: 0 }}>
          {eventIssues && <IssueList errors={eventIssues.errors} warnings={eventIssues.warnings} />}
          {version ? (
            <AoVGraph nodes={version.nodes} edges={version.edges} change={event?.change} beforeNodes={before?.nodes} />
          ) : (
            <Alert severity="error">這一步的修改無法套用到前一個版本，所以沒有流程圖。請查看上方錯誤並修正草稿。</Alert>
          )}
          {event && <EventDetail event={event} nodes={[...(before?.nodes ?? []), ...(version?.nodes ?? [])]} />}
          {!event && <Typography variant="body2" color="text.secondary">
            點選左側的步驟查看每次修改。綠色是這次新增的活動或依賴，橘色是這次更新的活動。
          </Typography>}
        </Stack>
      </Box>
    </Stack>
  )
}

function reasonSourceLabel(ev: PreviewEvent) {
  if (!ev.reason) return '未記錄原因'
  return ev.reason_source === 'human' ? '原因：原文記載' : '原因：LLM 推論'
}

function describeChange(change: Change, nodes: AoVNode[]): string[] {
  const name = (id: string) => {
    const node = nodes.find((n) => n.node_id === id)
    return node ? `「${node.name || node.activity_type}」` : id
  }
  const lines: string[] = []
  change.added_nodes.forEach((n) => lines.push(`新增活動「${n.name || n.activity_type}」（${n.activity_type}）`))
  change.removed_nodes.forEach((id) => lines.push(`刪除活動${name(id)}`))
  change.added_edges.forEach((e) => lines.push(`新增依賴 ${name(e.source)} → ${name(e.target)}`))
  change.removed_edges.forEach((e) => lines.push(`移除依賴 ${name(e.source)} → ${name(e.target)}`))
  change.updated_node_attributes.forEach((u) => {
    const parts: string[] = []
    if (u.name) parts.push(`名稱改為「${u.name}」`)
    if (u.activity_type) parts.push(`類型改為 ${u.activity_type}`)
    if (u.parameters_set && Object.keys(u.parameters_set).length) parts.push(`參數 ${JSON.stringify(u.parameters_set)}`)
    if (u.parameters_unset?.length) parts.push(`移除參數 ${u.parameters_unset.join('、')}`)
    lines.push(`更新活動${name(u.node_id)}：${parts.join('；')}`)
  })
  return lines
}

function EventDetail({ event, nodes }: { event: PreviewEvent; nodes: AoVNode[] }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '96px minmax(0, 1fr)' }, rowGap: 1, columnGap: 2, typography: 'body2' }}>
      <Box sx={{ color: 'text.secondary' }}>修改類型</Box>
      <Box>{operationLabel(event.operation_type)} <Box component="span" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>{event.operation_type}</Box></Box>
      <Box sx={{ color: 'text.secondary' }}>修改內容</Box>
      <Box>{describeChange(event.change, nodes).map((line) => <div key={line}>{line}</div>)}</Box>
      <Box sx={{ color: 'text.secondary' }}>時間</Box>
      <Box>{event.occurred_at ?? '原文未記錄'}</Box>
      <Box sx={{ color: 'text.secondary' }}>修改原因</Box>
      <Box>
        {event.reason ? (
          <Stack gap={0.5}>
            <span>{event.reason}</span>
            <Chip size="small" sx={{ alignSelf: 'flex-start' }}
              color={event.reason_source === 'human' ? 'success' : 'default'}
              label={event.reason_source === 'human' ? '原文記載（引文已核對）' : 'LLM 推論，僅供參考'} />
          </Stack>
        ) : '未記錄'}
      </Box>
      <Box sx={{ color: 'text.secondary' }}>原文依據</Box>
      <Box>{event.attributes.source_excerpt || '—'}</Box>
    </Box>
  )
}

function IssueList({ errors, warnings }: { errors: string[]; warnings: string[] }) {
  return (
    <Stack gap={1}>
      {errors.map((m) => <Alert key={`e-${m}`} severity="error" sx={{ py: 0 }} title={m}>{translateIssue(m)}</Alert>)}
      {warnings.map((m) => <Alert key={`w-${m}`} severity="warning" sx={{ py: 0 }} title={m}>{translateIssue(m)}</Alert>)}
    </Stack>
  )
}

function ChecksPanel({ detail }: { detail: ConversionDetail }) {
  const v = detail.validation
  if (v.ok === undefined) return <Typography color="text.secondary">尚未產生檢查結果。</Typography>
  return (
    <Stack gap={2}>
      <Alert severity={detail.can_approve ? 'success' : 'error'}>
        {detail.can_approve ? '沒有錯誤，可以核准。提醒事項不會阻擋核准，但請逐一確認。' : '有錯誤需要修正，修正前不能核准。'}
      </Alert>
      {v.schema_errors?.length ? (
        <Box>
          <Typography variant="subtitle2" fontWeight={700} gutterBottom>格式錯誤</Typography>
          <IssueList errors={v.schema_errors} warnings={[]} />
        </Box>
      ) : null}
      {(v.episodes ?? []).map((ep, i) => (
        <Box key={ep.episode_id}>
          <Typography variant="subtitle2" fontWeight={700} gutterBottom>
            序列 {i + 1}　<Box component="span" sx={{ fontFamily: 'monospace', fontWeight: 400, color: 'text.secondary' }}>{ep.episode_id}</Box>・{ep.event_count} 個修改
          </Typography>
          {ep.errors.length === 0 && ep.warnings.length === 0
            ? <Typography variant="body2" color="success.main">沒有問題。</Typography>
            : <IssueList errors={ep.errors} warnings={ep.warnings} />}
        </Box>
      ))}
      {v.notes && (
        <Box>
          <Typography variant="subtitle2" fontWeight={700} gutterBottom>LLM 的備註</Typography>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{v.notes}</Typography>
        </Box>
      )}
    </Stack>
  )
}

function DraftPanel({ caseId, detail, onSaved }: { caseId: string; detail: ConversionDetail; onSaved: (next: ConversionDetail) => void }) {
  const original = useMemo(() => (detail.draft === null || detail.draft === undefined ? '' : JSON.stringify(detail.draft, null, 2)), [detail.draft])
  const [text, setText] = useState(original)
  const [message, setMessage] = useState<{ severity: 'error' | 'success'; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { setText(original) }, [original])
  const editable = detail.status === 'needs_review' || detail.status === 'failed'

  const save = async () => {
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (err) {
      setMessage({ severity: 'error', text: `JSON 格式錯誤：${errorText(err)}` })
      return
    }
    setBusy(true)
    try {
      const next = await api.editDraft(caseId, detail.conversion_id, parsed)
      setMessage({ severity: 'success', text: next.can_approve ? '已儲存並重新檢查，沒有錯誤。' : '已儲存並重新檢查，仍有錯誤，請查看檢查結果。' })
      onSaved(next)
    } catch (err) {
      setMessage({ severity: 'error', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Stack gap={1.5}>
      <Typography variant="body2" color="text.secondary">
        可以直接修正 LLM 的草稿，例如更正活動類型、補上漏掉的連線或刪除捏造的修改。識別碼由 K 產生，不需要填寫。
      </Typography>
      <TextField id="draft-editor" multiline minRows={18} maxRows={40} value={text} onChange={(e) => setText(e.target.value)}
        disabled={!editable} inputProps={{ spellCheck: false, style: { fontFamily: 'monospace', fontSize: 13 } }} />
      {message && <Alert severity={message.severity}>{message.text}</Alert>}
      <Stack direction="row" gap={1}>
        <Button variant="contained" onClick={save} disabled={!editable || busy || text === original}>儲存並重新檢查</Button>
        <Button onClick={() => { setText(original); setMessage(null) }} disabled={text === original}>還原</Button>
      </Stack>
    </Stack>
  )
}

function ReviewBar({ caseId, detail, onDone }: { caseId: string; detail: ConversionDetail; onDone: () => void }) {
  const [reviewer, setReviewer] = useState(readReviewer())
  const [note, setNote] = useState('')
  const [confirm, setConfirm] = useState<'approve' | 'reject' | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (detail.status === 'approved' || detail.status === 'rejected') return null

  const reviewable = detail.status === 'needs_review'
  const events = detail.preview.reduce((n, ep) => n + ep.events.length, 0)
  const approveBlocker = !reviewable ? '轉換尚未完成' : !detail.can_approve ? '還有錯誤，修正後才能核准' : !reviewer.trim() ? '請填寫審核人' : ''

  const submit = async () => {
    if (!confirm) return
    setBusy(true)
    setError('')
    try {
      saveReviewer(reviewer.trim())
      if (confirm === 'approve') await api.approve(caseId, detail.conversion_id, reviewer.trim(), note)
      else await api.reject(caseId, detail.conversion_id, reviewer.trim(), note)
      setConfirm(null)
      onDone()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Paper sx={{ p: 2, position: 'sticky', bottom: 0, zIndex: 2 }}>
      <Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }}>
        <TextField id="reviewer" size="small" label="審核人" value={reviewer} onChange={(e) => setReviewer(e.target.value)} sx={{ width: { md: 180 } }} />
        <TextField id="review-note" size="small" label="備註（選填）" value={note} onChange={(e) => setNote(e.target.value)} sx={{ flex: 1 }} />
        <Stack direction="row" gap={1}>
          <Button color="inherit" variant="outlined" startIcon={<CloseRounded />} onClick={() => setConfirm('reject')}
            disabled={!reviewer.trim() || isActive(detail.status)}>
            退回
          </Button>
          <Tooltip title={approveBlocker}>
            <span>
              <Button variant="contained" color="success" startIcon={<CheckRounded />} onClick={() => setConfirm('approve')} disabled={!!approveBlocker}>
                核准並匯入
              </Button>
            </span>
          </Tooltip>
        </Stack>
      </Stack>
      {error && <Alert severity="error" sx={{ mt: 1.5 }}>{error}</Alert>}

      <Dialog open={!!confirm} onClose={() => setConfirm(null)}>
        <DialogTitle>{confirm === 'approve' ? '核准並匯入？' : '退回這次轉換？'}</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            {confirm === 'approve'
              ? `將匯入 ${detail.preview.length} 條演變序列、共 ${events} 個修改事件，並記錄審核人「${reviewer}」。匯入後會成為訓練資料，無法在此畫面撤回。`
              : '退回後這次轉換不會匯入。之後可以修改提示詞，再對同一份文件重新轉換。'}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirm(null)}>取消</Button>
          <Button variant="contained" color={confirm === 'approve' ? 'success' : 'inherit'} onClick={submit} disabled={busy}>
            {confirm === 'approve' ? '核准並匯入' : '退回'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  )
}
