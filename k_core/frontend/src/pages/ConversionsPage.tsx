import PlayArrowRounded from '@mui/icons-material/PlayArrowRounded'
import {
  Alert, Box, Button, Chip, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Page } from '../App'
import { api, errorText } from '../api'
import { STATUS_COLOR, STATUS_LABEL, formatTime, isActive } from '../labels'
import type { Conversion, ConversionStatus, DocumentInfo, Profile } from '../types'

type Filter = 'all' | ConversionStatus

export function latestProfiles(profiles: Profile[]): Profile[] {
  const latest = new Map<string, Profile>()
  for (const profile of profiles) {
    const current = latest.get(profile.profile_id)
    if (!current || profile.version > current.version) latest.set(profile.profile_id, profile)
  }
  return [...latest.values()]
}

export function ConversionsPage({ caseId, onOpen, onNavigate }: {
  caseId: string
  onOpen: (id: string) => void
  onNavigate: (page: Page) => void
}) {
  const [conversions, setConversions] = useState<Conversion[]>([])
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [documentId, setDocumentId] = useState('')
  const [profileId, setProfileId] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      const [conv, docs, profs] = await Promise.all([
        api.listConversions(caseId), api.listDocuments(caseId), api.listProfiles(caseId),
      ])
      setConversions(conv)
      setDocuments(docs)
      setProfiles(latestProfiles(profs))
      setError('')
    } catch (err) {
      setError(errorText(err))
    }
  }, [caseId])

  useEffect(() => { void load() }, [load])

  // 有轉換正在進行時，每 3 秒更新一次清單
  const anyActive = conversions.some((item) => isActive(item.status))
  useEffect(() => {
    if (!anyActive) return
    const timer = window.setInterval(() => { void load() }, 3000)
    return () => window.clearInterval(timer)
  }, [anyActive, load])

  useEffect(() => {
    if (!documents.some((doc) => doc.document_id === documentId)) setDocumentId(documents.at(-1)?.document_id ?? '')
  }, [documents, documentId])
  useEffect(() => {
    if (!profiles.some((p) => p.profile_id === profileId)) setProfileId(profiles[0]?.profile_id ?? '')
  }, [profiles, profileId])

  const start = async () => {
    setBusy(true)
    try {
      await api.createConversion(caseId, documentId, profileId)
      await load()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  const docName = useMemo(() => new Map(documents.map((doc) => [doc.document_id, doc.filename])), [documents])
  const counts = useMemo(() => {
    const result: Partial<Record<ConversionStatus, number>> = {}
    for (const item of conversions) result[item.status] = (result[item.status] ?? 0) + 1
    return result
  }, [conversions])
  const visible = [...conversions].reverse().filter((item) => filter === 'all' || item.status === filter)

  return (
    <Stack gap={3}>
      {error && <Alert severity="error">{error}</Alert>}

      <Paper sx={{ p: 2.5 }}>
        <Typography variant="subtitle1" fontWeight={700} gutterBottom>開始新的轉換</Typography>
        {documents.length === 0 || profiles.length === 0 ? (
          <Alert severity="info" action={
            <Button onClick={() => onNavigate(documents.length === 0 ? 'documents' : 'profiles')}>
              {documents.length === 0 ? '上傳文件' : '建立轉換設定'}
            </Button>
          }>
            開始轉換前，需要至少一份原始文件與一組轉換設定（本地 LLM 網址與這個案例的提示詞）。
          </Alert>
        ) : (
          <Stack direction={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }}>
            <TextField id="conv-document" select size="small" label="原始文件" value={documentId} InputLabelProps={{ shrink: true }}
              onChange={(e) => setDocumentId(e.target.value)} sx={{ minWidth: 260, flex: 1 }}>
              {documents.map((doc) => (
                <MenuItem key={doc.document_id} value={doc.document_id}>
                  {doc.filename}（{doc.characters.toLocaleString()} 字）
                </MenuItem>
              ))}
            </TextField>
            <TextField id="conv-profile" select size="small" label="轉換設定" value={profileId} InputLabelProps={{ shrink: true }}
              onChange={(e) => setProfileId(e.target.value)} sx={{ minWidth: 260, flex: 1 }}>
              {profiles.map((p) => (
                <MenuItem key={p.profile_id} value={p.profile_id}>{p.name}（第 {p.version} 版・{p.model}）</MenuItem>
              ))}
            </TextField>
            <Button variant="contained" startIcon={<PlayArrowRounded />} onClick={start}
              disabled={busy || !documentId || !profileId}>
              開始轉換
            </Button>
          </Stack>
        )}
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
          轉換在背景進行，完成後狀態會變成「待審核」。本地模型處理長文件可能需要數分鐘。
        </Typography>
      </Paper>

      <Stack gap={1.5}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1}>
          <Typography variant="subtitle1" fontWeight={700}>轉換紀錄</Typography>
          <ToggleButtonGroup size="small" exclusive value={filter} onChange={(_, value) => value && setFilter(value)}>
            <ToggleButton value="all">全部 {conversions.length}</ToggleButton>
            {(['needs_review', 'running', 'failed', 'approved', 'rejected'] as ConversionStatus[]).map((status) => (
              <ToggleButton key={status} value={status}>{STATUS_LABEL[status]} {counts[status] ?? 0}</ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Stack>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>轉換</TableCell>
                <TableCell>原始文件</TableCell>
                <TableCell>轉換設定</TableCell>
                <TableCell>狀態</TableCell>
                <TableCell>檢查結果</TableCell>
                <TableCell>建立時間</TableCell>
                <TableCell>審核人</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {visible.length === 0 && (
                <TableRow><TableCell colSpan={7}>
                  <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>沒有符合的轉換紀錄。</Typography>
                </TableCell></TableRow>
              )}
              {visible.map((item) => (
                <TableRow key={item.conversion_id} hover onClick={() => onOpen(item.conversion_id)} sx={{ cursor: 'pointer' }}>
                  <TableCell sx={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{item.conversion_id}</TableCell>
                  <TableCell>{docName.get(item.document_id) ?? item.document_id}</TableCell>
                  <TableCell sx={{ whiteSpace: 'nowrap' }}>{item.profile_id}・v{item.profile_version}</TableCell>
                  <TableCell><Chip size="small" label={STATUS_LABEL[item.status]} color={STATUS_COLOR[item.status]} /></TableCell>
                  <TableCell><CheckSummary conversion={item} /></TableCell>
                  <TableCell sx={{ whiteSpace: 'nowrap' }}>{formatTime(item.created_at)}</TableCell>
                  <TableCell>{item.reviewer ?? '—'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Stack>
    </Stack>
  )
}

function CheckSummary({ conversion }: { conversion: Conversion }) {
  if (conversion.status === 'failed') {
    return <Typography variant="body2" color="error.main" noWrap sx={{ maxWidth: 260 }}>{conversion.error}</Typography>
  }
  const v = conversion.validation
  if (!v || v.ok === undefined) return <Typography variant="body2" color="text.secondary">—</Typography>
  const errors = (v.schema_errors?.length ?? 0) + (v.episodes ?? []).reduce((n, e) => n + e.errors.length, 0)
    + (v.already_ingested?.length ? 1 : 0)
  const warnings = (v.episodes ?? []).reduce((n, e) => n + e.warnings.length, 0)
  const sequences = v.episodes?.length ?? 0
  return (
    <Box sx={{ display: 'flex', gap: 1.5, typography: 'body2', whiteSpace: 'nowrap' }}>
      <span>{sequences} 條序列</span>
      <Box component="span" sx={{ color: errors ? 'error.main' : 'success.main' }}>{errors} 錯誤</Box>
      <Box component="span" sx={{ color: warnings ? 'warning.main' : 'text.secondary' }}>{warnings} 提醒</Box>
    </Box>
  )
}
