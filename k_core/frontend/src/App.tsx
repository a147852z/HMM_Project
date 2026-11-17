import AddRounded from '@mui/icons-material/AddRounded'
import {
  Alert, AppBar, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Stack, Tab, Tabs,
  TextField, Toolbar, Typography,
} from '@mui/material'
import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from './api'
import { DOMAIN_LABEL } from './labels'
import { AssistPage } from './pages/AssistPage'
import { ConversionsPage } from './pages/ConversionsPage'
import { DocumentsPage } from './pages/DocumentsPage'
import { ProfilesPage } from './pages/ProfilesPage'
import { ResearchPage } from './pages/ResearchPage'
import { ReviewView } from './pages/ReviewView'
import type { CaseInfo } from './types'

export type Page = 'research' | 'assist' | 'conversions' | 'documents' | 'profiles'

const CASE_KEY = 'kcore.caseId'

function readStored(key: string): string {
  try { return localStorage.getItem(key) ?? '' } catch { return '' }
}
function readHash(): string | null {
  const match = /^#review\/(.+)$/.exec(window.location.hash)
  return match ? decodeURIComponent(match[1]) : null
}
function writeStored(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* 瀏覽器不允許儲存時忽略 */ }
}

export function App() {
  const [cases, setCases] = useState<CaseInfo[] | null>(null)
  const [caseId, setCaseId] = useState(readStored(CASE_KEY))
  const [page, setPage] = useState<Page>('research')
  // 網址 #review/<轉換代號> 可以直接開啟某次轉換的審核畫面，方便分享給其他審核者
  const [openConversion, setOpenConversionState] = useState<string | null>(readHash)
  const setOpenConversion = useCallback((id: string | null) => {
    setOpenConversionState(id)
    const hash = id ? `#review/${encodeURIComponent(id)}` : ''
    if (window.location.hash !== hash) history.replaceState(null, '', hash || window.location.pathname)
  }, [])
  useEffect(() => {
    const onHash = () => setOpenConversionState(readHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const [loadError, setLoadError] = useState('')
  const [creating, setCreating] = useState(false)

  const loadCases = useCallback(async () => {
    try {
      const list = await api.listCases()
      setCases(list)
      setLoadError('')
      // 從網址開啟的轉換代號以「案例代號-C」開頭，據此選擇對應案例
      const fromHash = readHash()
      // #case/<案例代號> 直接開啟該案例；#review/<轉換代號> 依轉換代號推得案例
      // #case/<案例代號>/<分頁> 也可以直接開啟指定分頁，例如 #case/CVFLOW/assist
      const caseHash = /^#case\/([^/]+)(?:\/(research|assist|conversions|documents|profiles))?$/.exec(window.location.hash)
      if (caseHash?.[2]) setPage(caseHash[2] as Page)
      const hashCase = caseHash
        ? list.find((item) => item.case_id === decodeURIComponent(caseHash[1]))?.case_id
        : fromHash ? list.find((item) => fromHash.startsWith(`${item.case_id}-C`))?.case_id : undefined
      setCaseId((current) => hashCase ?? (list.some((item) => item.case_id === current) ? current : list[0]?.case_id ?? ''))
    } catch (error) {
      setLoadError(errorText(error))
      setCases([])
    }
  }, [])

  useEffect(() => { void loadCases() }, [loadCases])
  useEffect(() => { if (caseId) writeStored(CASE_KEY, caseId) }, [caseId])

  const selectCase = (id: string) => {
    if (id !== caseId) setOpenConversion(null)
    setCaseId(id)
  }

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <AppBar position="sticky" color="inherit" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar sx={{ gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="h6" fontWeight={700} sx={{ flexGrow: 1 }}>
            <Box component="span" sx={{ color: 'primary.main' }}>K</Box>　HMM 知識核心
          </Typography>
          {cases && cases.length > 0 && (
            <TextField
              id="case-select" select size="small" label="場域案例" value={caseId}
              onChange={(event) => selectCase(event.target.value)} sx={{ minWidth: 220 }}
            >
              {cases.map((item) => (
                <MenuItem key={item.case_id} value={item.case_id}>
                  {item.name}（{item.case_id}）
                </MenuItem>
              ))}
            </TextField>
          )}
          <Button startIcon={<AddRounded />} onClick={() => setCreating(true)}>新增案例</Button>
        </Toolbar>
        {caseId && !openConversion && (
          <Tabs value={page} onChange={(_, value: Page) => setPage(value)} sx={{ px: 2 }}>
            <Tab value="research" label="模型研究" />
            <Tab value="assist" label="生成協助測試" />
            <Tab value="conversions" label="轉換與審核" />
            <Tab value="documents" label="原始文件" />
            <Tab value="profiles" label="轉換設定（提示詞）" />
          </Tabs>
        )}
      </AppBar>

      <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, md: 3 } }}>
        {loadError && <Alert severity="error" sx={{ mb: 2 }}>{loadError}</Alert>}
        {cases && cases.length === 0 && !loadError && (
          <Alert severity="info" action={<Button onClick={() => setCreating(true)}>新增案例</Button>}>
            尚未建立任何場域案例。每個場域（例如 AP-K、AP-D）有自己的提示詞、資料與模型，請先新增一個。
          </Alert>
        )}
        {caseId && openConversion && (
          <ReviewView caseId={caseId} conversionId={openConversion} onBack={() => setOpenConversion(null)} />
        )}
        {caseId && !openConversion && page === 'conversions' && (
          <ConversionsPage caseId={caseId} onOpen={setOpenConversion} onNavigate={setPage} />
        )}
        {caseId && !openConversion && page === 'research' && <ResearchPage caseId={caseId} />}
        {caseId && !openConversion && page === 'assist' && <AssistPage caseId={caseId} />}
        {caseId && !openConversion && page === 'documents' && <DocumentsPage caseId={caseId} />}
        {caseId && !openConversion && page === 'profiles' && <ProfilesPage caseId={caseId} />}
      </Box>

      <CreateCaseDialog
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={async (created) => {
          setCreating(false)
          await loadCases()
          selectCase(created.case_id)
        }}
      />
    </Box>
  )
}

function CreateCaseDialog({ open, onClose, onCreated }: {
  open: boolean
  onClose: () => void
  onCreated: (created: CaseInfo) => void
}) {
  const [caseId, setCaseId] = useState('')
  const [name, setName] = useState('')
  const [kind, setKind] = useState('erp')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    setBusy(true)
    setError('')
    try {
      const created = await api.createCase({ case_id: caseId.trim(), name: name.trim() || caseId.trim(), domain_kind: kind })
      setCaseId('')
      setName('')
      onCreated(created)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle>新增場域案例</DialogTitle>
      <DialogContent>
        <Stack gap={2} sx={{ pt: 1 }}>
          <TextField id="new-case-id" label="案例代號" value={caseId} onChange={(e) => setCaseId(e.target.value)}
            helperText="英數字，例如 AP-K-ERP；建立後不可修改" required autoFocus />
          <TextField id="new-case-name" label="名稱" value={name} onChange={(e) => setName(e.target.value)}
            helperText="例如：高健" />
          <TextField id="new-case-kind" select label="來源種類" value={kind} onChange={(e) => setKind(e.target.value)}>
            {Object.entries(DOMAIN_LABEL).map(([value, label]) => <MenuItem key={value} value={value}>{label}</MenuItem>)}
          </TextField>
          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>取消</Button>
        <Button variant="contained" onClick={submit} disabled={!caseId.trim() || busy}>建立</Button>
      </DialogActions>
    </Dialog>
  )
}
