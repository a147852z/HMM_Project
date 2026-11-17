import AddRounded from '@mui/icons-material/AddRounded'
import {
  Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Paper, Stack,
  Switch, TextField, Typography,
} from '@mui/material'
import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from '../api'
import { formatTime } from '../labels'
import type { Profile, ProfileInput, ProfileTestResult } from '../types'
import { latestProfiles } from './ConversionsPage'

const EMPTY: ProfileInput = {
  profile_id: '',
  name: '',
  base_url: 'http://localhost:1234/v1',
  model: '',
  api_key_env: null,
  system_prompt: '',
  user_prompt_template: '以下是檔案 {filename} 的內容，請依系統指示整理：\n\n{document}',
  temperature: 0,
  max_tokens: null,
  structured_output: true,
  timeout_seconds: 600,
}

export function ProfilesPage({ caseId }: { caseId: string }) {
  const [all, setAll] = useState<Profile[]>([])
  const [editing, setEditing] = useState<{ mode: 'create' | 'edit'; value: ProfileInput } | null>(null)
  const [tests, setTests] = useState<Record<string, ProfileTestResult | 'testing'>>({})
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setAll(await api.listProfiles(caseId))
    } catch (err) {
      setError(errorText(err))
    }
  }, [caseId])
  useEffect(() => { void load() }, [load])

  const test = async (profileId: string) => {
    setTests((current) => ({ ...current, [profileId]: 'testing' }))
    try {
      const result = await api.testProfile(caseId, profileId)
      setTests((current) => ({ ...current, [profileId]: result }))
    } catch (err) {
      setTests((current) => ({ ...current, [profileId]: { reachable: false, model_available: false, error: errorText(err) } }))
    }
  }

  const latest = latestProfiles(all)

  return (
    <Stack gap={3}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" gap={2} flexWrap="wrap">
        <Box sx={{ maxWidth: 760 }}>
          <Typography variant="subtitle1" fontWeight={700}>轉換設定</Typography>
          <Typography variant="body2" color="text.secondary">
            每組設定包含本地 LLM 的網址、模型與這個案例的分析提示詞。K 會在系統提示詞後面自動附上固定的 JSON 格式說明。
            修改設定會建立新版本，舊的轉換紀錄仍保留當時使用的版本。
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<AddRounded />} onClick={() => setEditing({ mode: 'create', value: EMPTY })}>
          新增設定
        </Button>
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}
      {latest.length === 0 && <Alert severity="info">這個案例還沒有轉換設定。</Alert>}

      {latest.map((profile) => {
        const result = tests[profile.profile_id]
        const history = all.filter((item) => item.profile_id === profile.profile_id)
        return (
          <Paper key={profile.profile_id} sx={{ p: 2.5 }}>
            <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={2}>
              <Box sx={{ minWidth: 0 }}>
                <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
                  <Typography variant="subtitle1" fontWeight={700}>{profile.name}</Typography>
                  <Chip size="small" label={`第 ${profile.version} 版`} />
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>{profile.profile_id}</Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  {profile.model}　<Box component="span" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>{profile.base_url}</Box>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  溫度 {profile.temperature}・{profile.structured_output ? '要求結構化 JSON 輸出' : '一般輸出後擷取 JSON'}・
                  逾時 {profile.timeout_seconds} 秒・版本紀錄 {history.map((h) => `v${h.version}`).join('、')}
                </Typography>
              </Box>
              <Stack direction="row" gap={1} alignItems="flex-start">
                <Button variant="outlined" onClick={() => test(profile.profile_id)} disabled={result === 'testing'}>
                  {result === 'testing' ? '測試中…' : '測試連線'}
                </Button>
                <Button variant="outlined" onClick={() => setEditing({ mode: 'edit', value: { ...profile } })}>
                  修改（建立新版）
                </Button>
              </Stack>
            </Stack>
            {result && result !== 'testing' && <TestResult result={result} model={profile.model} />}
            <Box component="pre" sx={{
              mt: 2, mb: 0, p: 1.5, bgcolor: 'grey.50', border: 1, borderColor: 'divider', borderRadius: 1,
              whiteSpace: 'pre-wrap', fontSize: 13, maxHeight: 180, overflow: 'auto', fontFamily: 'inherit',
            }}>
              {profile.system_prompt}
            </Box>
            <Typography variant="caption" color="text.secondary">最後更新：{formatTime(profile.created_at)}</Typography>
          </Paper>
        )
      })}

      {editing && (
        <ProfileDialog
          caseId={caseId}
          mode={editing.mode}
          initial={editing.value}
          onClose={() => setEditing(null)}
          onSaved={async () => { setEditing(null); await load() }}
        />
      )}
    </Stack>
  )
}

function TestResult({ result, model }: { result: ProfileTestResult; model: string }) {
  if (!result.reachable) return <Alert severity="error" sx={{ mt: 2 }}>無法連線：{result.error}</Alert>
  if (!result.model_available) {
    return (
      <Alert severity="warning" sx={{ mt: 2 }}>
        已連線，但找不到模型「{model}」。可用的模型：{result.models?.join('、') || '（無）'}。請確認 LM Studio 已載入該模型，或修改設定中的模型名稱。
      </Alert>
    )
  }
  return <Alert severity="success" sx={{ mt: 2 }}>連線正常，模型「{model}」可以使用。</Alert>
}

function ProfileDialog({ caseId, mode, initial, onClose, onSaved }: {
  caseId: string
  mode: 'create' | 'edit'
  initial: ProfileInput
  onClose: () => void
  onSaved: () => void
}) {
  const [value, setValue] = useState<ProfileInput>(initial)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const set = <K extends keyof ProfileInput>(key: K, next: ProfileInput[K]) => setValue((v) => ({ ...v, [key]: next }))

  const save = async () => {
    setBusy(true)
    setError('')
    try {
      if (mode === 'create') {
        await api.createProfile(caseId, value)
      } else {
        const { profile_id: _ignored, ...changes } = value
        void _ignored
        await api.updateProfile(caseId, value.profile_id, changes)
      }
      onSaved()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  const valid = value.profile_id.trim() && value.name.trim() && value.model.trim() && value.base_url.trim()
    && value.system_prompt.trim()

  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle>{mode === 'create' ? '新增轉換設定' : `修改「${initial.name}」（將建立新版本）`}</DialogTitle>
      <DialogContent dividers>
        <Stack gap={2}>
          <Stack direction={{ xs: 'column', sm: 'row' }} gap={2}>
            <TextField id="profile-id" label="設定代號" value={value.profile_id} disabled={mode === 'edit'}
              onChange={(e) => set('profile_id', e.target.value)} helperText="英數字，例如 apk-default" required sx={{ flex: 1 }} />
            <TextField id="profile-name" label="名稱" value={value.name} onChange={(e) => set('name', e.target.value)}
              required sx={{ flex: 1 }} />
          </Stack>
          <Stack direction={{ xs: 'column', sm: 'row' }} gap={2}>
            <TextField id="profile-url" label="LLM 網址" value={value.base_url} onChange={(e) => set('base_url', e.target.value)}
              helperText="LM Studio 預設 http://localhost:1234/v1" required sx={{ flex: 2 }} />
            <TextField id="profile-model" label="模型名稱" value={value.model} onChange={(e) => set('model', e.target.value)}
              helperText="與 LM Studio 顯示的模型名稱相同" required sx={{ flex: 1 }} />
          </Stack>
          <TextField id="profile-system" label="系統提示詞（這個案例的分析方式）" value={value.system_prompt}
            onChange={(e) => set('system_prompt', e.target.value)} multiline minRows={6} required
            helperText="說明資料長什麼樣子、活動類型怎麼對應、哪些內容要忽略。JSON 格式要求由 K 自動附加，不需要寫。" />
          <TextField id="profile-user" label="使用者訊息範本" value={value.user_prompt_template}
            onChange={(e) => set('user_prompt_template', e.target.value)} multiline minRows={2}
            helperText="{document} 會被替換成文件內容，{filename} 會被替換成檔名。" />
          <Stack direction={{ xs: 'column', sm: 'row' }} gap={2} alignItems={{ sm: 'center' }}>
            <TextField id="profile-temp" label="溫度" type="number" value={value.temperature}
              onChange={(e) => set('temperature', Number(e.target.value))} inputProps={{ step: 0.1, min: 0, max: 2 }}
              helperText="建議 0，結果較穩定" sx={{ width: 140 }} />
            <TextField id="profile-timeout" label="逾時（秒）" type="number" value={value.timeout_seconds}
              onChange={(e) => set('timeout_seconds', Number(e.target.value))} sx={{ width: 140 }} />
            <TextField id="profile-key" label="API 金鑰環境變數（選填）" value={value.api_key_env ?? ''}
              onChange={(e) => set('api_key_env', e.target.value || null)} helperText="只存變數名稱，不存金鑰" sx={{ flex: 1 }} />
          </Stack>
          <FormControlLabel
            control={<Switch checked={value.structured_output} onChange={(e) => set('structured_output', e.target.checked)} />}
            label="要求模型依 JSON 結構輸出（服務不支援時請關閉）"
          />
          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>取消</Button>
        <Button variant="contained" onClick={save} disabled={!valid || busy}>
          {mode === 'create' ? '建立' : '儲存為新版本'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
