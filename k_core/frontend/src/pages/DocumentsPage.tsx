import UploadFileRounded from '@mui/icons-material/UploadFileRounded'
import {
  Alert, Box, Button, Dialog, DialogContent, DialogTitle, Paper, Stack, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Typography,
} from '@mui/material'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, errorText } from '../api'
import { formatTime } from '../labels'
import type { DocumentInfo } from '../types'

export function DocumentsPage({ caseId }: { caseId: string }) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [message, setMessage] = useState<{ severity: 'success' | 'info' | 'error'; text: string } | null>(null)
  const [viewing, setViewing] = useState<DocumentInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const load = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments(caseId))
    } catch (err) {
      setMessage({ severity: 'error', text: errorText(err) })
    }
  }, [caseId])
  useEffect(() => { void load() }, [load])

  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    setBusy(true)
    const results: string[] = []
    let failed = false
    for (const file of Array.from(files)) {
      try {
        const doc = await api.uploadDocument(caseId, file)
        results.push(doc.duplicate ? `${file.name}：內容與既有文件相同，未重複建立` : `${file.name}：已上傳（${doc.encoding}）`)
      } catch (err) {
        failed = true
        results.push(`${file.name}：${errorText(err)}`)
      }
    }
    setMessage({ severity: failed ? 'error' : 'success', text: results.join('\n') })
    if (input.current) input.current.value = ''
    setBusy(false)
    await load()
  }

  const open = async (doc: DocumentInfo) => {
    try {
      setViewing(await api.getDocument(caseId, doc.document_id))
    } catch (err) {
      setMessage({ severity: 'error', text: errorText(err) })
    }
  }

  return (
    <Stack gap={3}>
      <Paper sx={{ p: 2.5 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} gap={2} alignItems={{ sm: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="subtitle1" fontWeight={700}>上傳原始文件</Typography>
            <Typography variant="body2" color="text.secondary">
              目前支援文字檔（.txt、.csv、.md、.json 等），會自動判斷 UTF-8 或 Big5 編碼。Excel、PDF 請先轉成文字。
            </Typography>
          </Box>
          <Button variant="contained" component="label" startIcon={<UploadFileRounded />} disabled={busy}>
            選擇檔案
            <input id="document-upload" ref={input} type="file" hidden multiple onChange={(e) => upload(e.target.files)} />
          </Button>
        </Stack>
        {message && <Alert severity={message.severity} sx={{ mt: 2, whiteSpace: 'pre-line' }}>{message.text}</Alert>}
      </Paper>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>檔名</TableCell>
              <TableCell align="right">字數</TableCell>
              <TableCell>編碼</TableCell>
              <TableCell>文件代號</TableCell>
              <TableCell>上傳時間</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {documents.length === 0 && (
              <TableRow><TableCell colSpan={5}>
                <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>尚未上傳任何文件。</Typography>
              </TableCell></TableRow>
            )}
            {[...documents].reverse().map((doc) => (
              <TableRow key={doc.document_id} hover sx={{ cursor: 'pointer' }} onClick={() => open(doc)}>
                <TableCell>{doc.filename}</TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>{doc.characters.toLocaleString()}</TableCell>
                <TableCell>{doc.encoding}</TableCell>
                <TableCell sx={{ fontFamily: 'monospace' }}>{doc.document_id}</TableCell>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>{formatTime(doc.uploaded_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={!!viewing} onClose={() => setViewing(null)} fullWidth maxWidth="md">
        <DialogTitle>{viewing?.filename}</DialogTitle>
        <DialogContent dividers>
          <Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'inherit', fontSize: 14 }}>
            {viewing?.content}
          </Box>
        </DialogContent>
      </Dialog>
    </Stack>
  )
}
