import { Box, Typography } from '@mui/material'
import { useEffect, useMemo, useRef } from 'react'

export interface Highlight {
  text: string
  kind: 'excerpt' | 'quote'
}

function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** 在原文中找出片段位置；先精確比對，找不到時忽略空白差異再比對一次。 */
function findRange(content: string, snippet: string): [number, number] | null {
  const trimmed = snippet.trim()
  if (!trimmed) return null
  const exact = content.indexOf(trimmed)
  if (exact >= 0) return [exact, exact + trimmed.length]
  const chars = [...trimmed.replace(/\s+/g, '')]
  if (chars.length === 0) return null
  const match = new RegExp(chars.map(escapeRegExp).join('\\s*')).exec(content)
  return match ? [match.index, match.index + match[0].length] : null
}

/** 顯示原始文件，並以底色標出目前步驟在原文中的依據片段與原因引文。 */
export function SourceDocument({ content, highlights }: { content: string; highlights: Highlight[] }) {
  const firstMark = useRef<HTMLElement | null>(null)

  const { parts, missing } = useMemo(() => {
    // 逐字標記：原因引文優先於修改依據，因此引文落在依據片段內時也看得到
    const marks: (Highlight['kind'] | undefined)[] = new Array(content.length)
    const notFound: Highlight[] = []
    const ordered = [...highlights.filter((h) => h.kind === 'excerpt'), ...highlights.filter((h) => h.kind === 'quote')]
    for (const highlight of ordered) {
      const range = findRange(content, highlight.text)
      if (!range) { notFound.push(highlight); continue }
      for (let i = range[0]; i < range[1]; i += 1) marks[i] = highlight.kind
    }
    const result: { text: string; kind?: Highlight['kind'] }[] = []
    let start = 0
    for (let i = 1; i <= content.length; i += 1) {
      if (i === content.length || marks[i] !== marks[start]) {
        result.push({ text: content.slice(start, i), kind: marks[start] })
        start = i
      }
    }
    return { parts: result, missing: notFound }
  }, [content, highlights])

  // 只捲動原文面板本身，不捲動整個頁面
  useEffect(() => {
    const mark = firstMark.current
    if (!mark) return
    let panel = mark.parentElement
    while (panel && !(panel.scrollHeight > panel.clientHeight && /(auto|scroll)/.test(getComputedStyle(panel).overflowY))) {
      panel = panel.parentElement
    }
    if (!panel || panel === document.scrollingElement || panel === document.body) return
    const offset = mark.getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop
    panel.scrollTo({ top: Math.max(0, offset - panel.clientHeight / 2), behavior: 'smooth' })
  }, [parts])

  let assigned = false
  return (
    <Box>
      {missing.length > 0 && (
        <Typography variant="body2" color="warning.main" sx={{ mb: 1 }}>
          原文中找不到：{missing.map((m) => `「${m.text}」`).join('、')}
        </Typography>
      )}
      <Box component="pre" sx={{
        m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'inherit', fontSize: 14, lineHeight: 1.8,
      }}>
        {parts.map((part, index) => {
          if (!part.kind) return <span key={index}>{part.text}</span>
          const ref = !assigned ? (el: HTMLElement | null) => { firstMark.current = el } : undefined
          assigned = true
          return (
            <Box key={index} component="mark" ref={ref} sx={{
              bgcolor: part.kind === 'quote' ? 'success.light' : 'warning.light',
              color: 'text.primary', borderRadius: 0.5, px: 0.25,
            }}>
              {part.text}
            </Box>
          )
        })}
      </Box>
    </Box>
  )
}
