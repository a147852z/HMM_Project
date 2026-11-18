import { operationLabel } from './labels'

// 後端驗證訊息維持英文（API 慣例），在畫面上轉成中文；無法辨識的訊息原樣顯示。
const RULES: [RegExp, (m: RegExpExecArray) => string][] = [
  [/^initial_workflow: (.*)$/s, (m) => `初始流程：${translateIssue(m[1])}`],
  [/^operation_type derived by K as '(.+)'$/, (m) => `原文未標明修改類型，K 依差異判定為「${operationLabel(m[1])}」`],
  [/^reason marked human but its quote is not found verbatim/, () => '標示為「原文記載」的原因，在原文中找不到相同文字，已改為 LLM 推論'],
  [/^reason without reason_source is treated as LLM-inferred$/, () => '原因沒有標示來源，視為 LLM 推論'],
  [/^unknown reason_source '(.*)' treated as missing$/, (m) => `原因來源「${m[1]}」無法辨識，視為未記錄`],
  [/^provide change or after_workflow$/, () => '這次修改沒有提供 change 或 after_workflow'],
  [/^change and after_workflow disagree; used after_workflow$/, () => 'LLM 寫的修改差異與修改後的流程不一致，已依修改後的流程計算差異，請確認'],
  [/^change is empty$/, () => '這次修改沒有任何變更'],
  [/^workflow became disconnected$/, () => '修改後流程斷成好幾段，可能漏了連線（例如刪除站點後沒有接起前後兩站），請確認'],
  [/^node (.+) no longer has a next step$/, (m) => `${m[1]} 修改後沒有下一站，可能漏了連線，請確認`],
  [/^node (.+) no longer has a previous step$/, (m) => `${m[1]} 修改後沒有上一站，可能漏了連線，請確認`],
  [/^cycle detected among nodes: (.+)$/, (m) => `形成循環（活動：${m[1]}），AoV 不可以有循環`],
  [/^cannot remove missing node (.+)$/, (m) => `要刪除的活動 ${m[1]} 不存在`],
  [/^cannot remove missing edge (.+)$/, (m) => `要移除的依賴 ${m[1]} 不存在`],
  [/^cannot add existing node (.+)$/, (m) => `要新增的活動 ${m[1]} 已經存在`],
  [/^cannot add existing edge (.+)$/, (m) => `要新增的依賴 ${m[1]} 已經存在`],
  [/^cannot update missing node (.+)$/, (m) => `要更新的活動 ${m[1]} 不存在`],
  [/^edge (.+) references a missing node$/, (m) => `依賴 ${m[1]} 連到不存在的活動（刪除活動時要一併移除它的連線）`],
  [/^self-loop on node (.+)$/, (m) => `活動 ${m[1]} 連到自己`],
  [/^duplicate edge (.+)$/, (m) => `依賴 ${m[1]} 重複`],
  [/^duplicate node_id: (.+)$/, (m) => `活動代號 ${m[1]} 重複`],
  [/^node (.+) is missing activity_type$/, (m) => `活動 ${m[1]} 缺少活動類型`],
  [/^added node is missing node_id$/, () => '新增的活動缺少代號'],
  [/^episode has no events$/, () => '這條序列沒有任何修改'],
  [/^draft contains no episodes$/, () => '草稿中沒有任何演變序列'],
]

export function translateIssue(message: string): string {
  for (const [pattern, render] of RULES) {
    const match = pattern.exec(message)
    if (match) return render(match)
  }
  return message
}
