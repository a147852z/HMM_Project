import type {
  AssistEvaluation, AssistResult, CaseInfo, Conversion, ConversionDetail, DocumentInfo, EpisodeDetail, EpisodeSummary, ModelAnalysis, ModelSummary,
  Granularity, Profile, ProfileInput, ProfileTestResult, ReplayResult, StateMachine, TaskStateMachine, TeacherStateMachine,
} from './types'

const BASE = '/api/v1'

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'network', '無法連線到 K 後端，請確認後端已在 8900 埠啟動。')
  }
  const text = await response.text()
  const body = text ? JSON.parse(text) : null
  if (!response.ok) {
    const error = body?.error
    const detail = body?.detail
    throw new ApiError(
      response.status,
      error?.code ?? 'http_error',
      error?.message ?? (response.status === 404 && !error
        ? '找不到這項功能的 API。後端可能是較舊的版本，請重新啟動後端（k_core/backend）後再試。'
        : typeof detail === 'string' ? detail : `請求失敗（HTTP ${response.status}）`),
      error?.details ?? detail,
    )
  }
  return body as T
}

const json = (method: string, data: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data),
})

const c = (caseId: string) => `/cases/${encodeURIComponent(caseId)}`

export const api = {
  listCases: () => request<CaseInfo[]>('/cases'),
  createCase: (data: { case_id: string; name: string; domain_kind: string }) =>
    request<CaseInfo>('/cases', json('POST', data)),

  listEpisodes: (caseId: string) => request<EpisodeSummary[]>(`${c(caseId)}/episodes`),
  getEpisode: (caseId: string, episodeId: string) =>
    request<EpisodeDetail>(`${c(caseId)}/episodes/${encodeURIComponent(episodeId)}`),
  listModels: (caseId: string) => request<ModelSummary[]>(`${c(caseId)}/models`),
  modelAnalysis: (caseId: string, modelId: string, groupBy?: string) =>
    request<ModelAnalysis>(`${c(caseId)}/models/${encodeURIComponent(modelId)}/analysis`
      + (groupBy ? `?group_by=${encodeURIComponent(groupBy)}` : '')),
  replay: (caseId: string, modelId: string, episodeId: string) =>
    request<ReplayResult>(`${c(caseId)}/models/${encodeURIComponent(modelId)}/replay`,
      json('POST', { episode_id: episodeId, top_k: 5 })),

  stateMachine: (caseId: string, episodeId: string, granularity: Granularity) =>
    request<StateMachine>(`${c(caseId)}/episodes/${encodeURIComponent(episodeId)}/state-machine`
      + `?granularity=${granularity}`),

  taskStateMachine: (caseId: string, taskId: string, granularity: Granularity) =>
    request<TaskStateMachine>(`${c(caseId)}/tasks/${encodeURIComponent(taskId)}/state-machine`
      + `?granularity=${granularity}`),

  teacherStateMachine: (caseId: string, teacher: string, granularity: Granularity) =>
    request<TeacherStateMachine>(`${c(caseId)}/teachers/${encodeURIComponent(teacher)}/state-machine`
      + `?granularity=${granularity}`),

  assistEvaluation: (caseId: string) => request<AssistEvaluation>(`${c(caseId)}/aov/assist-evaluation`),
  assist: (caseId: string, episodeId: string, prefixLength: number, usePreviousResult = false) =>
    request<AssistResult>(`${c(caseId)}/aov/assist`, json('POST', {
      episode_id: episodeId, prefix_length: prefixLength, top_k: 10, use_previous_result: usePreviousResult,
    })),

  listProfiles: (caseId: string) => request<Profile[]>(`${c(caseId)}/llm-profiles`),
  createProfile: (caseId: string, data: ProfileInput) =>
    request<Profile>(`${c(caseId)}/llm-profiles`, json('POST', data)),
  updateProfile: (caseId: string, profileId: string, data: Partial<ProfileInput>) =>
    request<Profile>(`${c(caseId)}/llm-profiles/${encodeURIComponent(profileId)}`, json('PUT', data)),
  testProfile: (caseId: string, profileId: string) =>
    request<ProfileTestResult>(`${c(caseId)}/llm-profiles/${encodeURIComponent(profileId)}/test`, { method: 'POST' }),

  listDocuments: (caseId: string) => request<DocumentInfo[]>(`${c(caseId)}/documents`),
  getDocument: (caseId: string, documentId: string) =>
    request<DocumentInfo>(`${c(caseId)}/documents/${encodeURIComponent(documentId)}`),
  uploadDocument: (caseId: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<DocumentInfo>(`${c(caseId)}/documents`, { method: 'POST', body: form })
  },

  listConversions: (caseId: string) => request<Conversion[]>(`${c(caseId)}/conversions`),
  getConversion: (caseId: string, id: string) =>
    request<ConversionDetail>(`${c(caseId)}/conversions/${encodeURIComponent(id)}`),
  createConversion: (caseId: string, documentId: string, profileId: string) =>
    request<Conversion>(`${c(caseId)}/conversions`, json('POST', { document_id: documentId, profile_id: profileId })),
  editDraft: (caseId: string, id: string, draft: unknown) =>
    request<ConversionDetail>(`${c(caseId)}/conversions/${encodeURIComponent(id)}/draft`, json('PUT', { draft })),
  approve: (caseId: string, id: string, reviewer: string, note: string) =>
    request<Conversion>(`${c(caseId)}/conversions/${encodeURIComponent(id)}/approve`,
      json('POST', { reviewer, note: note || null })),
  reject: (caseId: string, id: string, reviewer: string, note: string) =>
    request<Conversion>(`${c(caseId)}/conversions/${encodeURIComponent(id)}/reject`,
      json('POST', { reviewer, note: note || null })),
}

export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return String(error)
}
