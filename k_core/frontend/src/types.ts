// 與後端 kcore（/api/v1）回應對齊的型別

export interface CaseInfo {
  case_id: string
  name: string
  domain_kind: 'erp' | 'cvflow' | 'hardware' | 'other'
  erp_instance_id: string | null
  recording_enabled: boolean
  episodes: { open: number; finalized: number }
  active_model_version_id: string | null
  created_at: string
}

export interface Profile {
  profile_id: string
  version: number
  case_id: string
  name: string
  base_url: string
  model: string
  api_key_env: string | null
  system_prompt: string
  user_prompt_template: string
  temperature: number
  max_tokens: number | null
  structured_output: boolean
  timeout_seconds: number
  created_at: string
}

export type ProfileInput = Omit<Profile, 'version' | 'case_id' | 'created_at'>

export interface ProfileTestResult {
  reachable: boolean
  models?: string[]
  model_available: boolean
  error?: string
}

export interface DocumentInfo {
  document_id: string
  case_id: string
  filename: string
  media_type: string | null
  encoding: string
  sha256: string
  characters: number
  uploaded_at: string
  content?: string
  duplicate?: boolean
}

export type ConversionStatus = 'queued' | 'running' | 'failed' | 'needs_review' | 'approved' | 'rejected'

export interface EpisodeReport {
  episode_id: string
  event_count: number
  errors: string[]
  warnings: string[]
}

export interface Validation {
  ok?: boolean
  schema_errors?: string[]
  episodes?: EpisodeReport[]
  notes?: string | null
  already_ingested?: string[]
}

export interface Conversion {
  conversion_id: string
  case_id: string
  document_id: string
  profile_id: string
  profile_version: number
  status: ConversionStatus
  error: string | null
  can_approve: boolean
  draft_edited: boolean
  reviewer: string | null
  review_note: string | null
  ingested_episode_ids: string[]
  validation: Validation
  created_at: string | null
  completed_at: string | null
  reviewed_at: string | null
}

export interface AoVNode {
  node_id: string
  activity_type: string
  name?: string | null
  parameters?: Record<string, unknown>
}

export interface AoVEdge {
  source: string
  target: string
}

export interface NodeUpdate {
  node_id: string
  parameters_set?: Record<string, unknown>
  parameters_unset?: string[]
  name?: string
  activity_type?: string
}

export interface Change {
  added_nodes: AoVNode[]
  removed_nodes: string[]
  added_edges: AoVEdge[]
  removed_edges: AoVEdge[]
  updated_node_attributes: NodeUpdate[]
}

export interface PreviewEvent {
  event_id: string
  sequence_index: number
  occurred_at: string | null
  operation_type: string
  change: Change
  reason: string | null
  reason_source: 'human' | 'llm' | null
  attributes: { source_excerpt?: string | null; reason_quote?: string | null }
}

export interface PreviewVersion {
  version_id: string
  event_index: number
  nodes: AoVNode[]
  edges: AoVEdge[]
}

export interface PreviewEpisode {
  episode_id: string
  versions: PreviewVersion[]
  stopped_at: number | null
  events: PreviewEvent[]
}

export interface ConversionDetail extends Conversion {
  draft: unknown
  raw_response: string | null
  preview: PreviewEpisode[]
}

// ---- 研究頁面 ----
export interface EpisodeSummary {
  episode_id: string
  case_id: string
  work_order_id: string | null
  workflow_id: string
  status: 'open' | 'finalized'
  is_synthetic: boolean
  initial_version_id: string
  final_version_id: string | null
  event_count: number
  provenance: Record<string, unknown> & { teacher?: string; outcome?: { final_ok?: boolean | null } }
  started_at: string
  finalized_at: string | null
}

export interface EpisodeEvent {
  event_id: string
  sequence_index: number
  operation_type: string
  change: Change
  reason: string | null
  reason_source: 'human' | 'llm' | 'actor' | null
  attributes: Record<string, unknown>
  features: Record<string, unknown> | null
}

export interface EpisodeDetail extends EpisodeSummary {
  initial_context: Record<string, unknown>
  events: EpisodeEvent[]
  snapshots: { version_id: string; nodes: AoVNode[]; edges: AoVEdge[] }[]
}

export interface MethodMetrics {
  events: number
  ranked_steps: number
  log_likelihood_per_event: number
  perplexity: number
  next_top1: number
  next_top3: number
  next_mrr: number
}

export interface ModelSummary {
  model_version_id: string
  case_id: string
  training_run_id: string
  feature_schema: { version: string; fields: string[] }
  status: 'active' | 'inactive'
  created_at: string
  n_states: number
  n_observations: number
  selected_k: number | null
  test_metrics: Record<string, MethodMetrics> | null
}

export interface Prediction { observation: string; probability: number }

export interface ReplayStep {
  sequence_index: number
  event_id: string
  operation_type: string
  token: string
  encoded_as: string
  route_before: Record<string, number>
  belief_after: Record<string, number>
  predictions: Record<string, { top: Prediction[]; actual_rank: number; actual_probability: number }>
}

export interface ReplayResult {
  episode_id: string
  model_version_id: string
  states: string[]
  methods: string[]
  default_method: string
  steps: ReplayStep[]
  summary: Record<string, { top1: number | null; top3: number | null; mean_log_probability: number | null }>
}

export interface StateInfo {
  state: string
  events: number
  self_transition: number
  initial: number
  top_observations: Prediction[]
  operation_share: Record<string, number>
}

export interface ModelAnalysis {
  model_version_id: string
  episodes: number
  events: number
  states: StateInfo[]
  transition: number[][]
  episodes_with_state_switch: number
  group_by: string | null
  state_share_by_group: Record<string, Record<string, number>>
  state_share_by_outcome: Record<string, Record<string, number>>
  test_metrics: Record<string, MethodMetrics> | null
  training: { selected_k: number | null; candidates: { k: number; log_likelihood: number; bic: number; converged: boolean }[] }
  feature_schema: { version: string; fields: string[] }
}
