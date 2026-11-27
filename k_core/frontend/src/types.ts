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

// ---- 生成協助（離線測試）----
export interface AssistCandidate {
  rank: number
  operation_type: string
  activity_type: string
  token: string
  score: number
  token_probability: number
  pattern_prior: number
  training_count: number
  detail: Record<string, unknown>
  validation: { valid: boolean; problems: string[] }
  proposed_workflow: { nodes: AoVNode[]; edges: AoVEdge[] }
  change: Change
}

export interface AssistResult {
  episode_id: string
  prefix_length: number
  total_events: number
  model_version_id: string
  method: string
  status: 'ok' | 'no_valid_candidates'
  use_previous_result: boolean
  previous_result: 'pass' | 'fail' | 'error' | null
  in_training_set: boolean
  current_workflow: { version_id: string; nodes: AoVNode[]; edges: AoVEdge[] }
  candidate_stats: { patterns: number; infeasible: number; invalid: number; valid_candidates: number }
  candidates: AssistCandidate[]
  actual: null | {
    operation_type: string
    activity_type: string
    in_candidates: boolean
    ranks: Record<'assist' | 'assist_result' | 'bigram' | 'frequency', number | null>
  }
}

export interface AssistEvaluation {
  model_version_id: string
  method: string
  test_episodes: number
  train_episodes: number
  steps: number
  coverage: number
  mean_candidates: number
  invalid_candidates_dropped: number
  pattern_library: { patterns: number; templates: number; unsupported_training_events: number }
  methods: Record<'assist' | 'assist_result' | 'bigram' | 'frequency', { hit1: number; hit3: number; hit5: number; mrr: number }>
  by_previous_result: Record<string, { steps: number; assist_hit3: number; assist_result_hit3: number }>
  changed_steps: { episode_id: string; prefix_length: number; previous_result: string | null; assist_rank: number | null; assist_result_rank: number | null }[]
  by_operation: Record<string, { steps: number; covered: number; hit3: number; coverage: number; assist_hit3: number }>
  episodes: { episode_id: string; events: number; assist_hit3: number | null }[]
}

// ------------------------------------------------------------------ 有限狀態機（一條演變過程）
export type Granularity = 'full' | 'structure' | 'category'

export interface FsmState {
  id: string
  order: number
  summary: string[]
  n_nodes: number
  visits: number[]
  version_ids: string[]
  hidden_states: Record<string, number>
  is_start: boolean
  is_final: boolean
}

export interface FsmEvent {
  step: number
  from_version: number
  to_version: number
  event_id: string
  operation_type: string
  activity_type: string | null
  change: Change
  reason: string | null
  token?: string
  hidden_state?: string
  posterior?: Record<string, number>
  hidden_transition?: { from: string | null; to: string; probability: number }
  model_probability?: number
  model_rank?: number
}

export interface FsmTransition {
  id: string
  source: string
  target: string
  count: number
  probability: number
  self_loop: boolean
  operations: Record<string, number>
  hidden_states: Record<string, number>
  events: FsmEvent[]
}

export interface StateMachine {
  case_id: string
  episode_id: string
  work_order_id: string | null
  teacher: string | null
  request: string | null
  granularity: Granularity
  path: string[]
  states: FsmState[]
  transitions: FsmTransition[]
  versions: { index: number; version_id: string; state: string; nodes: AoVNode[]; edges: AoVEdge[] }[]
  stats: { versions: number; states: number; transitions: number; self_loop_steps: number; revisits: number }
  hmm: null | {
    model_version_id: string
    states: string[]
    method: string
    top_observations: Record<string, { observation: string; probability: number }[]>
  }
}

// ------------------------------------------------------------------ 有限狀態機（同一題所有演變過程）
export interface TaskFsmState extends Omit<FsmState, 'visits' | 'version_ids'> {
  lane: number
  column: number
  visits: { episode_id: string; version: number }[]
  episodes: string[]
  starts: string[]
  finals: string[]
}

export interface TaskFsmEvent extends FsmEvent {
  episode_id: string
  lane: number
}

export interface TaskFsmTransition extends Omit<FsmTransition, 'events'> {
  episodes: string[]
  events: TaskFsmEvent[]
}

export interface TaskFsmLane {
  episode_id: string
  lane: number
  teacher: string | null
  scope: string | null
  run: string | null
  path: string[]
  events: number
  versions: { index: number; version_id: string; state: string; nodes: AoVNode[]; edges: AoVEdge[] }[]
}

export interface TaskStateMachine {
  case_id: string
  task_id: string
  request: string | null
  granularity: Granularity
  episodes: TaskFsmLane[]
  states: TaskFsmState[]
  transitions: TaskFsmTransition[]
  stats: {
    episodes: number; versions: number; states: number; transitions: number; shared_states: number
    self_loop_steps: number; revisits: number
  }
  hmm: StateMachine['hmm']
}

// ------------------------------------------------------------------ 有限狀態機（同一個解題 AI 的所有題目）
/** 一個版本的執行結果；unjudged＝沒有判分（例如沒有標準答案的照片題）。 */
export type FsmResult = 'pass' | 'fail' | 'error' | 'unjudged'

export interface TeacherFsmState {
  id: string
  order: number
  column: number
  /** 走到這個狀態的解題，平均在第幾版第一次走到（起點是 0） */
  first_reach: number
  summary: string[]
  n_nodes: number
  example: { episode_id: string; version: number; nodes: AoVNode[]; edges: AoVEdge[] }
  visits: number
  episodes: string[]
  tasks: string[]
  results: Partial<Record<FsmResult, number>>
  hidden_states: Record<string, number>
  starts: number
  finals: number
  final_results: Partial<Record<FsmResult, number>>
  judged_versions: number
  passed_versions: number
  reached_judged: number
  reached_passed: number
  /** 這個狀態的版本通過判分的比例（只算有判分的解題） */
  pass_rate: number | null
  /** 走到這裡的解題，最後通過的比例（只算有判分的解題） */
  reached_pass_rate: number | null
  is_start: boolean
  is_final: boolean
}

export interface TeacherFsmEvent {
  episode_id: string
  work_order_id: string | null
  step: number
  from_version: number
  to_version: number
  operation_type: string
  activity_type: string | null
  reason: string | null
  result: Exclude<FsmResult, 'unjudged'> | null
  judged: boolean
  token?: string
  hidden_state?: string
  posterior?: Record<string, number>
  hidden_transition?: { from: string | null; to: string; probability: number }
  model_probability?: number
  model_rank?: number
}

export interface TeacherFsmTransition {
  id: string
  source: string
  target: string
  count: number
  probability: number
  self_loop: boolean
  operations: Record<string, number>
  hidden_states: Record<string, number>
  results: Partial<Record<FsmResult, number>>
  /** 修改後那一版通過判分的比例（只算有判分的解題） */
  pass_rate: number | null
  episodes: string[]
  tasks: number
  events: TeacherFsmEvent[]
}

export interface TeacherFsmEpisode {
  episode_id: string
  work_order_id: string | null
  scope: string | null
  run: string | null
  path: string[]
  events: number
  results: (Exclude<FsmResult, 'unjudged'> | null)[]
  final_ok: boolean | null
}

export interface TeacherStateMachine {
  case_id: string
  teacher: string
  granularity: Granularity
  episodes: TeacherFsmEpisode[]
  states: TeacherFsmState[]
  transitions: TeacherFsmTransition[]
  stats: {
    episodes: number; tasks: number; versions: number; steps: number; states: number; transitions: number
    shared_states: number; self_loop_steps: number; revisits: number; judged_episodes: number; passed_episodes: number
  }
  hmm: StateMachine['hmm']
}
