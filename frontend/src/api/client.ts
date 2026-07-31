import { API_BASE_URL } from "../config";

// Voice-first: the selected interaction language is any BCP-47 tag the realtime
// voice loop supports (~24, sourced from the backend catalog). It's a string —
// not a closed union — so selection scales with the API's supported set. The zh
// ASR-comparison variants below remain first-class named codes.
export type InteractionLanguage = string;

// zh ASR-comparison variants surfaced in the benchmark UI (kept explicit because
// they map to specific STT models, not distinct interaction languages).
export const INTERACTION_LANGUAGES: { code: InteractionLanguage; label: string }[] = [
  { code: "en-US", label: "English" },
  { code: "th-TH", label: "Thai" },
  { code: "id-ID", label: "Indonesian" },
  { code: "zh-CN", label: "Chinese (Qwen3)" },
  { code: "zh-CN-sensevoice", label: "Chinese (SenseVoice)" },
  { code: "zh-CN-paraformer", label: "Chinese (Paraformer)" },
];

export interface InteractionLanguageOption {
  code: InteractionLanguage;
  label?: string;
  english_name?: string;
  base?: string;
  stt_endpoint?: string;
}

export interface ZhAsrComparisonModel {
  key: string;
  label: string;
  endpoint: string;
  transcript: string;
  error?: string | null;
  elapsed_ms?: number | null;
  selected?: boolean;
}

export interface ZhAsrComparison {
  comparison_id: string;
  call_id: string;
  created_at: string;
  selected_language: InteractionLanguage;
  primary_transcript: string;
  browser_caption?: string | null;
  mime_type?: string | null;
  status: string;
  models: ZhAsrComparisonModel[];
  notes?: string[];
}

export interface Stage {
  key: string;
  label: string;
  provider?: string;
  path?: string;
  calls?: number;
  mode?: "real-time" | "streaming" | "batch" | "landing" | "managed sync";
  where?: string;
  latency?: string;
  job?: string;
  model?: string;
  status?: "running" | "done" | "idle";
  count?: number;
}

export interface JobState {
  name: string;
  available?: boolean;
  deployed?: boolean;
  running?: boolean;
  life_cycle_state?: string;
  last_result?: string | null;
}

export interface CallState {
  call_id: string;
  customer_id?: string;
  state?: {
    gold?: Record<string, unknown>;
    live?: Record<string, unknown>;
    utterances?: { text: string; speaker?: number; language?: InteractionLanguage }[];
  };
}

export interface CustomerWithIssue {
  customer_id: string;
  full_name?: string;
  customer_status?: string;
  call_id?: string | null;
  issue_status?: string;
  overdue_invoice_count?: number;
  overdue_amount?: number;
  autopay_enabled?: boolean | null;
  recent_declined_payments?: number;
  rationale?: string;
  primary_intent?: string;
  sentiment_label?: string;
  next_best_action?: string;
}

export interface AssistPipelineStep {
  key: string;
  label: string;
  status: "pending" | "active" | "done" | "error" | "skipped" | "complete";
  detail?: string;
  elapsed_ms?: number;
}

export interface StatusResponse {
  mode: string;
  deployment?: string;
  stt_provider?: string;
  enrichment?: { model_endpoint?: string };
  languages?: {
    default?: InteractionLanguage;
    supported?: InteractionLanguageOption[];
  };
  jobs?: { lakeflow?: JobState };
  stages: Stage[];
  counts: Record<string, unknown>;
  call_states: CallState[];
}

export interface LiveNudge {
  call_id: string;
  model?: string;
  language?: InteractionLanguage;
  live: Record<string, any>;
  transcript?: string;
  raw_transcript?: string;
  canonical_transcript?: string;
  agent_reply?: string | null;
  agent_validation?: {
    genie_validated?: boolean;
    mismatches?: string[];
    genie_error?: string | null;
    genie_skipped?: boolean;
    output_validated?: boolean;
    output_issues?: string[];
    language_validation?: {
      checked?: boolean;
      expected_language?: InteractionLanguage;
      matches?: boolean;
      reason?: string;
      script_ratio?: number;
    };
    language_repaired?: boolean;
    language_repair_error?: string;
    reply_available?: boolean;
    authoritative_metrics?: {
      overdue_invoice_count?: number;
      overdue_amount?: number;
      recent_declined_payments?: number;
    };
  } | null;
  billing?: {
    applied?: boolean;
    reason?: string;
    adjustment?: Record<string, unknown>;
    uc?: Record<string, unknown>;
  } | null;
  close_block_reason?: string | null;
  resolution?: {
    status?: string;
    note?: string;
    actions?: Record<string, unknown>;
    resolved_at?: string;
  };
  pipeline_steps?: AssistPipelineStep[];
  total_elapsed_ms?: number;
  zh_asr_comparison_id?: string | null;
  content_language?: InteractionLanguage;
  asr_provider?: string;
}

export interface AccountFacts {
  customer_id: string;
  found: boolean;
  customer?: Record<string, any> | null;
  invoices?: Record<string, any>[];
  payments?: Record<string, any>[];
  summary?: {
    open_invoice_count?: number;
    overdue_invoice_count?: number;
    overdue_amount?: number;
    autopay_enabled?: boolean | null;
    status?: string | null;
    recent_declined_payments?: number;
    issue_status?: string | null;
    resolution_note?: string | null;
    resolved_at?: string | null;
  };
}

export interface GenieResponse {
  question: string;
  language?: InteractionLanguage;
  canonical_question?: string;
  answer?: string;
  description?: string;
  sql?: string;
  rows?: unknown[][];
  columns?: string[];
  suggested_followups?: string[];
  language_validation?: {
    checked?: boolean;
    expected_language?: InteractionLanguage;
    matches?: boolean;
    reason?: string;
    script_ratio?: number;
  };
  language_repaired?: boolean;
  language_repair_error?: string;
  conversation_id?: string;
  message_id?: string;
}

export interface ResolutionEvent {
  event_id: string;
  call_id: string;
  event_type: string;
  issue_status?: string | null;
  note?: string | null;
  actions?: Record<string, unknown>;
  created_at?: string;
}

export interface ASRProviderSummary {
  clips: number;
  avg_wer?: number | null;
  avg_cer?: number | null;
  avg_entity_accuracy?: number | null;
  avg_critical_entity_accuracy?: number | null;
  empty_transcript_rate?: number | null;
  unsafe_for_resolution_rate?: number | null;
  negation_mismatch_rate?: number | null;
  numeric_recall?: number | null;
  latency_ms?: {
    avg?: number | null;
    p50?: number | null;
    p90?: number | null;
    p95?: number | null;
    p99?: number | null;
  };
  entity_groups?: Record<string, { expected: number; matched: number; accuracy?: number | null }>;
  unsafe_reason_counts?: Record<string, number>;
}

export interface ASRBenchmarkModelRanking {
  model_id: string;
  model_label?: string;
  avg_wer?: number | null;
  avg_cer?: number | null;
  p95_latency_ms?: number | null;
  avg_critical_entity_accuracy?: number | null;
  unsafe_for_resolution_rate?: number | null;
}

export interface ASRBenchmarkExample {
  clip_id: string;
  scenario?: string | null;
  reference_transcript?: string;
  deepgram_transcript?: string;
  databricks_transcript?: string;
  deepgram_wer?: number | null;
  databricks_wer?: number | null;
  deepgram_critical_entity_accuracy?: number | null;
  databricks_critical_entity_accuracy?: number | null;
  deepgram_latency_ms?: number | null;
  databricks_latency_ms?: number | null;
  deepgram_unsafe_reasons?: string[];
  databricks_unsafe_reasons?: string[];
}

export interface ASRBenchmarkResponse {
  available: boolean;
  source?: "ml_asr" | "legacy";
  tier?: "business" | "acoustic";
  dataset_id?: string;
  clip_count?: number;
  language?: InteractionLanguage;
  available_languages?: (InteractionLanguageOption & {
    available?: boolean;
    ml_asr_available?: boolean;
    legacy_available?: boolean;
  })[];
  message?: string;
  summary_path?: string;
  index_path?: string;
  deepgram_output?: string;
  databricks_output?: string;
  models?: Record<string, ASRProviderSummary>;
  ranking?: ASRBenchmarkModelRanking[];
  summary?: {
    providers?: {
      deepgram?: ASRProviderSummary;
      databricks?: ASRProviderSummary;
    };
    pairwise?: {
      paired_clips?: number;
      winner_counts?: Record<string, number>;
    };
    promotion_read?: {
      recommended_headline?: string;
      databricks_business_delta?: number | null;
      databricks_wer_delta?: number | null;
      databricks_p95_latency_delta_ms?: number | null;
      paired_clips?: number;
    };
  };
  examples?: ASRBenchmarkExample[];
}

export interface ASRBenchmarkModelMetrics {
  model_id: string;
  model_label: string;
  provider: "deepgram" | "databricks";
  clips: number;
  wer?: number | null;
  cer?: number | null;
  critical_entity_accuracy?: number | null;
  unsafe_for_resolution_rate?: number | null;
  p95_latency_ms?: number | null;
  entity_groups?: Record<string, { expected: number; matched: number; accuracy?: number | null }>;
  entity_groups_source?: string;
}

export interface ASRBenchmarkMetricWinner {
  model_id?: string;
  model_label?: string;
  provider?: "deepgram" | "databricks";
  tie?: boolean;
  model_ids?: string[];
  model_labels?: string[];
}

export interface ASRBenchmarkLanguageOverview {
  code: string;
  label: string;
  models: Record<string, ASRBenchmarkModelMetrics>;
  winners: Record<string, ASRBenchmarkMetricWinner>;
  entity_winners?: Record<string, ASRBenchmarkMetricWinner>;
  source?: string;
}

export interface ASRBenchmarkTierOverview {
  dataset_id: string;
  languages: Record<string, ASRBenchmarkLanguageOverview>;
  scoreboard: Record<string, Array<{ model_id: string; wins: number }>>;
}

export interface ASRBenchmarkOverviewResponse {
  available: boolean;
  source?: "ml_asr" | "legacy";
  index_path?: string;
  message?: string;
  tiers?: Record<string, ASRBenchmarkTierOverview>;
  available_languages?: (InteractionLanguageOption & {
    available?: boolean;
    ml_asr_available?: boolean;
    legacy_available?: boolean;
  })[];
}

// ---- Voice observability / eval traces ------------------------------------ //
export type TraceSpanKind = "STT" | "LLM" | "TOOL" | "TTS" | "GUARD";

export interface TraceSpan {
  name: string;
  kind: TraceSpanKind;
  start_ms: number;
  end_ms?: number | null;
  duration_ms?: number | null;
  status: string;
  input?: unknown;
  output?: unknown;
  attributes?: Record<string, unknown>;
}

export interface TraceSummary {
  trace_id: string;
  session_id?: string;
  turn_id?: number;
  call_id?: string | null;
  customer_id?: string | null;
  capability?: string;
  language?: string | null;
  detected_language?: string | null;
  status?: string;
  input_transcript?: string | null;
  output_text?: string | null;
  tool_names?: string[];
  apply_billing_action_called?: boolean;
  lookup_account_count?: number;
  llm_iterations?: number;
  /** Time to any audio; a latency filler ends the silence before the reply exists. */
  ttft_ms?: number | null;
  /** Time until the caller heard the actual reply — the latency to judge a turn on. */
  answer_ttft_ms?: number | null;
  /** TTS-local time to the answer's first chunk (excludes STT + LLM + tools). */
  tts_first_ms?: number | null;
  /** TTS endpoint's own time to first chunk; the rest of tts_first_ms is transport. */
  server_ttfb_ms?: number | null;
  server_gen_ms?: number | null;
  /** Whole turn including how long the agent then spoke — not perceived latency. */
  total_ms?: number | null;
  started_at?: string;
  created_at?: string;
}

export interface TraceDetail extends TraceSummary {
  error?: string | null;
  spans: TraceSpan[];
}

export interface TraceSessionRollup {
  session_id: string;
  call_id?: string | null;
  customer_id?: string | null;
  turns: number;
  languages: string[];
  apply_billing_action_called: boolean;
  lookup_account_total: number;
  statuses: Record<string, number>;
  latest?: string | null;
}

// ---- Multilingual voice benchmarks (realtime STT + TTS pipeline) ---------- //
// Served by the realtime API (mounted at /realtime) straight from the Delta
// benchmark_runs table. FLEURS reports error rates (WER/CER, lower better) and
// TTS round-trip intelligibility. Baselines are published FLEURS ASR reference
// numbers, not re-measured here.
export interface VoiceBenchmarkLatencyStat {
  p50?: number | null;
  p95?: number | null;
  p99?: number | null;
  mean?: number | null;
}

export interface VoiceBenchmarkRun {
  system?: string;
  system_label?: string;
  source?: string; // "measured"
  run_id?: string | null;
  dataset: string; // fleurs
  language?: string | null;
  evaluator?: string; // asr
  samples?: number | null;
  errors?: number | null;
  primary_score?: number | null;
  primary_metric?: string | null;
  scores?: Record<string, unknown>;
  latency_ms?: Record<string, VoiceBenchmarkLatencyStat | number | null>;
  issues?: { count?: number | null; by_kind?: Record<string, number> };
  wall_seconds?: number | null;
  status?: string | null;
  timestamp?: string | null;
}

export interface VoiceBenchmarkBaseline {
  system?: string;
  system_label?: string;
  source?: string; // "reference"
  dataset: string;
  evaluator?: string;
  language?: string | null;
  scope?: "aggregate" | "language";
  primary_metric?: string | null;
  primary_score?: number | null;
  scores?: Record<string, number>;
  reference_source?: string;
  reference_url?: string;
  note?: string;
}

export interface VoiceBenchmarksResponse {
  available: boolean;
  message?: string;
  source?: string;
  table?: string;
  run_id?: string | null;
  run_ids?: string[];
  generated_at?: string | null;
  datasets?: string[];
  languages?: string[];
  runs?: VoiceBenchmarkRun[];
  our_system?: { id: string; label: string };
  baselines?: VoiceBenchmarkBaseline[];
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  status: () => getJSON<StatusResponse>("/status"),
  health: () => getJSON<Record<string, unknown>>("/health"),
  asrBenchmarkOverview: (source?: "auto" | "ml_asr" | "legacy") => {
    const params = new URLSearchParams();
    if (source) params.set("source", source);
    const query = params.toString();
    return getJSON<ASRBenchmarkOverviewResponse>(`/asr-benchmark/overview${query ? `?${query}` : ""}`);
  },
  asrBenchmark: (
    language?: InteractionLanguage,
    options?: { tier?: "business" | "acoustic"; source?: "auto" | "ml_asr" | "legacy" }
  ) => {
    const params = new URLSearchParams();
    if (language) params.set("language", language);
    if (options?.tier) params.set("tier", options.tier);
    if (options?.source) params.set("source", options.source);
    const query = params.toString();
    return getJSON<ASRBenchmarkResponse>(`/asr-benchmark${query ? `?${query}` : ""}`);
  },
  customersWithIssues: () =>
    getJSON<{ customers: CustomerWithIssue[]; count: number }>("/accounts/with-issues"),
  callAccount: (callId: string) =>
    getJSON<AccountFacts>(`/calls/${callId}/account`),
  resolutionEvents: (callId: string) =>
    getJSON<{ call_id: string; events: ResolutionEvent[] }>(`/calls/${callId}/resolution-events`),
  resetDemoSession: async (callId: string): Promise<{ call_id: string; reset: boolean }> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/reset-demo-session`, {
      method: "POST",
      headers: { "content-type": "application/json" },
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  sendUtterance: async (
    callId: string,
    text: string,
    speaker?: number,
    language?: InteractionLanguage
  ): Promise<LiveNudge> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/assist`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, speaker, language }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  prefetchGenieInsight: async (
    callId: string,
    language?: InteractionLanguage
  ): Promise<{ call_id: string; genie_insight: { text: string; language?: InteractionLanguage } | null }> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/genie-insight`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ language }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  askGenie: async (
    question: string,
    conversationId?: string,
    language?: InteractionLanguage
  ): Promise<GenieResponse> => {
    const res = await fetch(`${API_BASE_URL}/genie/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId, language }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  transcribeMic: async (
    callId: string,
    audioBase64: string,
    mimeType: string,
    speaker = 1,
    language?: InteractionLanguage,
    browserCaption?: string
  ): Promise<LiveNudge> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/mic-transcribe`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        audio_b64: audioBase64,
        mime_type: mimeType,
        speaker,
        language,
        browser_caption: browserCaption,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  voiceTraces: (opts?: { limit?: number; sessionId?: string; callId?: string }) => {
    const params = new URLSearchParams();
    if (opts?.limit) params.set("limit", String(opts.limit));
    if (opts?.sessionId) params.set("session_id", opts.sessionId);
    if (opts?.callId) params.set("call_id", opts.callId);
    const query = params.toString();
    return getJSON<{ traces: TraceSummary[]; count: number }>(`/traces${query ? `?${query}` : ""}`);
  },
  voiceTraceSessions: (limit = 200) =>
    getJSON<{ sessions: TraceSessionRollup[]; count: number }>(`/traces/sessions?limit=${limit}`),
  voiceTrace: (traceId: string) => getJSON<TraceDetail>(`/traces/${encodeURIComponent(traceId)}`),
  voiceBenchmarks: () => getJSON<VoiceBenchmarksResponse>("/realtime/v1/benchmarks"),
  zhAsrComparisons: (callId?: string, limit = 10) => {
    const params = new URLSearchParams();
    if (callId) params.set("call_id", callId);
    params.set("limit", String(limit));
    const query = params.toString();
    return getJSON<{ count: number; comparisons: ZhAsrComparison[] }>(
      `/calls/zh-asr-comparisons${query ? `?${query}` : ""}`
    );
  },
};
