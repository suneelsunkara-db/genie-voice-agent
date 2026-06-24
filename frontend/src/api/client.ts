import { API_BASE_URL } from "../config";

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
    utterances?: { text: string; speaker?: number }[];
  };
}

export interface StatusResponse {
  mode: string;
  deployment?: string;
  enrichment?: { model_endpoint?: string };
  jobs?: { lakeflow?: JobState };
  stages: Stage[];
  counts: Record<string, unknown>;
  call_states: CallState[];
}

export interface LiveNudge {
  call_id: string;
  model?: string;
  live: Record<string, any>;
  transcript?: string;
  agent_reply?: string | null;
  agent_validation?: {
    genie_validated?: boolean;
    mismatches?: string[];
    genie_error?: string | null;
    genie_skipped?: boolean;
    output_validated?: boolean;
    output_issues?: string[];
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
  answer?: string;
  sql?: string;
  rows?: unknown[][];
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

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  status: () => getJSON<StatusResponse>("/status"),
  health: () => getJSON<Record<string, unknown>>("/health"),
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
    speaker?: number
  ): Promise<LiveNudge> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/assist`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, speaker }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  askGenie: async (question: string): Promise<GenieResponse> => {
    const res = await fetch(`${API_BASE_URL}/genie/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  transcribeMic: async (
    callId: string,
    audioBase64: string,
    mimeType: string,
    speaker = 1
  ): Promise<LiveNudge> => {
    const res = await fetch(`${API_BASE_URL}/calls/${callId}/mic-transcribe`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ audio_b64: audioBase64, mime_type: mimeType, speaker }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};
