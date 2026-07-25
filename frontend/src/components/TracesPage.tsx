import { useEffect, useMemo, useState } from "react";
import { api, TraceDetail, TraceSpan, TraceSummary } from "../api/client";
import "../styles/traces.css";

type StatusFilter = "all" | "ok" | "issues";

function statusClass(status?: string): string {
  if (!status || status === "ok") return "ok";
  if (status === "error") return "err";
  return "warn";
}

function ms(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
}

function shortId(id?: string | null): string {
  if (!id) return "—";
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

function StatusBadge({ status }: { status?: string }) {
  return <span className={`tv-badge ${statusClass(status)}`}>{status || "ok"}</span>;
}

function KindTag({ kind }: { kind: string }) {
  return <span className={`tv-kind ${kind}`}>{kind}</span>;
}

/** Pretty JSON block for arbitrary span input/output. */
function JsonView({ value }: { value: unknown }) {
  const text = useMemo(() => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }, [value]);
  return <pre className="tv-json">{text}</pre>;
}

type ChatMessage = {
  role?: string;
  content?: unknown;
  tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>;
  tool_call_id?: string;
};

/** Render an LLM `messages` array as a chat transcript (what the model saw). */
function MessagesView({ messages }: { messages: ChatMessage[] }) {
  return (
    <div>
      {messages.map((m, i) => {
        const role = String(m.role || "?");
        const content =
          typeof m.content === "string" ? m.content : m.content ? JSON.stringify(m.content, null, 2) : "";
        return (
          <div key={i} className="tv-msg">
            <div className="tv-msg-head">
              <span className={`tv-msg-role ${role}`}>{role}</span>
              {m.tool_call_id && <span style={{ color: "var(--tv-muted)" }}>↳ {shortId(m.tool_call_id)}</span>}
            </div>
            {content && <div className="tv-msg-body">{content}</div>}
            {m.tool_calls && m.tool_calls.length > 0 && (
              <div className="tv-msg-toolcalls">
                {m.tool_calls.map((c, j) => (
                  <div key={j}>
                    → {c.function?.name}({c.function?.arguments})
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function isMessagesInput(input: unknown): input is { messages: ChatMessage[] } {
  return Boolean(
    input && typeof input === "object" && Array.isArray((input as { messages?: unknown }).messages)
  );
}

function SpanDetail({ span }: { span: TraceSpan }) {
  const input = span.input;
  return (
    <div className="tv-span-detail">
      <div className="tv-section-title">
        <KindTag kind={span.kind} /> {span.name} · {ms(span.duration_ms)}
      </div>
      <div className="tv-io">
        <div className="tv-io-col">
          <h4>Input</h4>
          {span.kind === "LLM" && isMessagesInput(input) ? (
            <>
              <MessagesView messages={input.messages} />
              {"tools_available" in (input as object) && (
                <div className="tv-msg-toolcalls" style={{ border: "1px solid var(--tv-border)", borderRadius: 8 }}>
                  tools available: {JSON.stringify((input as { tools_available?: unknown }).tools_available)}
                </div>
              )}
            </>
          ) : span.kind === "GUARD" && isMessagesInput(input) ? (
            <MessagesView messages={input.messages} />
          ) : (
            <JsonView value={input} />
          )}
        </div>
        <div className="tv-io-col">
          <h4>Output</h4>
          <JsonView value={span.output} />
          {span.attributes && Object.keys(span.attributes).length > 0 && (
            <>
              <h4 style={{ marginTop: 10 }}>Attributes</h4>
              <JsonView value={span.attributes} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Waterfall({
  spans,
  totalMs,
  selected,
  onSelect,
}: {
  spans: TraceSpan[];
  totalMs: number;
  selected: number;
  onSelect: (i: number) => void;
}) {
  const scale = totalMs > 0 ? totalMs : 1;
  return (
    <div className="tv-waterfall">
      {spans.map((s, i) => {
        const left = ((s.start_ms ?? 0) / scale) * 100;
        const width = Math.max(((s.duration_ms ?? 0) / scale) * 100, 0.6);
        return (
          <div
            key={i}
            className={`tv-span-row${i === selected ? " is-selected" : ""}`}
            onClick={() => onSelect(i)}
          >
            <div className="tv-span-name">
              <KindTag kind={s.kind} />
              <span className="n">{s.name}</span>
            </div>
            <div className="tv-bar-track">
              <div
                className={`tv-bar${s.status === "error" ? " err" : ""}`}
                style={{ left: `${left}%`, width: `${width}%` }}
              />
            </div>
            <div className="tv-span-dur">{ms(s.duration_ms)}</div>
          </div>
        );
      })}
    </div>
  );
}

function TraceDetailView({ trace }: { trace: TraceDetail }) {
  const [selectedSpan, setSelectedSpan] = useState(0);
  useEffect(() => setSelectedSpan(0), [trace.trace_id]);
  const totalMs = useMemo(
    () => Math.max(trace.total_ms ?? 0, ...trace.spans.map((s) => s.end_ms ?? 0), 1),
    [trace]
  );
  const applied = trace.apply_billing_action_called;
  return (
    <div>
      <div className="tv-meta-grid">
        <div className="tv-meta">
          <div className="tv-meta-label">Status</div>
          <div className="tv-meta-value"><StatusBadge status={trace.status} /></div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">Turn</div>
          <div className="tv-meta-value">#{trace.turn_id}</div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">Language</div>
          <div className="tv-meta-value">
            {trace.language || "—"}
            {trace.detected_language && trace.detected_language !== trace.language
              ? ` (heard ${trace.detected_language})`
              : ""}
          </div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">Total latency</div>
          <div className="tv-meta-value">{ms(trace.total_ms)}</div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">LLM iterations</div>
          <div className="tv-meta-value">{trace.llm_iterations ?? 0}</div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">lookup_account calls</div>
          <div className="tv-meta-value">{trace.lookup_account_count ?? 0}</div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">Billing action</div>
          <div className="tv-meta-value">
            <span className={`tv-badge ${applied ? "apply" : "noapply"}`}>
              {applied ? "applied" : "not applied"}
            </span>
          </div>
        </div>
        <div className="tv-meta">
          <div className="tv-meta-label">Call / Customer</div>
          <div className="tv-meta-value">{trace.call_id || "—"} · {trace.customer_id || "—"}</div>
        </div>
      </div>

      <div className="tv-meta" style={{ marginBottom: 12 }}>
        <div className="tv-meta-label">Caller said (STT)</div>
        <div className="tv-meta-value">{trace.input_transcript || "—"}</div>
        <div className="tv-meta-label" style={{ marginTop: 8 }}>Agent replied</div>
        <div className="tv-meta-value">{trace.output_text || "—"}</div>
      </div>

      {trace.error && <div className="tv-callout">error: {trace.error}</div>}

      <div className="tv-section-title">Span waterfall</div>
      <Waterfall
        spans={trace.spans}
        totalMs={totalMs}
        selected={selectedSpan}
        onSelect={setSelectedSpan}
      />
      {trace.spans[selectedSpan] && <SpanDetail span={trace.spans[selectedSpan]} />}
    </div>
  );
}

interface SessionGroup {
  sessionId: string;
  callId?: string | null;
  turns: TraceSummary[];
  applied: boolean;
  lookupTotal: number;
  languages: string[];
}

export function TracesPage() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");

  const load = () => {
    setLoading(true);
    api
      .voiceTraces({ limit: 300 })
      .then((r) => {
        setTraces(r.traces ?? []);
        setErr(null);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : "failed"))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let active = true;
    api
      .voiceTrace(selectedId)
      .then((d) => active && setDetail(d))
      .catch(() => active && setDetail(null));
    return () => {
      active = false;
    };
  }, [selectedId]);

  const filtered = useMemo(() => {
    if (statusFilter === "all") return traces;
    if (statusFilter === "ok") return traces.filter((t) => (t.status || "ok") === "ok");
    return traces.filter((t) => (t.status || "ok") !== "ok");
  }, [traces, statusFilter]);

  const sessions = useMemo<SessionGroup[]>(() => {
    const map = new Map<string, SessionGroup>();
    for (const t of filtered) {
      const sid = t.session_id || "(none)";
      const g =
        map.get(sid) ??
        map
          .set(sid, {
            sessionId: sid,
            callId: t.call_id,
            turns: [],
            applied: false,
            lookupTotal: 0,
            languages: [],
          })
          .get(sid)!;
      g.turns.push(t);
      g.applied = g.applied || Boolean(t.apply_billing_action_called);
      g.lookupTotal += t.lookup_account_count ?? 0;
      if (t.language && !g.languages.includes(t.language)) g.languages.push(t.language);
    }
    const groups = Array.from(map.values());
    for (const g of groups) g.turns.sort((a, b) => (a.turn_id ?? 0) - (b.turn_id ?? 0));
    return groups;
  }, [filtered]);

  return (
    <div className="tv-root">
      <div className="tv-header">
        <div className="tv-title">
          <span>Genie</span> Voice · Trace Explorer
        </div>
        <div className="tv-header-spacer" />
        <div className="tv-filters">
          {(["all", "ok", "issues"] as StatusFilter[]).map((f) => (
            <button
              key={f}
              className={`tv-chip-btn${statusFilter === f ? " is-active" : ""}`}
              onClick={() => setStatusFilter(f)}
            >
              {f}
            </button>
          ))}
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/")}>
            ← Cockpit
          </button>
          <button className="tv-btn" onClick={load}>
            Refresh
          </button>
        </div>
      </div>

      <div className="tv-body">
        <div className="tv-list">
          {loading && <div className="tv-empty">Loading traces…</div>}
          {err && <div className="tv-empty">Error: {err}</div>}
          {!loading && !err && sessions.length === 0 && (
            <div className="tv-empty">
              No traces yet. Start a voice call in the Cockpit — each turn is captured here.
            </div>
          )}
          {sessions.map((g) => (
            <div className="tv-session" key={g.sessionId}>
              <div className="tv-session-head">
                <span className="tv-session-id">◈ {shortId(g.sessionId)}</span>
                <div className="tv-session-meta">
                  {g.languages.map((l) => (
                    <span key={l} className="tv-badge">{l}</span>
                  ))}
                  <span className={`tv-badge ${g.applied ? "apply" : "noapply"}`}>
                    {g.applied ? "waiver applied" : `no apply · ${g.lookupTotal} lookups`}
                  </span>
                  <span className="tv-badge">{g.turns.length} turns</span>
                </div>
              </div>
              {g.turns.map((t) => (
                <div
                  key={t.trace_id}
                  className={`tv-turn${selectedId === t.trace_id ? " is-selected" : ""}`}
                  onClick={() => setSelectedId(t.trace_id)}
                >
                  <div className="tv-turn-idx">#{t.turn_id}</div>
                  <div className="tv-turn-main">
                    <div className="tv-turn-text">{t.input_transcript || <em>(no transcript)</em>}</div>
                    <div className="tv-turn-sub">
                      <StatusBadge status={t.status} />
                      {(t.tool_names ?? []).map((n, i) => (
                        <span key={i} className="tv-kind TOOL">{n}</span>
                      ))}
                    </div>
                  </div>
                  <div className="tv-turn-right">{ms(t.total_ms)}</div>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div className="tv-detail">
          {detail ? (
            <TraceDetailView trace={detail} />
          ) : (
            <div className="tv-empty">Select a turn to inspect its end-to-end trace.</div>
          )}
        </div>
      </div>
    </div>
  );
}
