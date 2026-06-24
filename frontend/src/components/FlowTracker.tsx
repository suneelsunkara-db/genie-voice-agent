import { JobState, Stage } from "../api/client";

const MODE_LABEL: Record<string, string> = {
  "real-time": "Real-time",
  streaming: "Streaming",
  batch: "Batch",
  landing: "Storage",
  "managed sync": "Managed Sync",
};

const MODE_ORDER = ["real-time", "streaming", "batch", "landing", "managed sync"] as const;

function modeClass(mode?: string): string {
  return mode ? `m-${mode.replace("real-time", "realtime").replace(/\s+/g, "-")}` : "";
}

const STATUS_LABEL: Record<string, string> = {
  running: "running",
  done: "complete",
  idle: "idle",
};

function StatusDot({ status }: { status?: string }) {
  if (!status) return null;
  return (
    <span className={`status-dot s-${status}`} title={STATUS_LABEL[status] ?? status} />
  );
}

function JobBanner({
  jobs,
}: {
  jobs?: { lakeflow?: JobState };
}) {
  if (!jobs) return null;
  const items: { key: string; label: string; job?: JobState }[] = [
    { key: "lakeflow", label: "Lakeflow refresh", job: jobs.lakeflow },
  ];
  return (
    <div className="job-banner">
      {items.map(({ key, label, job }) => {
        const state = !job?.available
          ? "unknown"
          : !job?.deployed
          ? "not deployed"
          : job?.running
          ? "running now"
          : job?.last_result
          ? `idle · last run ${job.last_result.toLowerCase()}`
          : "idle";
        const cls = job?.running ? "s-running" : !job?.available ? "s-idle" : "s-done";
        return (
          <span className="job-pill" key={key}>
            <span className={`status-dot ${cls}`} />
            <strong>{label}</strong>
            <span className="muted">{state}</span>
          </span>
        );
      })}
    </div>
  );
}

export function FlowTracker({
  stages,
  jobs,
}: {
  stages: Stage[];
  jobs?: { lakeflow?: JobState };
}) {
  return (
    <div>
      <JobBanner jobs={jobs} />

      <div className="flow-legend">
        {MODE_ORDER.map((m) => (
          <span className="legend-item" key={m}>
            <span className={`legend-dot ${modeClass(m)}`} />
            {MODE_LABEL[m]}
          </span>
        ))}
        <span className="legend-sep">|</span>
        <span className="legend-item">
          <span className="status-dot s-running" /> running
        </span>
        <span className="legend-item">
          <span className="status-dot s-done" /> complete
        </span>
        <span className="legend-item">
          <span className="status-dot s-idle" /> idle
        </span>
      </div>

      <div className="flow">
        {stages.map((s, i) => (
          <div className="flow-item" key={s.key}>
            <div className={`stage ${modeClass(s.mode)}`}>
              <div className="stage-head">
                <span className="stage-key">{s.key}</span>
                <StatusDot status={s.status} />
              </div>
              <div className="stage-label">{s.label}</div>
              {s.mode && (
                <div className="stage-tags">
                  <span className={`mode-chip ${modeClass(s.mode)}`}>
                    {MODE_LABEL[s.mode] ?? s.mode}
                  </span>
                  {s.latency && <span className="stage-latency">{s.latency}</span>}
                </div>
              )}
              {typeof s.count === "number" && (
                <div className="stage-count">{s.count.toLocaleString()} rows</div>
              )}
              {s.where && <div className="stage-meta">@ {s.where}</div>}
              {s.provider && <div className="stage-meta">provider: {s.provider}</div>}
              {s.model && <div className="stage-meta">model: {s.model}</div>}
            </div>
            {i < stages.length - 1 && <div className="arrow">→</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
