/**
 * Progress of a long governed workspace read, as reported by the workspace itself.
 *
 * A Genie One answer takes minutes, so the wait has to be legible: the runtime
 * converts Genie's internal trace into business-safe stages before forwarding
 * `tool.progress` / `speech.progress`, and the page renders those stages as a
 * timeline. SQL, table names, raw query results, chain-of-thought, and Databricks
 * implementation terminology never belong in this contract.
 */

export type ProgressStep = {
  label: string;
  status: "done" | "active" | "pending";
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

export function readProgressSteps(value: unknown): ProgressStep[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    const step = asRecord(entry);
    const label = typeof step?.label === "string" ? step.label.trim() : "";
    if (!label) return [];
    const status = step?.status;
    return [
      {
        label,
        status: status === "done" || status === "active" ? status : "pending",
      } satisfies ProgressStep,
    ];
  });
}

/** `m:ss` for the on-screen wait clock. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
