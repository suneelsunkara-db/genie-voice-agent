/**
 * Progress of a long governed workspace read, as reported by the workspace itself.
 *
 * A Genie One answer takes minutes, so the wait has to be legible: the runtime
 * classifies Genie's internal trace into one ordered pipeline of business-safe
 * phases before forwarding `tool.progress` / `speech.progress`, and the page renders
 * that pipeline as a timeline with a single phase in flight. The order is the
 * runtime's, so render the steps as given. SQL, table names, raw query results,
 * chain-of-thought, and Databricks implementation terminology never belong here.
 */

export type ProgressStep = {
  /** Stable phase code, localized here. Empty for a payload without one. */
  code: string;
  /** The runtime's English label, used only when no translation is available. */
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
    const code = typeof step?.code === "string" ? step.code.trim() : "";
    if (!label && !code) return [];
    const status = step?.status;
    return [
      {
        code,
        label,
        status: status === "done" || status === "active" ? status : "pending",
      } satisfies ProgressStep,
    ];
  });
}

/**
 * The phase name to show, in the caller's language.
 *
 * The runtime sends a stable code; the translation lives in the UI message
 * catalog like every other piece of chrome, so the timeline is not the one part
 * of a Hindi call that stays in English. An unknown code falls back to the
 * runtime's English label rather than rendering a raw identifier.
 */
export function stepLabel(step: ProgressStep, copy: Record<string, unknown>): string {
  const key = step.code
    ? `progressStage${step.code.replace(/_+([a-z0-9])/g, (_m, c: string) => c.toUpperCase()).replace(/^./, (c) => c.toUpperCase())}`
    : "";
  const translated = key ? copy[key] : undefined;
  return typeof translated === "string" && translated ? translated : step.label;
}

/**
 * The phase currently in flight, for the heading above the timeline. Keeping the
 * heading and the timeline on the same value is what stops the screen from saying
 * one thing while the list below it highlights another.
 */
export function activeStep(steps: ProgressStep[]): ProgressStep | null {
  return steps.find((step) => step.status === "active") ?? null;
}

/** `m:ss` for the on-screen wait clock. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
