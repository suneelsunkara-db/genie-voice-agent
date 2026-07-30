/**
 * Pure analytics for the credit-card canvas — framework-free so it can be unit
 * tested in isolation (see cardAnalytics.test.ts). These compute the expense
 * trend, the spike-driver waterfall, and the rewards-leakage waterfall from the
 * cardholder's own fast-lane data (deterministic; no Genie/LLM involved).
 */

export type SpendingRecord = {
  spend_cat_id: string;
  customer_id: string;
  cycle: string;
  category: string;
  total_amount: number | string;
  txn_count: number;
  largest_merchant: string;
  largest_amount: number | string;
  is_new_category: boolean;
  pct_change_vs_prior: number | string | null;
};

export type RewardRecord = {
  ledger_id: string;
  cycle: string;
  category: string;
  eligible_spend: number | string;
  points_earned: number | string;
  points_possible: number | string;
  reversed_points: number | string;
  expired_points: number | string;
  missed_reason: string | null;
};

export type WaterfallStep = {
  label: string;
  delta: number;
  color: string;
  kind: "base" | "up" | "down" | "final";
};

export type CycleTotal = { cycle: string; label: string; total: number; topCat: string };

export const CATEGORY_COLORS: Record<string, string> = {
  groceries: "#34d399",
  gas: "#38bdf8",
  dining: "#f59e0b",
  travel: "#a78bfa",
  streaming: "#f472b6",
  electronics: "#2dd4bf",
  shopping: "#fb923c",
  utilities: "#818cf8",
  fees: "#f87171",
  interest: "#fca5a5",
  other: "#94a3b8",
};

export const REASON_LABELS: Record<string, string> = {
  inactive_bonus: "Bonus never activated",
  non_bonus: "Spent outside bonus categories",
  reversed: "Reversed (returns/disputes)",
  expired: "Expired points",
  capped: "Category cap reached",
};

export function num(n: unknown): number {
  const v = Number(n);
  return Number.isFinite(v) ? v : 0;
}

export function colorFor(category: string): string {
  return CATEGORY_COLORS[category?.toLowerCase?.() ?? ""] || CATEGORY_COLORS.other;
}

export function reasonLabel(reason: string | null | undefined): string {
  if (!reason) return "Not earned";
  return REASON_LABELS[reason] || reason.replace(/_/g, " ");
}

/** Total spend per cycle, ascending by cycle, with the top category for tooltips. */
export function cycleTotals(spending: SpendingRecord[]): CycleTotal[] {
  const byCycle = new Map<string, SpendingRecord[]>();
  for (const s of spending) {
    const arr = byCycle.get(s.cycle) || [];
    arr.push(s);
    byCycle.set(s.cycle, arr);
  }
  return [...byCycle.keys()]
    .sort()
    .map((cycle) => {
      const rows = byCycle.get(cycle) || [];
      const total = rows.reduce((sum, r) => sum + num(r.total_amount), 0);
      const top = [...rows].sort((a, b) => num(b.total_amount) - num(a.total_amount))[0];
      const label = cycle.length >= 7 ? cycle.slice(2, 7) : cycle; // YY-MM
      return { cycle, label, total, topCat: top?.category ?? "" };
    });
}

/**
 * Decompose the latest cycle's spend increase over the typical month into its
 * top category drivers. Reconciles by construction: the driver deltas plus a
 * residual bucket always sum from the baseline to the current total.
 */
export function computeSpikeWaterfall(
  spending: SpendingRecord[],
): { steps: WaterfallStep[]; baseline: number; current: number; increase: number } | null {
  const totals = cycleTotals(spending);
  if (totals.length < 2) return null;
  const latest = totals[totals.length - 1];
  const prior = totals.slice(0, -1);
  const baseline = prior.reduce((s, t) => s + t.total, 0) / prior.length;
  const current = latest.total;

  const latestRows = spending.filter((s) => s.cycle === latest.cycle);
  const priorCycles = new Set(prior.map((t) => t.cycle));
  const priorByCat = new Map<string, number[]>();
  for (const s of spending) {
    if (!priorCycles.has(s.cycle)) continue;
    const arr = priorByCat.get(s.category) || [];
    arr.push(num(s.total_amount));
    priorByCat.set(s.category, arr);
  }
  const deltas = latestRows
    .map((r) => {
      const priorAmts = priorByCat.get(r.category) || [];
      const priorAvg = priorAmts.length ? priorAmts.reduce((a, b) => a + b, 0) / priorAmts.length : 0;
      return { category: r.category, delta: num(r.total_amount) - priorAvg, isNew: !!r.is_new_category };
    })
    .filter((d) => d.delta > 1)
    .sort((a, b) => b.delta - a.delta);

  const topDrivers = deltas.slice(0, 4);
  const steps: WaterfallStep[] = [
    { label: "Typical month", delta: baseline, color: "#64748b", kind: "base" },
  ];
  let accounted = 0;
  for (const d of topDrivers) {
    steps.push({
      label: d.isNew ? `${d.category} (new)` : d.category,
      delta: d.delta,
      color: colorFor(d.category),
      kind: "up",
    });
    accounted += d.delta;
  }
  const residual = current - baseline - accounted;
  if (Math.abs(residual) > 1) {
    steps.push({
      label: residual >= 0 ? "Everything else" : "Offsets",
      delta: residual,
      color: residual >= 0 ? CATEGORY_COLORS.other : "#22c55e",
      kind: residual >= 0 ? "up" : "down",
    });
  }
  steps.push({ label: "This month", delta: current, color: "#f59e0b", kind: "final" });
  return { steps, baseline, current, increase: current - baseline };
}

/**
 * Decompose the rewards a cardholder qualified for into what they kept vs each
 * leakage reason. Reconciles by construction: losses sum from "possible" down
 * to "kept".
 */
export function computeRewardsWaterfall(
  rewards: RewardRecord[],
): { steps: WaterfallStep[]; possible: number; earned: number; gap: number } | null {
  if (!rewards || rewards.length === 0) return null;
  const possible = rewards.reduce((s, r) => s + num(r.points_possible), 0);
  const reversed = rewards.reduce((s, r) => s + num(r.reversed_points), 0);
  const expired = rewards.reduce((s, r) => s + num(r.expired_points), 0);

  const byReason = new Map<string, number>();
  for (const r of rewards) {
    const gap = num(r.points_possible) - num(r.points_earned);
    if (gap > 0) {
      const key = r.missed_reason || "non_bonus";
      byReason.set(key, (byReason.get(key) || 0) + gap);
    }
  }
  const losses: { label: string; pts: number }[] = [];
  for (const [reason, p] of byReason.entries()) losses.push({ label: reasonLabel(reason), pts: p });
  if (reversed > 0) losses.push({ label: reasonLabel("reversed"), pts: reversed });
  if (expired > 0) losses.push({ label: reasonLabel("expired"), pts: expired });
  losses.sort((a, b) => b.pts - a.pts);

  const totalLoss = losses.reduce((s, l) => s + l.pts, 0);
  const earned = possible - totalLoss;

  const steps: WaterfallStep[] = [
    { label: "Points you qualified for", delta: possible, color: "#a78bfa", kind: "base" },
  ];
  for (const l of losses.slice(0, 4)) {
    steps.push({ label: l.label, delta: -l.pts, color: "#f87171", kind: "down" });
  }
  steps.push({ label: "Points you kept", delta: earned, color: "#34d399", kind: "final" });
  return { steps, possible, earned, gap: totalLoss };
}
