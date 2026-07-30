import { describe, expect, it } from "vitest";
import {
  computeRewardsWaterfall,
  computeSpikeWaterfall,
  cycleTotals,
  num,
  RewardRecord,
  SpendingRecord,
} from "./cardAnalytics";

function spend(cycle: string, category: string, total: number, isNew = false): SpendingRecord {
  return {
    spend_cat_id: `${cycle}-${category}`,
    customer_id: "CH-0001",
    cycle,
    category,
    total_amount: total,
    txn_count: 3,
    largest_merchant: "M",
    largest_amount: total / 2,
    is_new_category: isNew,
    pct_change_vs_prior: null,
  };
}

// Two typical prior months (~950 total) + a spike month (~3400) driven by travel.
const SPENDING: SpendingRecord[] = [
  spend("2025-11", "groceries", 500),
  spend("2025-11", "gas", 450),
  spend("2025-12", "groceries", 520),
  spend("2025-12", "gas", 430),
  spend("2026-01", "groceries", 520),
  spend("2026-01", "gas", 430),
  spend("2026-01", "travel", 2450, true), // new spike driver
];

function reward(
  cycle: string,
  earned: number,
  possible: number,
  reason: string | null,
  reversed = 0,
  expired = 0,
): RewardRecord {
  return {
    ledger_id: `${cycle}-${reason}`,
    cycle,
    category: "travel",
    eligible_spend: 1000,
    points_earned: earned,
    points_possible: possible,
    reversed_points: reversed,
    expired_points: expired,
    missed_reason: reason,
  };
}

describe("num", () => {
  it("coerces strings and guards non-finite", () => {
    expect(num("12.5")).toBe(12.5);
    expect(num(null)).toBe(0);
    expect(num("abc")).toBe(0);
  });
});

describe("cycleTotals", () => {
  it("totals per cycle, ascending, with a top category", () => {
    const totals = cycleTotals(SPENDING);
    expect(totals.map((t) => t.cycle)).toEqual(["2025-11", "2025-12", "2026-01"]);
    expect(totals[0].total).toBe(950);
    expect(totals[2].total).toBe(3400);
    expect(totals[2].topCat).toBe("travel");
    expect(totals[0].label).toBe("25-11"); // YY-MM
  });
});

describe("computeSpikeWaterfall", () => {
  it("reconciles: baseline + driver deltas === current total", () => {
    const w = computeSpikeWaterfall(SPENDING)!;
    expect(w).not.toBeNull();
    expect(w.baseline).toBeCloseTo(950, 5); // avg of the two prior months
    expect(w.current).toBe(3400);
    expect(w.increase).toBeCloseTo(2450, 5);

    // First step is the baseline; last step is the final "this month" total.
    expect(w.steps[0].kind).toBe("base");
    expect(w.steps[0].delta).toBeCloseTo(950, 5);
    const final = w.steps[w.steps.length - 1];
    expect(final.kind).toBe("final");
    expect(final.delta).toBe(3400);

    // The middle up/down deltas must bridge baseline → current exactly.
    const middle = w.steps.slice(1, -1).reduce((s, x) => s + x.delta, 0);
    expect(w.baseline + middle).toBeCloseTo(w.current, 5);

    // Travel is the dominant driver and flagged as new.
    const travel = w.steps.find((s) => s.label.startsWith("travel"));
    expect(travel).toBeTruthy();
    expect(travel!.label).toContain("new");
  });

  it("returns null with <2 cycles", () => {
    expect(computeSpikeWaterfall([spend("2026-01", "gas", 100)])).toBeNull();
  });
});

describe("computeRewardsWaterfall", () => {
  it("reconciles: possible - losses === kept", () => {
    const rewards: RewardRecord[] = [
      reward("2026-01", 200, 4000, "inactive_bonus"), // 3800 unmet
      reward("2026-01", 300, 300, null, 50, 25), // reversed 50 + expired 25
    ];
    const w = computeRewardsWaterfall(rewards)!;
    expect(w.possible).toBe(4300);
    // total loss = unmet(3800) + reversed(50) + expired(25) = 3875
    expect(w.gap).toBe(3875);
    expect(w.earned).toBe(4300 - 3875);

    expect(w.steps[0].kind).toBe("base");
    expect(w.steps[0].delta).toBe(4300);
    const final = w.steps[w.steps.length - 1];
    expect(final.kind).toBe("final");
    expect(final.delta).toBe(w.earned);

    // Every middle step is a loss (negative delta) and they sum to -gap.
    const losses = w.steps.slice(1, -1);
    expect(losses.every((s) => s.delta < 0)).toBe(true);
    const lossSum = losses.reduce((s, x) => s + x.delta, 0);
    expect(-lossSum).toBe(w.gap);
  });

  it("returns null with no rewards", () => {
    expect(computeRewardsWaterfall([])).toBeNull();
  });
});
