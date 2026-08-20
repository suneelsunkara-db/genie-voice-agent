import { describe, expect, it } from "vitest";

import { TurnCursor } from "./turnCursor";

describe("TurnCursor", () => {
  it("starts a turn and does not treat it as stale", () => {
    const c = new TurnCursor();
    c.start(3);
    expect(c.activeTurnId).toBe(3);
    expect(c.isStale(3)).toBe(false);
  });

  it("marks cancelled turns stale and keeps newer turns live", () => {
    const c = new TurnCursor();
    c.start(2);
    c.cancel(2);
    expect(c.isStale(2)).toBe(true);
    c.start(3);
    expect(c.isStale(3)).toBe(false);
    expect(c.isStale(2)).toBe(true);
  });

  it("treats older turn_ids as stale after a newer turn starts", () => {
    const c = new TurnCursor();
    c.start(5);
    expect(c.isStale(4)).toBe(true);
    expect(c.isStale(5)).toBe(false);
  });

  it("clears state", () => {
    const c = new TurnCursor();
    c.start(1);
    c.cancel(1);
    c.clear();
    expect(c.activeTurnId).toBe(0);
    expect(c.isStale(1)).toBe(false);
  });
});
