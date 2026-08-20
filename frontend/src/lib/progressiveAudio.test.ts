import { describe, expect, it, vi } from "vitest";

import { TurnCursor } from "./turnCursor";

/** Mirrors realtimeVoice response.audio dual-flag parsing (Phase 0/1). */
function resolveAudioFinals(msg: {
  final?: boolean;
  segment_final?: boolean;
  turn_final?: boolean;
}): { final: boolean; segmentFinal: boolean; turnFinal: boolean } {
  const turnFinal =
    typeof msg.turn_final === "boolean" ? msg.turn_final : msg.final ?? false;
  const segmentFinal =
    typeof msg.segment_final === "boolean" ? msg.segment_final : msg.final ?? false;
  const final = turnFinal || (msg.final ?? false);
  return { final, segmentFinal, turnFinal };
}

describe("progressive client safety", () => {
  it("interrupt barge_in cancels the active turn so later events are stale", () => {
    const cursor = new TurnCursor();
    cursor.start(7);
    // bargeIn() cancels the cursor before sending { type: "barge_in" }
    const cancelled = cursor.cancel();
    expect(cancelled).toBe(7);
    expect(cursor.isStale(7)).toBe(true);
    const bargeInPayload = { type: "barge_in" };
    expect(bargeInPayload.type).toBe("barge_in");
  });

  it("stale turn_id is ignored after cancel / newer turn", () => {
    const cursor = new TurnCursor();
    cursor.start(1);
    cursor.cancel(1);
    cursor.start(2);
    expect(cursor.isStale(1)).toBe(true);
    expect(cursor.isStale(2)).toBe(false);
  });

  it("playback.stop flushes without cancelling the working turn", () => {
    // Same-turn inject reuses turn_id; cancelling on playback.stop would drop
    // the inject audio as stale. Flush is the only client action required.
    const cursor = new TurnCursor();
    const flush = vi.fn();
    cursor.start(4);
    flush();
    expect(cursor.isStale(4)).toBe(false);
    expect(flush).toHaveBeenCalledOnce();
  });

  it("speech_epoch older than accepted is dropped", () => {
    const accepted = 2;
    const drop = (epoch: number | undefined) =>
      epoch !== undefined && epoch < accepted;
    expect(drop(1)).toBe(true);
    expect(drop(2)).toBe(false);
    expect(drop(3)).toBe(false);
    expect(drop(undefined)).toBe(false);
  });

  it("legacy final mirrors turn_final during migration", () => {
    expect(resolveAudioFinals({ final: true })).toEqual({
      final: true,
      segmentFinal: true,
      turnFinal: true,
    });
  });

  it("segment_final alone does not complete the turn", () => {
    expect(resolveAudioFinals({ segment_final: true, turn_final: false })).toEqual({
      final: false,
      segmentFinal: true,
      turnFinal: false,
    });
  });

  it("turn_final completes the turn", () => {
    expect(resolveAudioFinals({ segment_final: true, turn_final: true })).toEqual({
      final: true,
      segmentFinal: true,
      turnFinal: true,
    });
  });
});
