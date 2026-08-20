/**
 * Client-side turn cursor for progressive / barge-in safety.
 *
 * Tracks the active turn and a cancelled set so stale server events
 * (response.text/audio, tool.called) after barge-in or supersession are dropped
 * before they reach playback or UI reducers.
 */
export class TurnCursor {
  activeTurnId = 0;
  private cancelled = new Set<number>();

  /** Mark a turn as the live generation (from turn.started / speech.started). */
  start(turnId: number): void {
    if (typeof turnId !== "number" || Number.isNaN(turnId)) return;
    this.activeTurnId = turnId;
    // An amendment deliberately restarts the same logical id after playback was
    // stopped. Server event seq and speech_epoch still protect against old work.
    this.cancelled.delete(turnId);
  }

  /** Cancel the current (or explicit) turn and bump past it. */
  cancel(turnId?: number): number {
    const id = turnId ?? this.activeTurnId;
    if (id > 0) this.cancelled.add(id);
    // Keep accepting only strictly newer turns.
    if (id >= this.activeTurnId) this.activeTurnId = id;
    return id;
  }

  /** True when this turn_id should be ignored. */
  isStale(turnId: unknown): boolean {
    if (typeof turnId !== "number" || Number.isNaN(turnId)) return false;
    if (this.cancelled.has(turnId)) return true;
    // Older than the live cursor after a newer turn started.
    return this.activeTurnId > 0 && turnId < this.activeTurnId;
  }

  clear(): void {
    this.activeTurnId = 0;
    this.cancelled.clear();
  }
}
