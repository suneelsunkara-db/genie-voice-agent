import { describe, expect, it } from "vitest";

import {
  conversationList,
  emptyConversation,
  turnReducer,
} from "./turnState";

describe("turnReducer", () => {
  it("upserts one assistant bubble per turn_id", () => {
    let s = emptyConversation();
    s = turnReducer(s, { type: "response_text", turnId: 2, text: "Hello" });
    s = turnReducer(s, { type: "response_text", turnId: 2, text: "Hello world" });
    const list = conversationList(s);
    expect(list).toHaveLength(1);
    expect(list[0].text).toBe("Hello world");
    expect(list[0].status).toBe("streaming");
  });

  it("stores committed claims on turn_final", () => {
    let s = emptyConversation();
    s = turnReducer(s, { type: "response_text", turnId: 1, text: "Balance is $42" });
    s = turnReducer(s, {
      type: "turn_final",
      turnId: 1,
      claims: [{ text: "Balance is $42", cite: "row:0" }],
    });
    expect(s.byTurnId[1].status).toBe("final");
    expect(s.byTurnId[1].claims).toHaveLength(1);
  });

  it("keeps user and assistant turns distinct", () => {
    let s = emptyConversation();
    s = turnReducer(s, { type: "user_transcript", turnId: 1, text: "Hi" });
    s = turnReducer(s, { type: "response_text", turnId: 2, text: "Hello" });
    expect(conversationList(s).map((b) => b.role)).toEqual(["user", "assistant"]);
  });

  it("drops duplicate or regressing AgentEvent sequences", () => {
    let s = emptyConversation();
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 7,
      seq: 3,
      kind: "action.started",
      payload: { name: "lookup" },
    });
    const accepted = s;
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 7,
      seq: 2,
      kind: "answer.final",
      payload: { text: "stale" },
    });
    expect(s).toBe(accepted);
    expect(s.byTurnId[7].events).toHaveLength(1);
  });

  it("stores structured evidence separately from display prose", () => {
    let s = emptyConversation();
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 4,
      seq: 1,
      kind: "evidence.available",
      payload: { evidence: { source: "tool:lookup", table: { rows: [[42]] } } },
    });
    expect(s.byTurnId[4].evidence?.[0].source).toBe("tool:lookup");
    expect(s.byTurnId[4].text).toBe("");
  });

  it("shows the committed speech, not display-only model prose", () => {
    let s = emptyConversation();
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 9,
      seq: 1,
      kind: "answer.final",
      payload: { text: "prose the voice never says", display_only: true },
    });
    expect(s.byTurnId[9].text).toBe("");
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 9,
      seq: 2,
      kind: "speech.committed",
      payload: { text: "orders: 1042", basis: "evidence" },
    });
    expect(s.byTurnId[9].text).toBe("orders: 1042");
  });

  it("keeps model prose when the runtime says it is speakable", () => {
    let s = emptyConversation();
    s = turnReducer(s, {
      type: "turn_event",
      turnId: 10,
      seq: 1,
      kind: "answer.final",
      payload: { text: "sure, switching to Spanish", display_only: false },
    });
    expect(s.byTurnId[10].text).toBe("sure, switching to Spanish");
  });
});
