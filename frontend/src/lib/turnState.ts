/**
 * Shared TurnState reducer for progressive voice UI.
 *
 * One assistant bubble upsert per turn_id — later response.text / turn.event
 * patches replace the same row instead of appending duplicates.
 */

export type TurnRole = "user" | "assistant" | "system";

export interface TurnBubble {
  turnId: number;
  role: TurnRole;
  text: string;
  claims?: Array<Record<string, unknown>>;
  tools?: Array<{ name: string; result?: unknown }>;
  evidence?: Array<Record<string, unknown>>;
  events?: Array<{ seq: number; kind: string; payload: Record<string, unknown> }>;
  lastSeq?: number;
  status: "streaming" | "final" | "cancelled";
}

export interface ConversationState {
  byTurnId: Record<number, TurnBubble>;
  order: number[];
}

export type TurnAction =
  | { type: "user_transcript"; turnId: number; text: string }
  | { type: "response_text"; turnId: number; text: string }
  | { type: "tool_called"; turnId: number; name: string; result?: unknown }
  | {
      type: "turn_event";
      turnId: number;
      seq: number;
      kind: string;
      payload: Record<string, unknown>;
    }
  | {
      type: "turn_final";
      turnId: number;
      text?: string;
      claims?: Array<Record<string, unknown>>;
    }
  | { type: "turn_cancelled"; turnId: number }
  | { type: "reset" };

export function emptyConversation(): ConversationState {
  return { byTurnId: {}, order: [] };
}

function upsert(
  state: ConversationState,
  turnId: number,
  patch: Partial<TurnBubble> & { role: TurnRole }
): ConversationState {
  const prev = state.byTurnId[turnId];
  const next: TurnBubble = {
    turnId,
    role: patch.role,
    text: patch.text ?? prev?.text ?? "",
    claims: patch.claims ?? prev?.claims,
    tools: patch.tools ?? prev?.tools,
    evidence: patch.evidence ?? prev?.evidence,
    events: patch.events ?? prev?.events,
    lastSeq: patch.lastSeq ?? prev?.lastSeq,
    status: patch.status ?? prev?.status ?? "streaming",
  };
  const byTurnId = { ...state.byTurnId, [turnId]: next };
  const order = prev ? state.order : [...state.order, turnId];
  return { byTurnId, order };
}

export function turnReducer(
  state: ConversationState,
  action: TurnAction
): ConversationState {
  switch (action.type) {
    case "reset":
      return emptyConversation();
    case "user_transcript":
      return upsert(state, action.turnId, {
        role: "user",
        text: action.text,
        status: "final",
      });
    case "response_text":
      return upsert(state, action.turnId, {
        role: "assistant",
        text: action.text,
        status: "streaming",
      });
    case "tool_called": {
      const prev = state.byTurnId[action.turnId];
      const tools = [...(prev?.tools ?? []), { name: action.name, result: action.result }];
      return upsert(state, action.turnId, {
        role: "assistant",
        text: prev?.text ?? "",
        tools,
        status: "streaming",
      });
    }
    case "turn_event": {
      const prev = state.byTurnId[action.turnId];
      if (prev?.lastSeq !== undefined && action.seq <= prev.lastSeq) return state;
      const events = [
        ...(prev?.events ?? []),
        { seq: action.seq, kind: action.kind, payload: action.payload },
      ];
      const evidence =
        action.kind === "evidence.available" && action.payload.evidence
          ? [
              ...(prev?.evidence ?? []),
              action.payload.evidence as Record<string, unknown>,
            ]
          : prev?.evidence;
      // What was SPOKEN wins. `answer.final` can be display-only prose that a tool
      // turn never says out loud, so it only becomes the bubble text when the
      // runtime marks it speakable; `speech.committed` always does.
      const spoken =
        action.kind === "speech.committed" && typeof action.payload.text === "string"
          ? action.payload.text
          : undefined;
      const modelText =
        action.kind === "answer.final" &&
        typeof action.payload.text === "string" &&
        action.payload.display_only !== true
          ? action.payload.text
          : undefined;
      const text = spoken ?? modelText ?? prev?.text ?? "";
      const status: TurnBubble["status"] =
        action.kind === "turn.completed"
          ? "final"
          : action.kind === "turn.cancelled" || action.kind === "turn.failed"
            ? "cancelled"
            : prev?.status ?? "streaming";
      return upsert(state, action.turnId, {
        role: "assistant",
        text,
        evidence,
        events,
        lastSeq: action.seq,
        status,
      });
    }
    case "turn_final":
      return upsert(state, action.turnId, {
        role: "assistant",
        text: action.text,
        claims: action.claims,
        status: "final",
      });
    case "turn_cancelled":
      return upsert(state, action.turnId, {
        role: "assistant",
        status: "cancelled",
      });
    default:
      return state;
  }
}

/** Flatten for list rendering (stable order). */
export function conversationList(state: ConversationState): TurnBubble[] {
  return state.order.map((id) => state.byTurnId[id]).filter(Boolean);
}
