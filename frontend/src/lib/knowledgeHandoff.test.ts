import { beforeEach, describe, expect, it } from "vitest";
import {
  consumeKnowledgeHandoff,
  peekKnowledgeHandoff,
  storeKnowledgeHandoff,
} from "./knowledgeHandoff";

describe("Knowledge query handoff", () => {
  beforeEach(() => {
    consumeKnowledgeHandoff();
  });

  it("preserves the exact query without consuming it during a peek", () => {
    storeKnowledgeHandoff("  Analyze governed usage  ", "en-US");
    expect(peekKnowledgeHandoff()?.query).toBe("Analyze governed usage");
    expect(consumeKnowledgeHandoff()?.language).toBe("en-US");
    expect(consumeKnowledgeHandoff()).toBeNull();
  });

  it("does not store an empty query", () => {
    storeKnowledgeHandoff("   ", "en-US");
    expect(peekKnowledgeHandoff()).toBeNull();
  });
});
