import { describe, expect, it } from "vitest";

import { readClarification } from "./KnowledgeAgentPage";

describe("Knowledge clarification choices", () => {
  it("reads the typed prompt and selectable choices from the runtime event", () => {
    expect(
      readClarification({
        prompt: "How would you like to proceed?",
        choices: [
          "Use customer allocation",
          "Forecast account-wide usage",
          "Forecast tagged usage",
        ],
      })
    ).toEqual({
      prompt: "How would you like to proceed?",
      choices: [
        "Use customer allocation",
        "Forecast account-wide usage",
        "Forecast tagged usage",
      ],
    });
  });

  it("rejects an event without a caller-facing prompt", () => {
    expect(readClarification({ choices: ["Forecast account-wide usage"] })).toBeNull();
  });
});
