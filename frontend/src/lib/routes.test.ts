import { describe, expect, it } from "vitest";
import { APP_HASHES, resolveAppHash } from "./routes";

describe("application routes", () => {
  it("covers every user and operator page", () => {
    expect(APP_HASHES).toEqual(new Set([
      "#/",
      "#/card",
      "#/knowledge",
      "#/setup",
      "#/traces",
      "#/guardrails",
      "#/voice-benchmarks",
      "#/asr-benchmark",
      "#/telco",
    ]));
  });

  it("does not route an unknown hash to telco", () => {
    expect(resolveAppHash("#/does-not-exist")).toBe("#/");
    expect(resolveAppHash("#/traces?trace=abc")).toBe("#/traces");
  });
});
