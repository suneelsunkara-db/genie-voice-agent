import { beforeEach, describe, expect, it, vi } from "vitest";
import { getAppVoice, setAppVoice } from "./appVoice";

describe("app-wide voice preference", () => {
  const values = new Map<string, string>();
  const dispatchEvent = vi.fn();

  beforeEach(() => {
    values.clear();
    dispatchEvent.mockClear();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    });
    vi.stubGlobal("window", { dispatchEvent });
    vi.stubGlobal(
      "CustomEvent",
      class {
        constructor(
          public type: string,
          public init: { detail: string }
        ) {}
      }
    );
  });

  it("defaults to the female variant", () => {
    expect(getAppVoice()).toBe("female");
  });

  it("persists one choice for every page to read", () => {
    setAppVoice("male");
    expect(getAppVoice()).toBe("male");
    expect(dispatchEvent).toHaveBeenCalledOnce();
  });

  it("ignores an invalid stored value", () => {
    values.set("genie.appVoice", "../../voice.wav");
    expect(getAppVoice()).toBe("female");
  });
});
