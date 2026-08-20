import { describe, expect, it } from "vitest";
import { formatElapsed, readProgressSteps } from "./workspaceProgress";

describe("readProgressSteps", () => {
  it("keeps the workspace's own order and status", () => {
    expect(
      readProgressSteps([
        { label: "Generating SQL", status: "done" },
        { label: "Running the query", status: "active" },
        { label: "Formatting", status: "pending" },
      ])
    ).toEqual([
      { label: "Generating SQL", status: "done" },
      { label: "Running the query", status: "active" },
      { label: "Formatting", status: "pending" },
    ]);
  });

  it("treats an unknown status as pending rather than trusting it", () => {
    expect(readProgressSteps([{ label: "Scanning", status: "weird" }])).toEqual([
      { label: "Scanning", status: "pending" },
    ]);
  });

  it("degrades to no steps when the payload is not what we expect", () => {
    // The caller still gets the spoken cadence; the panel just shows no timeline.
    expect(readProgressSteps(undefined)).toEqual([]);
    expect(readProgressSteps({ steps: [] })).toEqual([]);
    expect(readProgressSteps([{ status: "active" }, {}, "", 7])).toEqual([]);
  });

  it("drops blank labels so the timeline never renders an empty row", () => {
    expect(readProgressSteps([{ label: "   " }, { label: " Executing " }])).toEqual([
      { label: "Executing", status: "pending" },
    ]);
  });
});

describe("formatElapsed", () => {
  it("reads as a wait clock", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(9_400)).toBe("0:09");
    expect(formatElapsed(65_000)).toBe("1:05");
    expect(formatElapsed(194_000)).toBe("3:14");
  });

  it("never shows a negative clock if the timer starts skewed", () => {
    expect(formatElapsed(-5_000)).toBe("0:00");
  });
});
