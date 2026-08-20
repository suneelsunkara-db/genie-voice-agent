import { describe, expect, it } from "vitest";
import {
  activeStep,
  formatElapsed,
  readProgressSteps,
  stepLabel,
} from "./workspaceProgress";

describe("readProgressSteps", () => {
  it("keeps the workspace's own order, code, and status", () => {
    expect(
      readProgressSteps([
        { code: "finding_data", label: "Finding the right data", status: "done" },
        { code: "running_analysis", label: "Running the analysis", status: "active" },
        { code: "preparing_answer", label: "Preparing your answer", status: "pending" },
      ])
    ).toEqual([
      { code: "finding_data", label: "Finding the right data", status: "done" },
      { code: "running_analysis", label: "Running the analysis", status: "active" },
      { code: "preparing_answer", label: "Preparing your answer", status: "pending" },
    ]);
  });

  it("treats an unknown status as pending rather than trusting it", () => {
    expect(readProgressSteps([{ code: "understanding", label: "Scanning", status: "weird" }])).toEqual(
      [{ code: "understanding", label: "Scanning", status: "pending" }]
    );
  });

  it("degrades to no steps when the payload is not what we expect", () => {
    // The caller still gets the spoken cadence; the panel just shows no timeline.
    expect(readProgressSteps(undefined)).toEqual([]);
    expect(readProgressSteps({ steps: [] })).toEqual([]);
    expect(readProgressSteps([{ status: "active" }, {}, "", 7])).toEqual([]);
  });

  it("drops blank rows so the timeline never renders an empty line", () => {
    expect(readProgressSteps([{ label: "   " }, { label: " Executing " }])).toEqual([
      { code: "", label: "Executing", status: "pending" },
    ]);
  });

  it("still accepts a step that carries only a code", () => {
    expect(readProgressSteps([{ code: "understanding", status: "active" }])).toEqual([
      { code: "understanding", label: "", status: "active" },
    ]);
  });
});

describe("stepLabel", () => {
  const copy = {
    progressStageRunningAnalysis: "กำลังวิเคราะห์ข้อมูล",
    progressStageUnderstanding: "กำลังทำความเข้าใจคำถามของคุณ",
  };

  it("renders the phase in the caller's language, not the runtime's English", () => {
    expect(
      stepLabel({ code: "running_analysis", label: "Running the analysis", status: "active" }, copy)
    ).toBe("กำลังวิเคราะห์ข้อมูล");
  });

  it("falls back to the runtime label for a phase the catalog has not caught up to", () => {
    expect(
      stepLabel({ code: "brand_new_phase", label: "Doing something new", status: "active" }, copy)
    ).toBe("Doing something new");
    expect(stepLabel({ code: "", label: "Unlabelled phase", status: "active" }, copy)).toBe(
      "Unlabelled phase"
    );
  });
});

describe("activeStep", () => {
  it("names the phase in flight so the heading matches the timeline", () => {
    expect(
      activeStep([
        { code: "finding_data", label: "Finding the right data", status: "done" },
        { code: "running_analysis", label: "Running the analysis", status: "active" },
        { code: "preparing_answer", label: "Preparing your answer", status: "pending" },
      ])?.code
    ).toBe("running_analysis");
  });

  it("is null when nothing is in flight, so the heading keeps its default", () => {
    expect(activeStep([])).toBeNull();
    expect(activeStep([{ code: "understanding", label: "Done", status: "done" }])).toBeNull();
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
