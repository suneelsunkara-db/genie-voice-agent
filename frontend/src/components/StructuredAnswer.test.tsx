import { describe, expect, it } from "vitest";
import {
  AnalysisResult,
  chartModel,
  withoutEmbeddedQueryMarkdown,
  withoutScaffoldingResults,
} from "./StructuredAnswer";

const spending: AnalysisResult = {
  itemId: "q1",
  sql: "select category, total_spend_sgd from spending",
  columns: [
    { name: "category", typeText: "STRING" },
    { name: "total_spend_sgd", typeText: "DECIMAL" },
    { name: "txn_count", typeText: "BIGINT" },
  ],
  rows: [
    ["Shopping", "73998.64", "295"],
    ["Other", "62203.32", "252"],
    ["Groceries", "48115.80", "447"],
  ],
  totalRowCount: 3,
  truncated: false,
  view: {
    kind: "bar",
    dimension: "category",
    measures: ["total_spend_sgd"],
  },
};

describe("StructuredAnswer data modeling", () => {
  it("builds a categorical bar chart from typed Genie rows", () => {
    const model = chartModel(spending);
    expect(model?.kind).toBe("bar");
    expect(model?.dimension.name).toBe("category");
    expect(model?.data[0]).toEqual({ category: "Shopping", total_spend_sgd: 73998.64 });
  });

  it("charts only the measures named by a governed view", () => {
    expect(chartModel(spending)?.measures.map((m) => m.name)).toEqual(["total_spend_sgd"]);
  });

  it("supports several measures when the governed view selects them", () => {
    const model = chartModel({
      ...spending,
      columns: [
        { name: "category", typeText: "STRING" },
        { name: "this_cycle", typeText: "DECIMAL" },
        { name: "last_cycle", typeText: "DECIMAL" },
      ],
      rows: [
        ["Shopping", "410", "380"],
        ["Dining", "260", "300"],
      ],
      view: {
        kind: "bar",
        dimension: "category",
        measures: ["this_cycle", "last_cycle"],
      },
    });
    expect(model?.measures.map((m) => m.name)).toEqual(["this_cycle", "last_cycle"]);
  });

  it("uses a line chart for a temporal dimension", () => {
    const model = chartModel({
      ...spending,
      columns: [
        { name: "month", typeText: "STRING" },
        { name: "revenue", typeText: "DOUBLE" },
      ],
      rows: [["2026-01", 10], ["2026-02", 12]],
      view: {
        kind: "line",
        dimension: "month",
        measures: ["revenue"],
      },
    });
    expect(model?.kind).toBe("line");
  });

  it("does not chart a one-row KPI result", () => {
    expect(chartModel({ ...spending, rows: [["Shopping", 73998, 295]] })).toBeNull();
  });

  it("does not guess a chart for a two-dimension scenario matrix", () => {
    expect(chartModel({
      ...spending,
      columns: [
        { name: "resource", typeText: "STRING" },
        { name: "scenario", typeText: "STRING" },
        { name: "monthly_value", typeText: "DOUBLE" },
        { name: "total_value", typeText: "DOUBLE" },
      ],
      rows: [
        ["CPU", "Base", 100, 1200],
        ["CPU", "High", 180, 1800],
        ["GPU", "Base", 10, 120],
      ],
      view: undefined,
    })).toBeNull();
  });

  it("does not guess a chart for a wide mixed-unit time result", () => {
    expect(chartModel({
      ...spending,
      columns: [
        { name: "month", typeText: "TIMESTAMP" },
        { name: "cpu_dbu", typeText: "DOUBLE" },
        { name: "gpu_dbu", typeText: "DOUBLE" },
        { name: "gpu_workspaces", typeText: "BIGINT" },
        { name: "cpu_workspaces", typeText: "BIGINT" },
      ],
      rows: [
        ["2026-01-01", 1000000, 100, 2, 40],
        ["2026-02-01", 1200000, 150, 3, 45],
      ],
      view: undefined,
    })).toBeNull();
  });

  it("renders a typed dataset as a table when no governed view exists", () => {
    expect(chartModel({ ...spending, view: undefined })).toBeNull();
  });

  it("hides the lookup Genie ran on the way to the answer", () => {
    // The regression: a turn that first asked which months exist rendered that bare
    // column as "Query result 1", above the result that actually answered.
    const months: AnalysisResult = {
      itemId: "q0",
      sql: "select distinct month from statements",
      columns: [{ name: "month", typeText: "STRING" }],
      rows: [["2025-06"], ["2025-05"], ["2025-04"]],
      totalRowCount: 3,
      truncated: false,
    };
    expect(withoutScaffoldingResults([months, spending])).toEqual([spending]);
  });

  it("keeps a single-column result that is the answer on its own", () => {
    const months: AnalysisResult = {
      itemId: "q0",
      sql: "select distinct month from statements",
      columns: [{ name: "month", typeText: "STRING" }],
      rows: [["2025-06"], ["2025-05"]],
      totalRowCount: 2,
      truncated: false,
    };
    expect(withoutScaffoldingResults([months])).toEqual([months]);
    expect(withoutScaffoldingResults([])).toEqual([]);
  });

  it("removes embedded query markdown only when typed rows exist", () => {
    const markdown = [
      "# Result",
      "<!-- begin-embedded:query_1 -->",
      "| category | spend |",
      "| --- | --- |",
      "| Shopping | 10 |",
      "<!-- end-embedded:query_1 -->",
      "Takeaway.",
    ].join("\n");

    expect(withoutEmbeddedQueryMarkdown(markdown, true)).toBe("# Result\n\nTakeaway.");
    expect(withoutEmbeddedQueryMarkdown(markdown, false)).toBe(markdown);
  });

  it("removes an unmarked markdown table when the same typed rows exist", () => {
    const markdown = [
      "## Forecast scenarios",
      "",
      "| resource | scenario | total |",
      "| --- | --- | --- |",
      "| CPU | Base | 1200 |",
      "| GPU | Base | 120 |",
      "",
      "The base case is directional.",
    ].join("\n");

    expect(withoutEmbeddedQueryMarkdown(markdown, true)).toBe(
      "## Forecast scenarios\n\nThe base case is directional."
    );
  });
});
