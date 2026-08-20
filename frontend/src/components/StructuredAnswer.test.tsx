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
};

describe("StructuredAnswer data modeling", () => {
  it("builds a categorical bar chart from typed Genie rows", () => {
    const model = chartModel(spending);
    expect(model?.kind).toBe("bar");
    expect(model?.dimension.name).toBe("category");
    expect(model?.data[0]).toEqual({ category: "Shopping", total_spend_sgd: 73998.64 });
  });

  it("charts only measures that can share one axis", () => {
    // Spend is ~250x the transaction count. Plotting both scales the axis to spend
    // and flattens the count onto the baseline, which reads as a broken chart.
    expect(chartModel(spending)?.measures.map((m) => m.name)).toEqual(["total_spend_sgd"]);
  });

  it("keeps several measures when they are of comparable size", () => {
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
    });
    expect(model?.kind).toBe("line");
  });

  it("does not chart a one-row KPI result", () => {
    expect(chartModel({ ...spending, rows: [["Shopping", 73998, 295]] })).toBeNull();
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
});
