import { describe, expect, it } from "vitest";
import {
  AnalysisResult,
  chartModel,
  withoutEmbeddedQueryMarkdown,
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
    expect(model?.measures.map((measure) => measure.name)).toEqual([
      "total_spend_sgd",
      "txn_count",
    ]);
    expect(model?.data[0]).toEqual({
      category: "Shopping",
      total_spend_sgd: 73998.64,
      txn_count: 295,
    });
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
