import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { InteractionLanguage } from "../api/client";
import { UiCopy, uiCopy } from "../i18n";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type AnalysisColumn = {
  name: string;
  typeText: string;
};

export type AnalysisResult = {
  itemId: string;
  sql: string | null;
  columns: AnalysisColumn[];
  rows: unknown[][];
  totalRowCount: number;
  truncated: boolean;
  view?: {
    kind: "bar" | "line";
    dimension: string;
    measures: string[];
    title?: string;
  };
};

const CHART_COLORS = ["#4ea8ff", "#8b7cff", "#31d0aa"];

function displayName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const normalized = value.replace(/[,$%\s]/g, "");
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Format a governed cell for display in the caller's locale.
 *
 * Grouping and decimal separators follow the language of the call rather than
 * whatever locale the browser happens to run in — a Genie figure shown to a German
 * caller should read 1.234,56. The VALUE is never changed: no rounding beyond two
 * decimals, no currency conversion, no unit inference. Digits stay Latin because
 * these are financial figures that also appear in the report, statements, and the
 * SQL beside them, and they have to match.
 */
function formatValue(value: unknown, language: string): string {
  if (value === null || value === undefined) return "—";
  const numeric = numberValue(value);
  if (numeric !== null && typeof value === "number") {
    try {
      return new Intl.NumberFormat(language || undefined, {
        maximumFractionDigits: 2,
        numberingSystem: "latn",
      }).format(numeric);
    } catch {
      return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(numeric);
    }
  }
  return String(value);
}

export function chartModel(result: AnalysisResult): {
  data: Record<string, string | number>[];
  dimension: AnalysisColumn;
  measures: AnalysisColumn[];
  kind: "bar" | "line";
} | null {
  // A table's shape is evidence, not visualization intent. Render a chart only
  // from an upstream validated view; never infer dimensions, measures, units, or
  // forecast semantics from column names and parseable values.
  const view = result.view;
  if (!view) return null;
  if (result.rows.length < 2 || result.rows.length > 40) return null;
  const dimension = result.columns.find((column) => column.name === view.dimension);
  if (!dimension) return null;
  const dimensionIndex = result.columns.indexOf(dimension);
  const measures = view.measures
    .map((name) => result.columns.find((column) => column.name === name))
    .filter((column): column is AnalysisColumn => Boolean(column))
    .slice(0, 3);
  if (!measures.length) return null;
  if (measures.some((measure) => {
    const index = result.columns.indexOf(measure);
    return !result.rows.every((row) => numberValue(row[index]) !== null);
  })) return null;
  const data = result.rows.map((row, rowIndex) => {
    const point: Record<string, string | number> = {
      [dimension.name]: String(row[dimensionIndex] ?? rowIndex + 1),
    };
    for (const measure of measures) {
      const value = numberValue(row[result.columns.indexOf(measure)]);
      if (value !== null) point[measure.name] = value;
    }
    return point;
  });
  return {
    data,
    dimension,
    measures,
    kind: view.kind,
  };
}

function AnalysisChart({ result, copy }: { result: AnalysisResult; copy: UiCopy }) {
  const model = chartModel(result);
  if (!model) return null;
  const common = (
    <>
      <CartesianGrid stroke="rgba(150, 160, 200, 0.14)" vertical={false} />
      <XAxis
        dataKey={model.dimension.name}
        tick={{ fill: "#9eabc9", fontSize: 11 }}
        axisLine={false}
        tickLine={false}
        label={{
          value: displayName(model.dimension.name),
          position: "insideBottom",
          offset: -4,
          fill: "#9eabc9",
          fontSize: 10,
        }}
      />
      <YAxis
        tick={{ fill: "#9eabc9", fontSize: 11 }}
        axisLine={false}
        tickLine={false}
        width={58}
        label={{
          value: model.measures.map((measure) => displayName(measure.name)).join(" / "),
          angle: -90,
          position: "insideLeft",
          fill: "#9eabc9",
          fontSize: 10,
        }}
      />
      <Tooltip
        contentStyle={{
          background: "#13172b",
          border: "1px solid rgba(120, 140, 200, 0.3)",
          borderRadius: 10,
        }}
      />
      {model.measures.length > 1 && <Legend />}
    </>
  );
  return (
    <div className="kb-analysis-chart" aria-label={copy.saChartAria}>
      <div className="kb-analysis-chart-title">
        {result.view?.title ||
          `${displayName(model.dimension.name)} · ${model.measures
            .map((measure) => displayName(measure.name))
            .join(", ")}`}
      </div>
      <ResponsiveContainer width="100%" height={280}>
        {model.kind === "line" ? (
          <LineChart data={model.data} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
            {common}
            {model.measures.map((measure, index) => (
              <Line
                key={measure.name}
                type="monotone"
                dataKey={measure.name}
                name={displayName(measure.name)}
                stroke={CHART_COLORS[index]}
                strokeWidth={2.5}
                dot={{ r: 3 }}
              />
            ))}
          </LineChart>
        ) : (
          <BarChart data={model.data} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
            {common}
            {model.measures.map((measure, index) => (
              <Bar
                key={measure.name}
                dataKey={measure.name}
                name={displayName(measure.name)}
                fill={CHART_COLORS[index]}
                radius={[5, 5, 0, 0]}
              />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function ResultTable({
  result,
  copy,
  language,
}: {
  result: AnalysisResult;
  copy: UiCopy;
  language: string;
}) {
  if (result.rows.length === 1) {
    return (
      <div className="kb-kpis">
        {result.columns.map((column, index) => (
          <div className="kb-kpi" key={column.name}>
            <span>{displayName(column.name)}</span>
            <strong>{formatValue(result.rows[0][index], language)}</strong>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className="kb-result-table-wrap">
      <table className="kb-result-table">
        <thead>
          <tr>
            {result.columns.map((column) => (
              <th key={column.name}>{displayName(column.name)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {result.columns.map((column, columnIndex) => (
                <td key={column.name}>{formatValue(row[columnIndex], language)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {(result.truncated || result.totalRowCount > result.rows.length) && (
        <div className="kb-result-note">
          {copy.saShowingRows(String(result.rows.length), String(result.totalRowCount))}
        </div>
      )}
    </div>
  );
}

/**
 * Drop the lookup steps Genie ran on its way to the answer.
 *
 * A governed turn often starts by asking the workspace what exists — which months
 * have data, which categories are spelled how — and each of those round-trips comes
 * back as its own typed result. Shown beside the real answer they read as a broken
 * table: one bare column of keys under the heading "Query result 1".
 *
 * A single-column result is treated as scaffolding only when a richer result exists
 * to be the answer instead. On its own it stays, because "which months do I have?"
 * is a legitimate question whose answer is exactly that one column.
 */
export function withoutScaffoldingResults(results: AnalysisResult[]): AnalysisResult[] {
  const richest = results.reduce((peak, result) => Math.max(peak, result.columns.length), 0);
  if (richest < 2) return results;
  const kept = results.filter((result) => result.columns.length > 1);
  return kept.length ? kept : results;
}

export function withoutEmbeddedQueryMarkdown(
  markdown: string,
  hasTypedResults: boolean
): string {
  if (!hasTypedResults) return markdown;
  const withoutEmbedded = markdown
    .replace(
      /<!--\s*begin-embedded:[\s\S]*?<!--\s*end-embedded:[^>]*-->/gi,
      ""
    );
  // Genie One may include the same result as both a Markdown table and typed
  // query rows without embedded markers. Keep prose, but render tabular evidence
  // only from the typed contract so callers do not see every result twice.
  const lines = withoutEmbedded.split("\n");
  const kept: string[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    const header = lines[index];
    const rule = lines[index + 1] ?? "";
    const isTable =
      header.includes("|") &&
      /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(rule);
    if (!isTable) {
      kept.push(header);
      continue;
    }
    index += 2;
    while (index < lines.length && lines[index].includes("|")) index += 1;
    index -= 1;
  }
  return kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function StructuredAnswer({
  markdown,
  results,
  language,
}: {
  markdown: string;
  results: AnalysisResult[];
  /** The call language: this panel's chrome and number formatting follow it. */
  language?: string;
}) {
  const copy = uiCopy(language as InteractionLanguage | undefined);
  const shown = withoutScaffoldingResults(results);
  const prose = withoutEmbeddedQueryMarkdown(markdown, shown.length > 0);
  return (
    <div className="kb-structured-answer">
      {prose && (
        <div className="kb-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {prose}
          </ReactMarkdown>
        </div>
      )}
      {shown.length > 0 && (
        <details className="kb-analysis-evidence">
          <summary>
            {copy.saGovernedAnalysis} ·{" "}
            {copy.saRows(
              String(shown.reduce((count, result) => count + result.totalRowCount, 0))
            )}
          </summary>
          {shown.map((result, index) => (
            <section className="kb-analysis" key={result.itemId || index}>
              <div className="kb-analysis-head">
                <div>
                  <span className="kb-analysis-eyebrow">{copy.saGovernedAnalysis}</span>
                  <h4>
                    {shown.length > 1
                      ? copy.saQueryResultNumbered(String(index + 1))
                      : copy.saQueryResult}
                  </h4>
                </div>
                <span>{copy.saRows(String(result.totalRowCount))}</span>
              </div>
              <AnalysisChart result={result} copy={copy} />
              <ResultTable result={result} copy={copy} language={language ?? ""} />
              {result.sql && (
                <details className="kb-analysis-sql">
                  <summary>{copy.saViewSql}</summary>
                  <pre>{result.sql}</pre>
                </details>
              )}
            </section>
          ))}
        </details>
      )}
    </div>
  );
}
