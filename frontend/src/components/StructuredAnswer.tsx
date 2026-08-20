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
};

const CHART_COLORS = ["#4ea8ff", "#8b7cff", "#31d0aa"];
const NUMERIC_TYPE = /(tiny|small|big)?int|decimal|double|float|number|numeric|real/i;
const TIME_NAME = /date|time|month|year|week|day|quarter|cycle/i;

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

/** Largest absolute value a column reaches, for comparing measure scales. */
function columnMagnitude(result: AnalysisResult, column: AnalysisColumn): number {
  const index = result.columns.indexOf(column);
  return result.rows.reduce((peak, row) => {
    const value = numberValue(row[index]);
    return value === null ? peak : Math.max(peak, Math.abs(value));
  }, 0);
}

/**
 * A shared Y axis only works for measures of comparable size.
 *
 * Genie results routinely mix a total, a count, and an average in one row — 416,659
 * beside 1,645 beside 252. Plotting those together scales the axis to the total and
 * flattens everything else onto the baseline, which reads as a chart that failed to
 * render. So chart the largest measure and only those within this factor of it; the
 * rest stay in the table underneath, where their values are legible.
 */
const MEASURE_SCALE_RATIO = 20;

function comparableMeasures(
  result: AnalysisResult,
  candidates: AnalysisColumn[]
): AnalysisColumn[] {
  const ranked = candidates
    .map((column) => ({ column, magnitude: columnMagnitude(result, column) }))
    .sort((a, b) => b.magnitude - a.magnitude);
  const largest = ranked[0]?.magnitude ?? 0;
  if (largest <= 0) return candidates.slice(0, 3);
  return ranked
    .filter(({ magnitude }) => magnitude * MEASURE_SCALE_RATIO >= largest)
    .map(({ column }) => column)
    .slice(0, 3);
}

export function chartModel(result: AnalysisResult): {
  data: Record<string, string | number>[];
  dimension: AnalysisColumn;
  measures: AnalysisColumn[];
  kind: "bar" | "line";
} | null {
  if (result.rows.length < 2 || result.rows.length > 40) return null;
  const numeric = result.columns.filter((column, index) => {
    if (NUMERIC_TYPE.test(column.typeText)) return true;
    return result.rows.every((row) => numberValue(row[index]) !== null);
  });
  if (!numeric.length) return null;
  const dimension =
    result.columns.find((column) => !numeric.includes(column)) ?? result.columns[0];
  const dimensionIndex = result.columns.indexOf(dimension);
  const measures = comparableMeasures(
    result,
    numeric.filter((column) => column !== dimension)
  );
  if (!measures.length) return null;
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
    kind: TIME_NAME.test(dimension.name) ? "line" : "bar",
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
      />
      <YAxis
        tick={{ fill: "#9eabc9", fontSize: 11 }}
        axisLine={false}
        tickLine={false}
        width={58}
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
  return markdown
    .replace(
      /<!--\s*begin-embedded:[\s\S]*?<!--\s*end-embedded:[^>]*-->/gi,
      ""
    )
    .trim();
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
    </div>
  );
}
