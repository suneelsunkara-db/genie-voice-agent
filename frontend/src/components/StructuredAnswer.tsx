import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const numeric = numberValue(value);
  if (numeric !== null && typeof value === "number") {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(numeric);
  }
  return String(value);
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
  const measures = numeric.filter((column) => column !== dimension).slice(0, 3);
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

function AnalysisChart({ result }: { result: AnalysisResult }) {
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
    <div className="kb-analysis-chart" aria-label="Chart of Genie One query results">
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

function ResultTable({ result }: { result: AnalysisResult }) {
  if (result.rows.length === 1) {
    return (
      <div className="kb-kpis">
        {result.columns.map((column, index) => (
          <div className="kb-kpi" key={column.name}>
            <span>{displayName(column.name)}</span>
            <strong>{formatValue(result.rows[0][index])}</strong>
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
                <td key={column.name}>{formatValue(row[columnIndex])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {(result.truncated || result.totalRowCount > result.rows.length) && (
        <div className="kb-result-note">
          Showing {result.rows.length} of {result.totalRowCount} rows
        </div>
      )}
    </div>
  );
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
}: {
  markdown: string;
  results: AnalysisResult[];
}) {
  const prose = withoutEmbeddedQueryMarkdown(markdown, results.length > 0);
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
      {results.map((result, index) => (
        <section className="kb-analysis" key={result.itemId || index}>
          <div className="kb-analysis-head">
            <div>
              <span className="kb-analysis-eyebrow">Governed analysis</span>
              <h4>Query result{results.length > 1 ? ` ${index + 1}` : ""}</h4>
            </div>
            <span>{result.totalRowCount} rows</span>
          </div>
          <AnalysisChart result={result} />
          <ResultTable result={result} />
          {result.sql && (
            <details className="kb-analysis-sql">
              <summary>View SQL</summary>
              <pre>{result.sql}</pre>
            </details>
          )}
        </section>
      ))}
    </div>
  );
}
