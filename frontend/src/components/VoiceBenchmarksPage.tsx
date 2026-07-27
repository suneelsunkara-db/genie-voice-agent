import { useEffect, useMemo, useState } from "react";
import {
  api,
  VoiceBenchmarkBaseline,
  VoiceBenchmarkRun,
  VoiceBenchmarksResponse,
} from "../api/client";
import "../styles/voice-benchmarks.css";

/**
 * Voice Benchmarks — a plain-language, new-user-friendly view of how the Genie
 * realtime voice APIs (speech-to-text and text-to-speech) score on public
 * FLEURS benchmark audio/text across the supported languages.
 *
 * Design goals: (1) anyone can understand it without prior benchmark knowledge —
 * every metric is spelled out and colour-coded; (2) it covers ALL benchmarked
 * languages; (3) it highlights where the API shines vs published baselines.
 *
 * Data comes straight from the Delta benchmark_runs table via the realtime API
 * (GET /realtime/v1/benchmarks). No re-computation of the science here — just a
 * clearer presentation of the same numbers the /realtime-test page used to show.
 */

// Friendly language names keyed by the base (2/3-letter) BCP-47 subtag.
const LANG_NAMES: Record<string, string> = {
  en: "English", zh: "Chinese", ja: "Japanese", ko: "Korean", es: "Spanish",
  fr: "French", de: "German", it: "Italian", pt: "Portuguese", ru: "Russian",
  ar: "Arabic", hi: "Hindi", th: "Thai", id: "Indonesian", vi: "Vietnamese",
  yue: "Cantonese", nl: "Dutch", pl: "Polish", tr: "Turkish", sv: "Swedish",
  ro: "Romanian", el: "Greek", cs: "Czech", da: "Danish", fi: "Finnish",
  hu: "Hungarian", fa: "Persian", fil: "Filipino", ms: "Malay", lo: "Lao",
  km: "Khmer", my: "Burmese",
};

// Non-spaced scripts scored with Character Error Rate rather than Word ER.
const CER_LANGS = new Set(["zh", "ja", "th", "lo", "km", "my", "yue"]);

const DATASET_META: Record<
  string,
  { title: string; blurb: string; kind: "error"; asks: string }
> = {
  fleurs: {
    title: "FLEURS",
    blurb: "How accurately the API transcribes real speech across the supported languages.",
    asks: "Transcribe the spoken sentence and score against the reference.",
    kind: "error",
  },
};

const DATASET_ORDER = ["fleurs"];

// Vendor datasets are measured internally for comparison but not surfaced in the UI.
const HIDDEN_DATASETS = new Set(["fleurs_deepgram_stt", "fleurs_elevenlabs_tts"]);

// ---- number helpers -------------------------------------------------------- //
function baseLang(code?: string | null): string {
  return (code || "").split("-")[0];
}
function langName(code?: string | null): string {
  const b = baseLang(code);
  return LANG_NAMES[b] || (code || "").toUpperCase();
}
function num(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}
function fmtPct(v: number | null, digits = 1): string {
  if (v == null) return "—";
  const pct = v <= 1 ? v * 100 : v;
  const f = Math.pow(10, digits);
  return `${Math.round(pct * f) / f}%`;
}
function fmtMs(v: number | null): string {
  if (v == null) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`;
}
function avg(arr: (number | null)[]): number | null {
  const v = arr.filter((x): x is number => typeof x === "number");
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
// A latency_ms value is either a percentile object {p50,p95,...} or a bare number.
function lat(run: VoiceBenchmarkRun, key: string, stat: "p50" | "p95" | "mean" = "p50"): number | null {
  const v = (run.latency_ms || {})[key];
  if (v == null) return null;
  if (typeof v === "number") return v;
  return num((v as Record<string, unknown>)[stat]);
}
function sScore(run: VoiceBenchmarkRun, key: string): number | null {
  return num((run.scores || {})[key]);
}
// FLEURS primary error honours CER for non-spaced scripts.
function asrErr(run: VoiceBenchmarkRun): number | null {
  if (num(run.primary_score) != null) return num(run.primary_score);
  return CER_LANGS.has(baseLang(run.language)) ? sScore(run, "cer") : sScore(run, "wer");
}
function score(run: VoiceBenchmarkRun): number | null {
  if (run.evaluator === "asr") return asrErr(run);
  return asrErr(run);
}
function errRate(run: VoiceBenchmarkRun): number | null {
  return run.samples ? (run.errors || 0) / run.samples : 0;
}

type Quality = "good" | "warn" | "bad" | "neutral";
function classAcc(pct: number | null): Quality {
  if (pct == null) return "neutral";
  return pct >= 70 ? "good" : pct >= 40 ? "warn" : "bad";
}
function classErr(rate: number | null): Quality {
  if (rate == null) return "neutral";
  const r = rate <= 1 ? rate : rate / 100;
  return r <= 0.15 ? "good" : r <= 0.35 ? "warn" : "bad";
}
function classRel(rate: number | null): Quality {
  if (rate == null) return "neutral";
  return rate === 0 ? "good" : rate <= 0.15 ? "warn" : "bad";
}

// ---- small presentational bits -------------------------------------------- //
function Bar({ pct, cls, label }: { pct: number | null; cls: Quality; label: string }) {
  const w = Math.max(2, Math.min(100, pct == null ? 0 : pct));
  return (
    <div className="vb-bar">
      <span className={`vb-fill ${cls}`} style={{ width: `${w}%` }} />
      <em>{label}</em>
    </div>
  );
}

// ---- baseline indexing ----------------------------------------------------- //
interface AggRef {
  label: string;
  metric?: string | null;
  value: number | null;
  source?: string;
  url?: string;
  note?: string;
}
function indexBaselines(baselines: VoiceBenchmarkBaseline[] | undefined) {
  const aggByDs: Record<string, AggRef[]> = {};
  const langByDs: Record<string, Record<string, { value: number | null; label?: string }>> = {};
  (baselines || []).forEach((b) => {
    if (b.scope === "aggregate") {
      (aggByDs[b.dataset] = aggByDs[b.dataset] || []).push({
        label: b.system_label || "Reference",
        metric: b.primary_metric,
        value: num(b.primary_score),
        source: b.reference_source,
        url: b.reference_url,
        note: b.note,
      });
    } else if (b.scope === "language" && b.language) {
      const m = (langByDs[b.dataset] = langByDs[b.dataset] || {});
      m[b.language] = { value: num(b.primary_score), label: b.system_label };
    }
  });
  return { aggByDs, langByDs };
}

// ---- dataset section ------------------------------------------------------- //
function DatasetSection({
  dataset,
  runs,
  langRef,
  aggRefs,
}: {
  dataset: string;
  runs: VoiceBenchmarkRun[];
  langRef: Record<string, { value: number | null; label?: string }>;
  aggRefs: AggRef[];
}) {
  const meta = DATASET_META[dataset] || { title: dataset, blurb: "", asks: "", kind: "error" as const };
  const isErr = meta.kind === "error";
  const refLabel = aggRefs[0]?.label || "Reference";

  const sorted = useMemo(() => {
    return runs.slice().sort((a, b) => {
      const sa = score(a);
      const sb = score(b);
      if (sa == null) return 1;
      if (sb == null) return -1;
      return isErr ? sa - sb : sb - sa; // best first
    });
  }, [runs, isErr]);

  return (
    <section className="vb-ds">
      <header className="vb-ds-head">
        <div>
          <h3>
            {meta.title}
            <span className={`vb-tag ${isErr ? "err" : "acc"}`}>
              {isErr ? "lower is better ↓" : "higher is better ↑"}
            </span>
          </h3>
          <p className="vb-ds-blurb">{meta.blurb}</p>
          <p className="vb-ds-asks">Task: {meta.asks}</p>
        </div>
        <div className="vb-ds-count">{runs.length} languages</div>
      </header>

      {aggRefs.length > 0 && (
        <ComparisonStrip
          dataset={dataset}
          isErr={isErr}
          headline={avg((isErr ? sorted : sorted.filter((r) => !r.errors)).map(score))}
          refs={aggRefs}
        />
      )}

      <div className="vb-table-wrap">
        <table className="vb-table">
          <thead>
            <tr>
              <th>Language</th>
              <th>{isErr ? "Transcription error" : "Accuracy"}</th>
              <th className="num">Detail</th>
              {isErr && <th className="num">{refLabel}</th>}
              <th>Reliability</th>
              <th className="num" title="Median speech-to-text time">STT p50</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((run, i) => {
              const s = run.scores || {};
              let barCell: JSX.Element;
              let detailCell: JSX.Element;
              let refCell: JSX.Element | null = null;
              if (isErr) {
                const e = asrErr(run);
                const cls = classErr(e);
                const ePct = e == null ? null : e <= 1 ? e * 100 : e;
                const fill = ePct == null ? 0 : Math.max(0, 100 - ePct); // lower err => fuller
                barCell = <Bar pct={fill} cls={cls} label={fmtPct(e)} />;
                const primaryCer = CER_LANGS.has(baseLang(run.language));
                detailCell = (
                  <span className="vb-muted">
                    {fmtPct(sScore(run, "wer"))} · {fmtPct(sScore(run, "cer"))}
                    {primaryCer ? " *" : ""}
                  </span>
                );
                const rv = langRef[baseLang(run.language)]?.value ?? null;
                let delta: JSX.Element | null = null;
                if (rv != null && e != null) {
                  const d = (e <= 1 ? e * 100 : e) - (rv <= 1 ? rv * 100 : rv);
                  const better = d < 0;
                  delta = (
                    <span className={better ? "vb-delta good" : "vb-delta bad"}>
                      {better ? "▼" : "▲"}
                      {Math.abs(Math.round(d * 10) / 10)}
                    </span>
                  );
                }
                refCell = (
                  <td className="num">
                    <span className="vb-muted">{rv == null ? "—" : fmtPct(rv)}</span> {delta}
                  </td>
                );
              } else {
                const a = score(run);
                const cls = classAcc(a);
                barCell = <Bar pct={a} cls={cls} label={fmtPct(a, 0)} />;
                detailCell =
                  run.evaluator === "mcq" ? (
                    <span className="vb-muted">parse-fail {fmtPct(sScore(run, "parse_fail_rate"))}</span>
                  ) : (
                    <span className="vb-muted">{s.method === "gpt_judge" ? "AI-judged" : "match"}</span>
                  );
              }
              const rel = errRate(run);
              const relCls = classRel(rel);
              return (
                <tr key={`${run.dataset}-${run.language}-${i}`}>
                  <td className="vb-lang">
                    <span className="vb-lang-name">{langName(run.language)}</span>
                    <span className="vb-lang-code">{run.language}</span>
                  </td>
                  <td>{barCell}</td>
                  <td className="num">{detailCell}</td>
                  {refCell}
                  <td>
                    <span className={`vb-pill ${relCls === "good" ? "ok" : relCls}`}>
                      {run.errors ? `${run.errors}/${run.samples} err` : `${run.samples ?? "—"} ok`}
                    </span>
                  </td>
                  <td className="num">{fmtMs(lat(run, "stt_ms"))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ComparisonStrip({
  dataset,
  isErr,
  headline,
  refs,
}: {
  dataset: string;
  isErr: boolean;
  headline: number | null;
  refs: AggRef[];
}) {
  const rows = [
    { ours: true as const, label: "Genie Voice (measured)", value: headline, source: "this run", url: undefined },
    ...refs.map((r) => ({ ours: false as const, label: r.label, value: r.value, source: r.source, url: r.url })),
  ];
  return (
    <div className="vb-compare">
      <div className="vb-compare-head">
        <span>Compared to published leaderboards</span>
        <span className="vb-muted">{isErr ? "error rate ↓" : "accuracy ↑"}</span>
      </div>
      {rows.map((row, i) => {
        const v = row.value;
        const pctV = v == null ? null : v <= 1 ? v * 100 : v;
        const fill = pctV == null ? 0 : isErr ? Math.max(0, 100 - pctV) : pctV;
        const cls = v == null ? "neutral" : isErr ? classErr(v) : classAcc(v);
        return (
          <div key={i} className={`vb-compare-row${row.ours ? " ours" : ""}`}>
            <div className="vb-compare-who">
              <b>{row.label}</b>
              <span className={`vb-tag2 ${row.ours ? "ours" : "ref"}`}>
                {row.ours ? "measured" : "reference"}
              </span>
              {row.url ? (
                <a className="vb-cite" href={row.url} target="_blank" rel="noopener noreferrer">
                  {row.source || "source"}
                </a>
              ) : (
                <span className="vb-cite">{row.source || ""}</span>
              )}
            </div>
            <Bar pct={fill} cls={cls} label={v == null ? "—" : fmtPct(v, isErr ? 1 : 0)} />
          </div>
        );
      })}
      {dataset === "fleurs" && (
        <p className="vb-compare-note">
          References are published numbers for the same benchmark — not re-measured here — so treat cross-system gaps as
          directional.
        </p>
      )}
    </div>
  );
}

function avgScoreForDataset(runs: VoiceBenchmarkRun[], dataset: string, key?: string): number | null {
  const selected = runs.filter((r) => r.dataset === dataset && !r.errors);
  if (!selected.length) return null;
  return avg(selected.map((r) => (key ? sScore(r, key) : asrErr(r))));
}

function ModelComparisonSection({
  runs,
  refs,
}: {
  runs: VoiceBenchmarkRun[];
  refs: AggRef[];
}) {
  const sttRows = [
    { label: "Genie Voice API", value: avgScoreForDataset(runs, "fleurs"), kind: "measured" },
    ...refs.map((r) => ({ label: r.label, value: r.value, kind: "published" })),
  ].filter((r) => r.value != null);

  if (!sttRows.length) return null;

  const renderRows = (rows: typeof sttRows) =>
    rows.map((row, i) => {
      const v = row.value;
      const pctV = v == null ? null : v <= 1 ? v * 100 : v;
      const fill = pctV == null ? 0 : Math.max(0, 100 - pctV);
      return (
        <div key={`${row.label}-${i}`} className="vb-compare-row">
          <div className="vb-compare-who">
            <b>{row.label}</b>
            <span className={`vb-tag2 ${row.kind === "measured" ? "ours" : "ref"}`}>{row.kind}</span>
          </div>
          <Bar pct={fill} cls={classErr(v)} label={fmtPct(v)} />
        </div>
      );
    });

  return (
    <section className="vb-block">
      <h2 className="vb-block-title">Model Comparison</h2>
      <div className="vb-grid">
        <div className="vb-card">
          <h3>Speech-to-text on FLEURS</h3>
          <p className="vb-muted">
            Genie is measured on this benchmark table. Whisper, MMS, and SeamlessM4T rows are published FLEURS
            references.
          </p>
          <div className="vb-compare">{renderRows(sttRows)}</div>
        </div>
      </div>
    </section>
  );
}

// ---- "where Genie shines" highlights (derived from the data) --------------- //
interface Highlight {
  icon: string;
  title: string;
  body: string;
}
function buildHighlights(
  runs: VoiceBenchmarkRun[],
  langByDs: Record<string, Record<string, { value: number | null; label?: string }>>,
  languageCount: number,
): Highlight[] {
  const out: Highlight[] = [];

  // 1. Breadth of languages.
  if (languageCount > 0) {
    out.push({
      icon: "🌐",
      title: `${languageCount} languages, one voice API`,
      body: "The same deployed realtime STT API is evaluated across every language below.",
    });
  }

  // 2. FLEURS: how often we match/beat Whisper large-v3.
  const fleurs = runs.filter((r) => r.dataset === "fleurs");
  const fleursRef = langByDs["fleurs"] || {};
  if (fleurs.length) {
    let compared = 0;
    let beatenOrMatched = 0;
    fleurs.forEach((r) => {
      const rv = fleursRef[baseLang(r.language)]?.value;
      const e = asrErr(r);
      if (rv != null && e != null) {
        compared += 1;
        const ours = e <= 1 ? e * 100 : e;
        const ref = rv <= 1 ? rv * 100 : rv;
        if (ours <= ref + 0.5) beatenOrMatched += 1; // within 0.5pt counts as matched
      }
    });
    const best = fleurs
      .map((r) => ({ lang: langName(r.language), e: asrErr(r) }))
      .filter((x) => x.e != null)
      .sort((a, b) => (a.e! - b.e!))[0];
    if (compared > 0) {
      out.push({
        icon: "🎯",
        title: `Matches or beats Whisper large-v3 on ${beatenOrMatched}/${compared} languages`,
        body: best
          ? `Strongest transcription: ${best.lang} at ${fmtPct(best.e)} error. Lower is better.`
          : "Transcription error rate vs the published Whisper large-v3 FLEURS numbers.",
      });
    } else if (best) {
      out.push({
        icon: "🎯",
        title: `Transcription as low as ${fmtPct(best.e)} error`,
        body: `Strongest language: ${best.lang}. FLEURS word/character error rate — lower is better.`,
      });
    }
  }

  // 3. Reliability across the whole sweep.
  const totSamples = runs.reduce((a, r) => a + (r.samples || 0), 0);
  const totErrors = runs.reduce((a, r) => a + (r.errors || 0), 0);
  if (totSamples > 0) {
    const ok = 1 - totErrors / totSamples;
    out.push({
      icon: "✅",
      title: `${fmtPct(ok, 1)} of turns completed cleanly`,
      body: `${totSamples - totErrors} of ${totSamples} evaluated turns ran without an endpoint error.`,
    });
  }

  // 5. Latency.
  const medStt = avg(runs.map((r) => lat(r, "stt_ms")));
  if (medStt != null) {
    out.push({
      icon: "⚡",
      title: `Median speech-to-text in ${fmtMs(medStt)}`,
      body: "Median p50 latency of the speech-to-text stage.",
    });
  }

  return out;
}

// ---- page ------------------------------------------------------------------ //
export function VoiceBenchmarksPage() {
  const [data, setData] = useState<VoiceBenchmarksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    api
      .voiceBenchmarks()
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : "failed"))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  const runs = useMemo(
    () => (data?.runs || []).filter((r) => r && r.dataset && !HIDDEN_DATASETS.has(r.dataset)),
    [data],
  );
  const { aggByDs, langByDs } = useMemo(() => indexBaselines(data?.baselines), [data]);
  const byDataset = useMemo(() => {
    const m: Record<string, VoiceBenchmarkRun[]> = {};
    runs.forEach((r) => (m[r.dataset] = m[r.dataset] || []).push(r));
    return m;
  }, [runs]);
  const datasetOrder = useMemo(
    () =>
      DATASET_ORDER.filter((d) => byDataset[d]).concat(
        Object.keys(byDataset).filter((d) => !DATASET_ORDER.includes(d)),
      ),
    [byDataset],
  );
  const highlights = useMemo(
    () => buildHighlights(runs, langByDs, (data?.languages || []).length),
    [runs, langByDs, data],
  );

  const generatedAt = (data?.generated_at || "").replace("T", " ").replace(/\.\d+.*/, "").replace(/Z?$/, " UTC");

  return (
    <div className="vb-root">
      <header className="vb-header">
        <div className="vb-title">
          <span>Genie</span> Voice · Benchmarks
        </div>
        <div className="vb-header-spacer" />
        <button className="vb-btn ghost" onClick={() => (window.location.hash = "#/")}>
          ← Cockpit
        </button>
        <button className="vb-btn ghost" onClick={() => (window.location.hash = "#/traces")}>
          Traces
        </button>
        <button className="vb-btn" onClick={load}>
          Refresh
        </button>
      </header>

      <div className="vb-scroll">
        {/* Intro / what am I looking at */}
        <section className="vb-hero">
          <h1>How well does the voice API hear?</h1>
          <p>
            These are <b>real, measured results</b> from the Genie realtime STT API on FLEURS: speech transcription
            accuracy and latency across the supported languages. Published model rows are references from papers or
            leaderboards, not re-measured competitor runs.
          </p>
          {data?.available && (
            <div className="vb-run-meta">
              <span>
                Source <b>Delta</b>
              </span>
              <span>
                run <b>{data.run_id || "—"}</b>
              </span>
              <span>{generatedAt || "—"}</span>
              <span>
                <b>{(data.languages || []).length}</b> languages
              </span>
              <span>
                <b>{datasetOrder.length}</b> benchmarks
              </span>
            </div>
          )}
        </section>

        {loading && <div className="vb-empty">Loading benchmark results…</div>}
        {err && !loading && <div className="vb-empty">Could not load benchmarks: {err}</div>}
        {!loading && !err && data && !data.available && (
          <div className="vb-empty">{data.message || "No benchmark results yet."}</div>
        )}

        {!loading && data?.available && (
          <>
            {/* Where Genie shines */}
            {highlights.length > 0 && (
              <section className="vb-block">
                <h2 className="vb-block-title">Where Genie Voice shines</h2>
                <div className="vb-highlights">
                  {highlights.map((h, i) => (
                    <div className="vb-highlight" key={i}>
                      <div className="vb-highlight-icon">{h.icon}</div>
                      <div className="vb-highlight-title">{h.title}</div>
                      <div className="vb-highlight-body">{h.body}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* How to read this */}
            <section className="vb-block">
              <h2 className="vb-block-title">How to read this</h2>
              <div className="vb-legend">
                <div className="vb-legend-item">
                  <div className="vb-legend-k">Transcription error (WER / CER)</div>
                  <div className="vb-legend-v">
                    Share of words (or characters for Chinese/Japanese/Thai, marked <code>*</code>) the agent got wrong
                    when writing down speech. <b>Lower is better ↓.</b>
                  </div>
                </div>
                <div className="vb-legend-item">
                  <div className="vb-legend-k">Accuracy</div>
                  <div className="vb-legend-v">
                    Share of questions answered correctly after listening. <b>Higher is better ↑.</b>
                  </div>
                </div>
                <div className="vb-legend-item">
                  <div className="vb-legend-k">Reliability</div>
                  <div className="vb-legend-v">
                    How many evaluated turns completed without a service error. Errored turns count as wrong, so low
                    accuracy with high errors reflects capacity, not model quality.
                  </div>
                </div>
                <div className="vb-legend-item">
                  <div className="vb-legend-k">STT p50</div>
                  <div className="vb-legend-v">
                    Median time for the speech-to-text stage.
                  </div>
                </div>
                <div className="vb-legend-swatches">
                  <span className="vb-sw good">good</span>
                  <span className="vb-sw warn">fair</span>
                  <span className="vb-sw bad">needs work</span>
                </div>
              </div>
            </section>

            <ModelComparisonSection runs={runs} refs={aggByDs["fleurs"] || []} />

            {/* Per-dataset per-language detail */}
            <section className="vb-block">
              <h2 className="vb-block-title">Full results by benchmark & language</h2>
              {datasetOrder.map((d) => (
                <DatasetSection
                  key={d}
                  dataset={d}
                  runs={byDataset[d]}
                  langRef={langByDs[d] || {}}
                  aggRefs={aggByDs[d] || []}
                />
              ))}
            </section>

            <footer className="vb-footnote">
              FLEURS is the public multilingual speech benchmark used here. Published model references may use different
              language subsets and decoding settings, so treat cross-system gaps as directional. Genie and vendor rows are
              measured by jobs that write to Delta.
            </footer>
          </>
        )}
      </div>
    </div>
  );
}
