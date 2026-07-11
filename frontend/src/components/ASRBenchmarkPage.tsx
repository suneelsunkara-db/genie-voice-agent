import { useEffect, useState } from "react";
import {
  api,
  ASRBenchmarkExample,
  ASRBenchmarkResponse,
  ASRProviderSummary,
  InteractionLanguage,
  INTERACTION_LANGUAGES,
} from "../api/client";

function pct(value?: number | null) {
  if (value === null || value === undefined) return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

function num(value?: number | null, suffix = "") {
  if (value === null || value === undefined) return "n/a";
  return `${Math.round(value * 100) / 100}${suffix}`;
}

function winnerFor({
  deepgram,
  databricks,
  lowerIsBetter = false,
}: {
  deepgram?: number | null;
  databricks?: number | null;
  lowerIsBetter?: boolean;
}) {
  if (deepgram === null || databricks === null || deepgram === undefined || databricks === undefined) return "Tie";
  if (Math.abs(deepgram - databricks) < 0.0001) return "Tie";
  const databricksWins = lowerIsBetter ? databricks < deepgram : databricks > deepgram;
  return databricksWins ? "Databricks" : "Deepgram";
}

function deltaPct(databricks?: number | null, deepgram?: number | null) {
  if (databricks === null || databricks === undefined || deepgram === null || deepgram === undefined) return "n/a";
  const delta = (databricks - deepgram) * 100;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${Math.round(delta * 10) / 10} pts`;
}

function deltaMs(databricks?: number | null, deepgram?: number | null) {
  if (databricks === null || databricks === undefined || deepgram === null || deepgram === undefined) return "n/a";
  const delta = Math.round(databricks - deepgram);
  return `${delta > 0 ? "+" : ""}${delta}ms`;
}

function shortPath(path?: string) {
  if (!path) return "n/a";
  const parts = path.split("/");
  return parts.slice(-3).join("/");
}

function MetricCard({
  label,
  help,
  deepgram,
  databricks,
  winner,
  delta,
}: {
  label: string;
  help: string;
  deepgram: string;
  databricks: string;
  winner: string;
  delta: string;
}) {
  return (
    <div className="benchmark-metric-card">
      <div className="benchmark-metric-topline">
        <div>
          <div className="benchmark-metric-label">{label}</div>
          <p>{help}</p>
        </div>
        <span className={`benchmark-winner ${winner.toLowerCase()}`}>{winner}</span>
      </div>
      <div className="benchmark-metric-grid">
        <span>Deepgram</span>
        <strong>{deepgram}</strong>
        <span>Databricks</span>
        <strong>{databricks}</strong>
        <span>Databricks delta</span>
        <strong>{delta}</strong>
      </div>
    </div>
  );
}

function ProviderSummary({
  name,
  summary,
}: {
  name: string;
  summary?: ASRProviderSummary;
}) {
  return (
    <div className="benchmark-provider-card">
      <div className="benchmark-provider-title">
        <span>{name}</span>
        <em>{summary?.clips ?? "n/a"} clips</em>
      </div>
      <div className="benchmark-provider-kpis">
        <div>
          <span>WER</span>
          <strong>{pct(summary?.avg_wer)}</strong>
        </div>
        <div>
          <span>critical entities</span>
          <strong>{pct(summary?.avg_critical_entity_accuracy)}</strong>
        </div>
        <div>
          <span>p95 latency</span>
          <strong>{num(summary?.latency_ms?.p95, "ms")}</strong>
        </div>
        <div>
          <span>unsafe rate</span>
          <strong>{pct(summary?.unsafe_for_resolution_rate)}</strong>
        </div>
      </div>
    </div>
  );
}

const entityLabels: Record<string, { label: string; why: string }> = {
  invoice_ids: { label: "Invoice IDs", why: "wrong invoice can trigger the wrong billing action" },
  amounts: { label: "Dollar amounts", why: "used in agent explanations and adjustment checks" },
  dates: { label: "Dates", why: "payment timing and due-date context" },
  billing_actions: { label: "Billing actions", why: "waiver, payment-plan, refund language" },
  confirmations: { label: "Confirmations", why: "controls whether the app can safely close" },
  refusals: { label: "Refusals / negation", why: "prevents acting against customer intent" },
  account_terms: { label: "Account terms", why: "billing vocabulary preservation" },
};

function EntityBreakdown({
  deepgram,
  databricks,
}: {
  deepgram?: ASRProviderSummary;
  databricks?: ASRProviderSummary;
}) {
  const groups = Array.from(
    new Set([
      ...Object.keys(deepgram?.entity_groups ?? {}),
      ...Object.keys(databricks?.entity_groups ?? {}),
    ])
  ).sort();

  return (
    <div className="benchmark-panel">
      <div className="benchmark-section-head">
        <div>
          <h2>Business Entity Accuracy</h2>
          <p>These are the words and values the agent workflow acts on. They matter more than generic WER.</p>
        </div>
      </div>
      <div className="benchmark-entity-table">
        <div className="benchmark-table-head">Entity</div>
        <div className="benchmark-table-head">Why It Matters</div>
        <div className="benchmark-table-head">Deepgram</div>
        <div className="benchmark-table-head">Databricks</div>
        <div className="benchmark-table-head">Winner</div>
        {groups.map((group) => (
          <div className="benchmark-table-row" key={group}>
            <div>{entityLabels[group]?.label ?? group.split("_").join(" ")}</div>
            <div>{entityLabels[group]?.why ?? "business-critical phrase preservation"}</div>
            <div>{pct(deepgram?.entity_groups?.[group]?.accuracy)}</div>
            <div>{pct(databricks?.entity_groups?.[group]?.accuracy)}</div>
            <div>
              <span
                className={`benchmark-winner ${winnerFor({
                  deepgram: deepgram?.entity_groups?.[group]?.accuracy,
                  databricks: databricks?.entity_groups?.[group]?.accuracy,
                }).toLowerCase()}`}
              >
                {winnerFor({
                  deepgram: deepgram?.entity_groups?.[group]?.accuracy,
                  databricks: databricks?.entity_groups?.[group]?.accuracy,
                })}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ExampleCard({ example }: { example: ASRBenchmarkExample }) {
  const databricksUnsafe = example.databricks_unsafe_reasons ?? [];
  const deepgramUnsafe = example.deepgram_unsafe_reasons ?? [];
  return (
    <div className={`benchmark-example ${databricksUnsafe.length || deepgramUnsafe.length ? "unsafe" : ""}`}>
      <div className="benchmark-example-head">
        <strong>{example.clip_id}</strong>
        {example.scenario && <span>{example.scenario}</span>}
        {deepgramUnsafe.length > 0 && <em>Deepgram unsafe: {deepgramUnsafe.join(", ")}</em>}
        {databricksUnsafe.length > 0 && <em>Databricks unsafe: {databricksUnsafe.join(", ")}</em>}
      </div>
      <div className="benchmark-transcript-grid">
        <div>
          <h3>Reference</h3>
          <p>{example.reference_transcript}</p>
        </div>
        <div>
          <h3>Deepgram</h3>
          <p>{example.deepgram_transcript}</p>
          <small>
            WER {pct(example.deepgram_wer)} · critical entities{" "}
            {pct(example.deepgram_critical_entity_accuracy)} · latency {num(example.deepgram_latency_ms, "ms")}
          </small>
        </div>
        <div>
          <h3>Databricks</h3>
          <p>{example.databricks_transcript}</p>
          <small>
            WER {pct(example.databricks_wer)} · critical entities{" "}
            {pct(example.databricks_critical_entity_accuracy)} · latency {num(example.databricks_latency_ms, "ms")}
          </small>
        </div>
      </div>
    </div>
  );
}

export function ASRBenchmarkPage() {
  const [language, setLanguage] = useState<InteractionLanguage>("th-TH");
  const [data, setData] = useState<ASRBenchmarkResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setData(null);
    setErr(null);
    api
      .asrBenchmark(language)
      .then((result) => {
        if (active) {
          setData(result);
          setErr(null);
        }
      })
      .catch((e) => {
        if (active) setErr(String(e));
      });
    return () => {
      active = false;
    };
  }, [language]);

  const deepgram = data?.summary?.providers?.deepgram;
  const databricks = data?.summary?.providers?.databricks;
  const promotion = data?.summary?.promotion_read;
  const clipCount = databricks?.clips ?? deepgram?.clips ?? promotion?.paired_clips;
  const latencyPenalty = deltaMs(databricks?.latency_ms?.p95, deepgram?.latency_ms?.p95);
  const responseLanguage = data?.language ?? language;
  const transcriptWinner = winnerFor({ deepgram: deepgram?.avg_wer, databricks: databricks?.avg_wer, lowerIsBetter: true });
  const entityWinner = winnerFor({
    deepgram: deepgram?.avg_critical_entity_accuracy,
    databricks: databricks?.avg_critical_entity_accuracy,
  });
  const latencyWinner = winnerFor({
    deepgram: deepgram?.latency_ms?.p95,
    databricks: databricks?.latency_ms?.p95,
    lowerIsBetter: true,
  });
  const languageOptions =
    data?.available_languages && data.available_languages.length > 0
      ? data.available_languages
      : INTERACTION_LANGUAGES.map((item) => ({ ...item, available: undefined }));
  const selectedLanguageLabel =
    languageOptions.find((item) => item.code === language)?.label ??
    language;

  return (
    <section className="benchmark-page">
      <div className="benchmark-page-head">
        <div>
          <div className="eyebrow">ASR Model Evaluation</div>
          <h1>Deepgram vs Databricks ASR endpoint</h1>
          <p>
            Offline 420-clip voice-model benchmark focused on what this app needs: accurate final
            utterances, billing entity preservation, safe resolution signals, and latency after mic stop.
          </p>
        </div>
        <div className="benchmark-head-actions">
          <label className="benchmark-language-control">
            <span>Benchmark language</span>
            <select value={language} onChange={(e) => setLanguage(e.target.value as InteractionLanguage)}>
              {languageOptions.map((item) => (
                <option key={item.code} value={item.code} disabled={item.available === false}>
                  {item.label}{item.available === false ? " (not packaged)" : ""}
                </option>
              ))}
            </select>
          </label>
          <a className="benchmark-back-link" href="#/">Back to cockpit</a>
        </div>
      </div>

      {err && <div className="error">ASR benchmark API error: {err}</div>}
      {data && !data.available && (
        <div className="benchmark-empty">
          <strong>No benchmark results found.</strong>
          <p>{data.message}</p>
          <code>scripts/asr/07_deep_voice_model_eval.sh run</code>
        </div>
      )}

      {data?.available && (
        <>
          <div className="benchmark-verdict">
            <div>
              <div className="eyebrow">Executive Readout</div>
              <h2>{selectedLanguageLabel}: Databricks vs Deepgram ASR benchmark.</h2>
              <p>
                Use this per-language result to compare final transcript accuracy, business entity preservation,
                unsafe resolution rate, and post-stop latency. Deepgram remains the streaming caption path; the
                Databricks model is evaluated as the final utterance transcription path.
              </p>
              <div className="benchmark-result-identity">
                <span>requested: {language}</span>
                <span>served: {responseLanguage}</span>
                <span>summary: {shortPath(data.summary_path)}</span>
              </div>
            </div>
            <div className="benchmark-verdict-numbers">
              <div>
                <span>clips compared</span>
                <strong>{clipCount ?? "n/a"}</strong>
              </div>
              <div>
                <span>critical entity delta</span>
                <strong>{deltaPct(databricks?.avg_critical_entity_accuracy, deepgram?.avg_critical_entity_accuracy)}</strong>
              </div>
              <div>
                <span>p95 latency delta</span>
                <strong>{latencyPenalty}</strong>
              </div>
            </div>
          </div>

          <div className="benchmark-reading-guide">
            <div>
              <strong>How to read this page</strong>
              <span>Each card declares its own winner. A lower error/latency is good; a higher accuracy is good.</span>
            </div>
            <div>
              <strong>No single winner</strong>
              <span>Use the category winners: accuracy-sensitive workflows favor Databricks; real-time UX favors Deepgram.</span>
            </div>
          </div>

          <div className="benchmark-glossary">
            <div>
              <strong>WER</strong>
              <span>Word Error Rate: percentage of word insertions, deletions, and substitutions. Lower is better.</span>
            </div>
            <div>
              <strong>CER</strong>
              <span>Character Error Rate: character-level transcript error. Useful for IDs and spelling. Lower is better.</span>
            </div>
            <div>
              <strong>Business Entity Accuracy</strong>
              <span>Whether billing facts like invoice IDs, amounts, dates, and confirmations were preserved. Higher is better.</span>
            </div>
            <div>
              <strong>Unsafe Rate</strong>
              <span>Share of transcripts that should not drive automatic resolution because a critical signal is missing. Lower is better.</span>
            </div>
          </div>

          <div className="benchmark-provider-row">
            <ProviderSummary name="Deepgram Nova-3" summary={deepgram} />
            <ProviderSummary name="Databricks Fine-Tuned Whisper" summary={databricks} />
          </div>

          <EntityBreakdown deepgram={deepgram} databricks={databricks} />

          <div className="benchmark-metric-row">
            <MetricCard
              label="Transcript Error"
              help="Average WER. Lower means fewer word-level mistakes."
              deepgram={pct(deepgram?.avg_wer)}
              databricks={pct(databricks?.avg_wer)}
              winner={winnerFor({ deepgram: deepgram?.avg_wer, databricks: databricks?.avg_wer, lowerIsBetter: true })}
              delta={deltaPct(databricks?.avg_wer, deepgram?.avg_wer)}
            />
            <MetricCard
              label="Critical Entity Accuracy"
              help="Invoice IDs, amounts, dates, actions, confirmations, and refusals."
              deepgram={pct(deepgram?.avg_critical_entity_accuracy)}
              databricks={pct(databricks?.avg_critical_entity_accuracy)}
              winner={winnerFor({
                deepgram: deepgram?.avg_critical_entity_accuracy,
                databricks: databricks?.avg_critical_entity_accuracy,
              })}
              delta={deltaPct(databricks?.avg_critical_entity_accuracy, deepgram?.avg_critical_entity_accuracy)}
            />
            <MetricCard
              label="Unsafe For Auto-Resolution"
              help="Rows with empty transcript, missing invoice/amount, or negation/entity risk."
              deepgram={pct(deepgram?.unsafe_for_resolution_rate)}
              databricks={pct(databricks?.unsafe_for_resolution_rate)}
              winner={winnerFor({
                deepgram: deepgram?.unsafe_for_resolution_rate,
                databricks: databricks?.unsafe_for_resolution_rate,
                lowerIsBetter: true,
              })}
              delta={deltaPct(databricks?.unsafe_for_resolution_rate, deepgram?.unsafe_for_resolution_rate)}
            />
            <MetricCard
              label="P95 Latency"
              help="Time from provider request to final transcript. Lower is better."
              deepgram={num(deepgram?.latency_ms?.p95, "ms")}
              databricks={num(databricks?.latency_ms?.p95, "ms")}
              winner={winnerFor({
                deepgram: deepgram?.latency_ms?.p95,
                databricks: databricks?.latency_ms?.p95,
                lowerIsBetter: true,
              })}
              delta={deltaMs(databricks?.latency_ms?.p95, deepgram?.latency_ms?.p95)}
            />
          </div>

          <div className="benchmark-tradeoff-grid">
            <div className="benchmark-tradeoff-card good">
              <h2>Transcript Readout</h2>
              <p>
                WER winner for {selectedLanguageLabel}: {transcriptWinner}. Deepgram is {pct(deepgram?.avg_wer)};
                Databricks is {pct(databricks?.avg_wer)}.
              </p>
            </div>
            <div className="benchmark-tradeoff-card caution">
              <h2>Business Entity Readout</h2>
              <p>
                Critical entity winner for {selectedLanguageLabel}: {entityWinner}. Deepgram is{" "}
                {pct(deepgram?.avg_critical_entity_accuracy)}; Databricks is{" "}
                {pct(databricks?.avg_critical_entity_accuracy)}.
              </p>
            </div>
            <div className="benchmark-tradeoff-card risk">
              <h2>Latency / Stability Readout</h2>
              <p>
                P95 latency winner for {selectedLanguageLabel}: {latencyWinner}. Deepgram is{" "}
                {num(deepgram?.latency_ms?.p95, "ms")}; Databricks is{" "}
                {num(databricks?.latency_ms?.p95, "ms")}.
              </p>
            </div>
          </div>

          <div className="benchmark-panel">
            <div className="benchmark-section-head">
              <div>
                <h2>Failure Examples Worth Reviewing</h2>
                <p>Sorted toward unsafe or high-delta examples so the page explains what to improve next.</p>
              </div>
            </div>
            <div className="benchmark-examples">
              {(data.examples ?? []).map((example) => (
                <ExampleCard key={example.clip_id} example={example} />
              ))}
            </div>
          </div>

          <div className="benchmark-paths">
            <span>Summary: {data.summary_path}</span>
            <span>Deepgram JSONL: {data.deepgram_output}</span>
            <span>Databricks JSONL: {data.databricks_output}</span>
          </div>
        </>
      )}
    </section>
  );
}
