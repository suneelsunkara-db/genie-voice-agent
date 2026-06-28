import { useEffect, useState } from "react";
import { api, ASRBenchmarkExample, ASRBenchmarkResponse, ASRProviderSummary } from "../api/client";

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
  const [data, setData] = useState<ASRBenchmarkResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    api
      .asrBenchmark()
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
  }, []);

  const deepgram = data?.summary?.providers?.deepgram;
  const databricks = data?.summary?.providers?.databricks;
  const promotion = data?.summary?.promotion_read;
  const clipCount = databricks?.clips ?? deepgram?.clips ?? promotion?.paired_clips;
  const latencyPenalty = deltaMs(databricks?.latency_ms?.p95, deepgram?.latency_ms?.p95);

  return (
    <section className="benchmark-page">
      <div className="benchmark-page-head">
        <div>
          <div className="eyebrow">ASR Model Evaluation</div>
          <h1>Deepgram vs Fine-Tuned Databricks Whisper</h1>
          <p>
            Offline 403-clip voice-model benchmark focused on what this app needs: accurate final
            utterances, billing entity preservation, safe resolution signals, and latency after mic stop.
          </p>
        </div>
        <a className="benchmark-back-link" href="#/">Back to cockpit</a>
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
              <h2>Mixed result: Databricks is stronger on accuracy, Deepgram is stronger on latency.</h2>
              <p>
                The full eval does not produce a single universal winner. Databricks has lower transcript error
                and better business entity accuracy, while Deepgram is materially faster and remains the better
                real-time streaming experience.
              </p>
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
              <h2>Where Databricks is stronger</h2>
              <p>
                Lower WER, higher entity accuracy, no empty transcripts in this run, and better preservation
                for amounts, dates, billing actions, confirmations, and refusals.
              </p>
            </div>
            <div className="benchmark-tradeoff-card caution">
              <h2>Where Deepgram is still stronger</h2>
              <p>
                Much lower latency and mature streaming UX. The Databricks model is currently best treated as
                push-to-talk or stop-to-transcribe.
              </p>
            </div>
            <div className="benchmark-tradeoff-card risk">
              <h2>Main remaining risk</h2>
              <p>
                Invoice IDs remain hard for both models. Databricks improved the aggregate result, but invoice
                ID misses still drive unsafe-resolution flags.
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
