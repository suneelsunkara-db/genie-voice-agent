import { useEffect, useState } from "react";
import {
  api,
  ASRBenchmarkLanguageOverview,
  ASRBenchmarkMetricWinner,
  ASRBenchmarkOverviewResponse,
  ASRBenchmarkTierOverview,
} from "../api/client";

type BenchmarkTier = "business" | "acoustic";
type MetricKey = "wer" | "cer" | "critical_entity_accuracy" | "unsafe_for_resolution_rate" | "p95_latency_ms";

const LANGUAGE_ORDER = ["en-US", "th-TH", "id-ID", "zh-CN"];

const METRIC_SPECS: Record<
  BenchmarkTier,
  Array<{ key: MetricKey; label: string; lowerIsBetter: boolean }>
> = {
  business: [
    { key: "wer", label: "WER", lowerIsBetter: true },
    { key: "cer", label: "CER", lowerIsBetter: true },
    { key: "critical_entity_accuracy", label: "Critical entities", lowerIsBetter: false },
    { key: "unsafe_for_resolution_rate", label: "Unsafe", lowerIsBetter: true },
    { key: "p95_latency_ms", label: "P95 latency", lowerIsBetter: true },
  ],
  acoustic: [
    { key: "wer", label: "WER", lowerIsBetter: true },
    { key: "cer", label: "CER", lowerIsBetter: true },
    { key: "p95_latency_ms", label: "P95 latency", lowerIsBetter: true },
  ],
};

function pct(value?: number | null) {
  if (value === null || value === undefined) return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

function ms(value?: number | null) {
  if (value === null || value === undefined) return "n/a";
  if (value >= 60000) return `${Math.round(value / 1000)}s`;
  return `${Math.round(value)} ms`;
}

function formatMetric(key: MetricKey, value?: number | null) {
  if (key === "p95_latency_ms") return ms(value);
  return pct(value);
}

function shortPath(path?: string) {
  if (!path) return "n/a";
  return path.split("/").slice(-3).join("/");
}

function isWinner(modelId: string, _metric: MetricKey, winner?: ASRBenchmarkMetricWinner) {
  return Boolean(winner && !winner.tie && winner.model_id === modelId);
}

function WinnerBadge({ winner }: { winner?: ASRBenchmarkMetricWinner }) {
  if (!winner) return null;
  if (winner.tie) {
    return <span className="benchmark-winner tie">Tie</span>;
  }
  return (
    <span className={`benchmark-winner ${winner.provider ?? "tie"}`}>
      {winner.model_label}
    </span>
  );
}

const entityLabels: Record<string, { label: string; why: string }> = {
  invoice_ids: { label: "Invoice IDs", why: "Wrong invoice can trigger the wrong billing action" },
  amounts: { label: "Dollar amounts", why: "Used in agent explanations and adjustment checks" },
  dates: { label: "Dates", why: "Payment timing and due-date context" },
  billing_actions: { label: "Billing actions", why: "Waiver, payment-plan, refund language" },
  confirmations: { label: "Confirmations", why: "Controls whether the app can safely close" },
  refusals: { label: "Refusals / negation", why: "Prevents acting against customer intent" },
  account_terms: { label: "Account terms", why: "Billing vocabulary preservation" },
};

function entityGroupsForLanguages(languages: ASRBenchmarkLanguageOverview[]) {
  return Array.from(
    new Set(
      languages.flatMap((language) =>
        Object.values(language.models).flatMap((model) =>
          Object.entries(model.entity_groups ?? {})
            .filter(([, stats]) => (stats?.expected ?? 0) > 0)
            .map(([group]) => group),
        ),
      ),
    ),
  ).sort();
}

function scoreboardColumns(languageCount: number) {
  return `minmax(160px, 1.4fr) repeat(${languageCount}, minmax(130px, 1fr))`;
}

function isEntityWinner(modelId: string, winner?: ASRBenchmarkMetricWinner) {
  return Boolean(winner && !winner.tie && winner.model_id === modelId);
}

function EntityWinnersByLanguage({ languages }: { languages: ASRBenchmarkLanguageOverview[] }) {
  const groups = entityGroupsForLanguages(languages);
  if (!groups.length) return null;

  return (
    <div className="benchmark-panel">
      <div className="benchmark-section-head">
        <div>
          <h2>Business entity winners</h2>
          <p>Best model per entity group within each language. Higher accuracy is better.</p>
        </div>
      </div>
      <div className="benchmark-matrix-scroll">
        <div
          className="benchmark-matrix-table scoreboard"
          style={{ gridTemplateColumns: scoreboardColumns(languages.length) }}
        >
        <div className="benchmark-table-head">Entity</div>
        {languages.map((language) => (
          <div className="benchmark-table-head" key={language.code}>
            {language.label}
          </div>
        ))}
        {groups.map((group) => (
          <div className="benchmark-table-row" key={group}>
            <div className="benchmark-metric-name">
              {entityLabels[group]?.label ?? group.split("_").join(" ")}
              <small>{entityLabels[group]?.why ?? "business-critical phrase preservation"}</small>
            </div>
            {languages.map((language) => (
              <div className="benchmark-scoreboard-cell" key={`${language.code}-${group}`}>
                <WinnerBadge winner={language.entity_winners?.[group]} />
              </div>
            ))}
          </div>
        ))}
        </div>
      </div>
    </div>
  );
}

function EntityAccuracyMatrix({ languages }: { languages: ASRBenchmarkLanguageOverview[] }) {
  const groups = entityGroupsForLanguages(languages);
  if (!groups.length) return null;

  const hasLegacySource = languages.some((language) =>
    Object.values(language.models).some((model) => model.entity_groups_source === "legacy_holdout")
  );

  return (
    <div className="benchmark-panel">
      <div className="benchmark-section-head">
        <div>
          <h2>Business entity accuracy</h2>
          <p>
            Per-entity preservation rates across models and languages.
            {hasLegacySource && " Entity columns from legacy multilingual_gold holdout where ml_asr clip results are not synced locally."}
          </p>
        </div>
      </div>
      <div className="benchmark-matrix-scroll">
        <div
          className="benchmark-matrix-table holistic entity-matrix"
          style={{ gridTemplateColumns: `minmax(110px, 0.9fr) minmax(180px, 1.8fr) repeat(${groups.length}, minmax(88px, 0.85fr))` }}
        >
        <div className="benchmark-table-head">Language</div>
        <div className="benchmark-table-head">Model</div>
        {groups.map((group) => (
          <div className="benchmark-table-head" key={group}>
            {entityLabels[group]?.label ?? group.split("_").join(" ")}
          </div>
        ))}

        {languages.map((language) => {
          const modelList = Object.values(language.models).sort((a, b) => {
            if (a.provider !== b.provider) return a.provider === "deepgram" ? -1 : 1;
            return a.model_label.localeCompare(b.model_label);
          });
          return modelList.map((model, index) => (
            <div className="benchmark-table-row" key={`${language.code}-${model.model_id}-entities`}>
              <div className={index === 0 ? "benchmark-language-cell" : "benchmark-language-cell muted"}>
                {index === 0 ? (
                  <>
                    <strong>{language.label}</strong>
                    <small>{language.code}</small>
                  </>
                ) : null}
              </div>
              <div className="benchmark-model-cell">
                <span className={`benchmark-provider-tag ${model.provider}`}>{model.provider}</span>
                <span>{model.model_label}</span>
                {model.entity_groups_source && (
                  <small className="benchmark-entity-source">{model.entity_groups_source}</small>
                )}
              </div>
              {groups.map((group) => {
                const accuracy = model.entity_groups?.[group]?.accuracy;
                const winner = language.entity_winners?.[group];
                const winnerClass = isEntityWinner(model.model_id, winner) ? `winner ${model.provider}` : "";
                return (
                  <div key={group} className={winnerClass}>
                    {pct(accuracy)}
                  </div>
                );
              })}
            </div>
          ));
        })}
        </div>
      </div>
    </div>
  );
}

function Scoreboard({
  tier,
  tierData,
  languages,
}: {
  tier: BenchmarkTier;
  tierData: ASRBenchmarkTierOverview;
  languages: ASRBenchmarkLanguageOverview[];
}) {
  const specs = METRIC_SPECS[tier];
  return (
    <div className="benchmark-panel">
      <div className="benchmark-section-head">
        <div>
          <h2>Winners by language</h2>
          <p>Best model per metric in each language. Green = Databricks route, amber = Deepgram Nova-3.</p>
        </div>
      </div>
      <div className="benchmark-matrix-scroll">
        <div
          className="benchmark-matrix-table scoreboard"
          style={{ gridTemplateColumns: scoreboardColumns(languages.length) }}
        >
          <div className="benchmark-table-head">Metric</div>
          {languages.map((language) => (
            <div className="benchmark-table-head" key={language.code}>
              {language.label}
            </div>
          ))}
          {specs.map((spec) => (
            <div className="benchmark-table-row" key={spec.key}>
              <div className="benchmark-metric-name">
                {spec.label}
                <small>{spec.lowerIsBetter ? "lower is better" : "higher is better"}</small>
              </div>
              {languages.map((language) => (
                <div className="benchmark-scoreboard-cell" key={`${language.code}-${spec.key}`}>
                  <WinnerBadge winner={language.winners[spec.key]} />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      {Object.keys(tierData.scoreboard).length > 0 && (
        <div className="benchmark-scoreboard-totals">
          {specs.map((spec) => {
            const counts = tierData.scoreboard[spec.key] ?? [];
            if (!counts.length) return null;
            return (
              <div key={spec.key}>
                <strong>{spec.label}</strong>
                <span>
                  {counts
                    .map((entry) => {
                      const label =
                        languages
                          .flatMap((language) => Object.values(language.models))
                          .find((model) => model.model_id === entry.model_id)?.model_label ?? entry.model_id;
                      return `${label} (${entry.wins})`;
                    })
                    .join(" · ")}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function HolisticMatrix({
  tier,
  languages,
}: {
  tier: BenchmarkTier;
  languages: ASRBenchmarkLanguageOverview[];
}) {
  const specs = METRIC_SPECS[tier];

  return (
    <div className="benchmark-panel">
      <div className="benchmark-section-head">
        <div>
          <h2>All models · all languages</h2>
          <p>Highlighted cells are best for that metric within the language group.</p>
        </div>
      </div>
      <div className="benchmark-matrix-scroll">
        <div
          className="benchmark-matrix-table holistic"
          style={{ gridTemplateColumns: `minmax(110px, 0.9fr) minmax(180px, 1.8fr) minmax(56px, 0.6fr) repeat(${specs.length}, minmax(88px, 0.85fr))` }}
        >
        <div className="benchmark-table-head">Language</div>
        <div className="benchmark-table-head">Model</div>
        <div className="benchmark-table-head">Clips</div>
        {specs.map((spec) => (
          <div className="benchmark-table-head" key={spec.key}>
            {spec.label}
          </div>
        ))}

        {languages.map((language) => {
          const modelList = Object.values(language.models).sort((a, b) => {
            if (a.provider !== b.provider) return a.provider === "deepgram" ? -1 : 1;
            return a.model_label.localeCompare(b.model_label);
          });
          return modelList.map((model, index) => (
            <div className="benchmark-table-row" key={`${language.code}-${model.model_id}`}>
              <div className={index === 0 ? "benchmark-language-cell" : "benchmark-language-cell muted"}>
                {index === 0 ? (
                  <>
                    <strong>{language.label}</strong>
                    <small>{language.code}</small>
                  </>
                ) : null}
              </div>
              <div className="benchmark-model-cell">
                <span className={`benchmark-provider-tag ${model.provider}`}>{model.provider}</span>
                <span>{model.model_label}</span>
              </div>
              <div>{model.clips}</div>
              {specs.map((spec) => {
                const winner = language.winners[spec.key];
                const value = model[spec.key];
                const winnerClass = isWinner(model.model_id, spec.key, winner) ? `winner ${model.provider}` : "";
                return (
                  <div key={spec.key} className={winnerClass}>
                    {formatMetric(spec.key, value)}
                  </div>
                );
              })}
            </div>
          ));
        })}
        </div>
      </div>
    </div>
  );
}

function MixedEntityDataNotice({ languages }: { languages: ASRBenchmarkLanguageOverview[] }) {
  const usesLegacyEntities = languages.some((language) =>
    Object.values(language.models).some((model) => model.entity_groups_source === "legacy_holdout"),
  );
  if (!usesLegacyEntities) return null;

  return (
    <div className="benchmark-mixed-source">
      <strong>Entity columns use legacy holdout data</strong>
      <p>
        Aggregate WER/CER/latency above come from the current ml_asr FLEURS smoke eval (small clip counts).
        Per-entity accuracy below is filled from packaged multilingual_gold holdout (~420 clips/lang) until local
        ml_asr results.jsonl files are synced.
      </p>
    </div>
  );
}

function TierSection({
  tier,
  tierData,
}: {
  tier: BenchmarkTier;
  tierData: ASRBenchmarkTierOverview;
}) {
  const languages = LANGUAGE_ORDER.map((code) => tierData.languages[code]).filter(Boolean);
  if (!languages.length) return null;

  return (
    <section className="benchmark-tier-section">
      <div className="benchmark-tier-head">
        <div>
          <div className="eyebrow">{tier === "business" ? "Business tier" : "Acoustic tier"}</div>
          <h2>{tier === "business" ? "Entity readiness + transcript quality" : "Read-speech WER/CER"}</h2>
          <p>Dataset: {tierData.dataset_id}</p>
        </div>
      </div>
      <Scoreboard tier={tier} tierData={tierData} languages={languages} />
      {tier === "business" && (
        <>
          <MixedEntityDataNotice languages={languages} />
          <EntityWinnersByLanguage languages={languages} />
          <EntityAccuracyMatrix languages={languages} />
        </>
      )}
      <HolisticMatrix tier={tier} languages={languages} />
    </section>
  );
}

export function ASRBenchmarkPage() {
  const [data, setData] = useState<ASRBenchmarkOverviewResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setData(null);
    setErr(null);
    api
      .asrBenchmarkOverview("auto")
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

  const dataSource = data?.source ?? "legacy";

  return (
    <section className="benchmark-page benchmark-page-holistic">
      <div className="benchmark-page-head">
        <div>
          <div className="eyebrow">ASR Model Evaluation</div>
          <h1>Holistic ASR benchmark</h1>
          <p>
            All languages and models side by side. Highlighted cells show the best route per metric; badges name the
            winning Deepgram or Databricks model.
          </p>
        </div>
        <div className="benchmark-head-actions">
          <a className="benchmark-back-link" href="#/">Back to cockpit</a>
        </div>
      </div>

      {err && <div className="error">ASR benchmark API error: {err}</div>}
      {data && !data.available && (
        <div className="benchmark-empty">
          <strong>No benchmark results found.</strong>
          <p>{data.message}</p>
          <code>./scripts/ml_asr.sh eval && ./scripts/ml_asr/05_eval.sh sync-index</code>
        </div>
      )}

      {data?.available && data.tiers && (
        <>
          <div className="benchmark-reading-guide">
            <div>
              <strong>Data source</strong>
              <span>
                {dataSource === "ml_asr"
                  ? "ml_asr FLEURS smoke eval (config/ml_asr_eval.yaml)"
                  : "Legacy multilingual_gold holdout (~420 clips/lang)"}
              </span>
            </div>
            <div>
              <strong>Index</strong>
              <span>{shortPath(data.index_path)}</span>
            </div>
          </div>

          <div className="benchmark-glossary">
            <div>
              <strong>WER / CER</strong>
              <span>Transcript error rates — lower is better.</span>
            </div>
            <div>
              <strong>Critical entities</strong>
              <span>Invoice IDs, amounts, confirmations preserved — higher is better (business tier).</span>
            </div>
            <div>
              <strong>Unsafe</strong>
              <span>Transcripts missing signals needed for auto-resolution — lower is better.</span>
            </div>
            <div>
              <strong>P95 latency</strong>
              <span>Post-stop transcription time — lower is better.</span>
            </div>
          </div>

          {data.tiers.business && <TierSection tier="business" tierData={data.tiers.business} />}
          {data.tiers.acoustic && <TierSection tier="acoustic" tierData={data.tiers.acoustic} />}
        </>
      )}
    </section>
  );
}
