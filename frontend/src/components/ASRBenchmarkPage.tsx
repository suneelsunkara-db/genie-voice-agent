import { useEffect, useState } from "react";
import {
  api,
  ASRBenchmarkLanguageOverview,
  ASRBenchmarkMetricWinner,
  ASRBenchmarkOverviewResponse,
  ASRBenchmarkTierOverview,
} from "../api/client";
import { SentientHCol } from "./sentient/Sentient";

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
    return <span className="sentient-badge">Tie</span>;
  }
  return <span className="sentient-badge">{winner.model_label}</span>;
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
    <div className="sentient-glass sentient-section">
      <h2>Business entity winners</h2>
      <p className="sentient-muted-text">Best model per entity group within each language.</p>
      <div className="sentient-matrix-scroll">
        <div
          className="sentient-matrix"
          style={{ gridTemplateColumns: scoreboardColumns(languages.length) }}
        >
        <div className="head">Entity</div>
        {languages.map((language) => (
          <div className="head" key={language.code}>
            {language.label}
          </div>
        ))}
        {groups.map((group) => (
          <div className="row" key={group}>
            <div className="metric">
              {entityLabels[group]?.label ?? group.split("_").join(" ")}
              <small>{entityLabels[group]?.why ?? "business-critical phrase preservation"}</small>
            </div>
            {languages.map((language) => (
              <div key={`${language.code}-${group}`}>
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
    <div className="sentient-glass sentient-section">
      <h2>Business entity accuracy</h2>
      <p className="sentient-muted-text">
            Per-entity preservation rates across models and languages.
            {hasLegacySource && " Entity columns from legacy holdout where ml_asr clips are not synced locally."}
          </p>
      <div className="sentient-matrix-scroll">
        <div
          className="sentient-matrix"
          style={{ gridTemplateColumns: `minmax(110px, 0.9fr) minmax(180px, 1.8fr) repeat(${groups.length}, minmax(88px, 0.85fr))` }}
        >
        <div className="head">Language</div>
        <div className="head">Model</div>
        {groups.map((group) => (
          <div className="head" key={group}>
            {entityLabels[group]?.label ?? group.split("_").join(" ")}
          </div>
        ))}

        {languages.map((language) => {
          const modelList = Object.values(language.models).sort((a, b) => {
            if (a.provider !== b.provider) return a.provider === "deepgram" ? -1 : 1;
            return a.model_label.localeCompare(b.model_label);
          });
          return modelList.map((model, index) => (
            <div className="row" key={`${language.code}-${model.model_id}-entities`}>
              <div className={index === 0 ? "" : "sentient-muted-text"}>
                {index === 0 ? (
                  <>
                    <strong>{language.label}</strong>
                    <small>{language.code}</small>
                  </>
                ) : null}
              </div>
              <div>
                <span className="sentient-badge">{model.provider}</span>{" "}
                <span>{model.model_label}</span>
                {model.entity_groups_source && (
                  <small className="sentient-muted-text"> {model.entity_groups_source}</small>
                )}
              </div>
              {groups.map((group) => {
                const accuracy = model.entity_groups?.[group]?.accuracy;
                const winner = language.entity_winners?.[group];
                const winnerClass = isEntityWinner(model.model_id, winner) ? "winner" : "";
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
    <div className="sentient-glass sentient-section">
      <h2>Winners by language</h2>
      <p className="sentient-muted-text">Best model per metric in each language.</p>
      <div className="sentient-matrix-scroll">
        <div
          className="sentient-matrix"
          style={{ gridTemplateColumns: scoreboardColumns(languages.length) }}
        >
          <div className="head">Metric</div>
          {languages.map((language) => (
            <div className="head" key={language.code}>
              {language.label}
            </div>
          ))}
          {specs.map((spec) => (
            <div className="row" key={spec.key}>
              <div className="metric">
                {spec.label}
                <small>{spec.lowerIsBetter ? "lower is better" : "higher is better"}</small>
              </div>
              {languages.map((language) => (
                <div key={`${language.code}-${spec.key}`}>
                  <WinnerBadge winner={language.winners[spec.key]} />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      {Object.keys(tierData.scoreboard).length > 0 && (
        <div className="sentient-info-grid">
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
    <div className="sentient-glass sentient-section">
      <h2>All models · all languages</h2>
      <p className="sentient-muted-text">Highlighted cells are best for that metric within the language group.</p>
      <div className="sentient-matrix-scroll">
        <div
          className="sentient-matrix"
          style={{ gridTemplateColumns: `minmax(110px, 0.9fr) minmax(180px, 1.8fr) minmax(56px, 0.6fr) repeat(${specs.length}, minmax(88px, 0.85fr))` }}
        >
        <div className="head">Language</div>
        <div className="head">Model</div>
        <div className="head">Clips</div>
        {specs.map((spec) => (
          <div className="head" key={spec.key}>
            {spec.label}
          </div>
        ))}

        {languages.map((language) => {
          const modelList = Object.values(language.models).sort((a, b) => {
            if (a.provider !== b.provider) return a.provider === "deepgram" ? -1 : 1;
            return a.model_label.localeCompare(b.model_label);
          });
          return modelList.map((model, index) => (
            <div className="row" key={`${language.code}-${model.model_id}`}>
              <div className={index === 0 ? "" : "sentient-muted-text"}>
                {index === 0 ? (
                  <>
                    <strong>{language.label}</strong>
                    <small>{language.code}</small>
                  </>
                ) : null}
              </div>
              <div>
                <span className="sentient-badge">{model.provider}</span>{" "}
                <span>{model.model_label}</span>
              </div>
              <div>{model.clips}</div>
              {specs.map((spec) => {
                const winner = language.winners[spec.key];
                const value = model[spec.key];
                const winnerClass = isWinner(model.model_id, spec.key, winner) ? "winner" : "";
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
    <div className="sentient-glass sentient-section">
      <strong>Entity columns use legacy holdout data</strong>
      <p className="sentient-muted-text">
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
    <section className="sentient-section">
      <div className="sentient-kicker">{tier === "business" ? "Business tier" : "Acoustic tier"}</div>
      <h2>{tier === "business" ? "Entity readiness + transcript quality" : "Read-speech WER/CER"}</h2>
      <p className="sentient-muted-text">Dataset: {tierData.dataset_id}</p>
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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
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
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const dataSource = data?.source ?? "legacy";

  return (
    <div className="sentient-h-flow sentient-h-flow-benchmark">
      <SentientHCol
        step={1}
        title="ASR benchmark"
        description="Holistic model comparison"
      >
        {loading && !data && !err && (
          <p className="sentient-muted-text">Loading…</p>
        )}
        {err && <div className="error">ASR benchmark API error: {err}</div>}
        {data && !data.available && (
          <div className="sentient-glass sentient-h-panel">
            <strong>No benchmark results found.</strong>
            <p className="sentient-muted-text">{data.message}</p>
          </div>
        )}
        {data?.available && (
          <div className="sentient-info-grid sentient-info-grid-tight">
            <div>
              <strong>Data source</strong>
              <span>
                {dataSource === "ml_asr" ? "ml_asr FLEURS smoke" : "Legacy holdout"}
              </span>
            </div>
            <div>
              <strong>Index</strong>
              <span>{shortPath(data.index_path)}</span>
            </div>
          </div>
        )}
      </SentientHCol>

      {data?.available && data.tiers?.business && (
        <SentientHCol step={2} title="Business tier" description="Entity + transcript quality">
          <TierSection tier="business" tierData={data.tiers.business} />
        </SentientHCol>
      )}

      {data?.available && data.tiers?.acoustic && (
        <SentientHCol step={3} title="Acoustic tier" description="WER / CER / latency">
          <TierSection tier="acoustic" tierData={data.tiers.acoustic} />
        </SentientHCol>
      )}
    </div>
  );
}
