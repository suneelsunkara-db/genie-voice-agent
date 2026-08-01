import { useEffect, useState } from "react";
import { useAppLanguage } from "../lib/appLanguage";
import { getMe } from "../lib/me";
import { DEFAULT_LANGUAGE_OPTIONS, Lang, fetchSupportedLanguages } from "../lib/languages";
import { LanguageBar } from "./LanguageBar";
import { BrandLockup } from "./BrandLockup";
import { VoiceBackdrop } from "./VoiceBackdrop";
import { VoiceOrb } from "./VoiceOrb";
import "../styles/home.css";

/**
 * Landing page for "Genie Assisted Voice" (a Databricks demo).
 *
 * This is the ONE place a language is chosen. The picker writes the global app
 * language (lib/appLanguage); every use-case surface then opens its voice session
 * in that language from the first greeting and renders its own picker locked. The
 * user picks a language here, then clicks an industry to enter — there is no voice
 * concierge on this page, so short spoken cues can't be mis-detected and routing
 * has nothing to get wrong.
 */

type Industry = {
  id: "telco" | "fsi" | "healthcare";
  hash: string;
  title: string;
  tag: string;
  blurb: string;
};

const INDUSTRIES: Industry[] = [
  {
    id: "telco",
    hash: "#/telco",
    title: "Telco",
    tag: "Billing Support",
    blurb: "Resolve charges, waive fees, and set up payment plans on a live call.",
  },
  {
    id: "fsi",
    hash: "#/card",
    title: "Financial Services",
    tag: "Credit-Card Assistant",
    blurb: "Understand statements and rewards, with deep “why” reasoning on demand.",
  },
  {
    id: "healthcare",
    hash: "#/hls",
    title: "Healthcare",
    tag: "Care & Claims",
    blurb: "Explain claims, coverage, and visit summaries in plain language.",
  },
];

const ONTOLOGY_NODES = [
  { id: "customer", label: "Customer", x: 50, y: 18 },
  { id: "account", label: "Account", x: 18, y: 46 },
  { id: "statement", label: "Statement", x: 50, y: 50 },
  { id: "claim", label: "Claim", x: 82, y: 46 },
  { id: "charge", label: "Charge", x: 34, y: 82 },
  { id: "reward", label: "Reward", x: 66, y: 82 },
];
const ONTOLOGY_EDGES: Array<[string, string]> = [
  ["customer", "account"],
  ["customer", "statement"],
  ["customer", "claim"],
  ["account", "charge"],
  ["statement", "charge"],
  ["statement", "reward"],
];

// A concrete illustrative "why" investigation — this is what DEEP reasoning does
// that a scripted bot can't: pull governed history, compare, isolate drivers,
// and explain with evidence. (Illustrative content; the live version runs on the
// card deep-dive page.)
const REASONING_QUESTION = "“Why is my bill higher this month?”";
const REASONING_HOPS = [
  "Pulls 6 months of billing history",
  "Compares charges line-by-line vs. the usual baseline",
  "Isolates what actually changed this cycle",
];
const REASONING_CONCLUSION =
  "$41 of the $47 increase is a one-time device fee — next cycle returns to normal.";

export function HomePage() {
  const [userName, setUserName] = useState<string>("");
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  const [language, setLanguage] = useAppLanguage();

  // Prefetch the signed-in user + the config-driven supported language set.
  useEffect(() => {
    let active = true;
    void getMe().then((me) => {
      if (active && me.name) setUserName(me.name);
    });
    void (async () => {
      const langs = await fetchSupportedLanguages();
      if (active && langs.length) setLangOptions(langs);
    })();
    return () => {
      active = false;
    };
  }, []);

  const go = (hash: string) => {
    window.location.hash = hash;
  };

  return (
    <div className="home-root">
      <VoiceBackdrop />

      <header className="home-top">
        <BrandLockup product="Assisted Voice" />
        <div className="home-topright">
          {/* The single language selector for the whole app. Every use-case page
              inherits this choice and locks its own picker. */}
          <LanguageBar
            value={language}
            options={langOptions}
            onChange={setLanguage}
            label="Language"
          />
          <nav className="home-topnav">
            <a href="#/voice-benchmarks">Benchmarks</a>
            <a href="#/traces">Traces</a>
            <a href="#/guardrails">Guardrails</a>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <section className="home-hero">
          <VoiceOrb state="idle" level={0} size="clamp(74px, 11vh, 112px)" ariaLabel="Genie" disabled />

          <h1 className="home-title">
            {userName ? (
              <>Welcome back, {userName}.</>
            ) : (
              <>Welcome to Databricks Genie Assisted Voice.</>
            )}
          </h1>
          <p className="home-sub">
            One voice platform, {langOptions.length}{" "}
            {langOptions.length === 1 ? "language" : "languages"}, three industries — powered by the
            Databricks Genie ontology and deep reasoning.
          </p>

          <p className="home-instruction">
            Choose your language above, then open an experience below.
          </p>
        </section>

        <section className="home-industries">
          {INDUSTRIES.map((ind) => (
            <button
              key={ind.id}
              type="button"
              className={`home-card home-card-${ind.id}`}
              onClick={() => go(ind.hash)}
            >
              <div className="home-card-tag">{ind.tag}</div>
              <div className="home-card-title">{ind.title}</div>
              <div className="home-card-blurb">{ind.blurb}</div>
              <div className="home-card-go">Open →</div>
            </button>
          ))}
        </section>

        <section className="home-genie">
          <OntologyPanel />
          <ReasoningPanel />
        </section>
      </main>

      <footer className="home-foot">
        Powered by Databricks Genie · Unity Catalog governed · Realtime API built on OSS voice models
      </footer>
    </div>
  );
}

function OntologyPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">Genie Ontology</div>
      <div className="home-panel-sub">
        A governed semantic model — the entities and relationships Genie understands.
      </div>
      <svg className="home-ontology" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {ONTOLOGY_EDGES.map(([a, b], i) => {
          const na = ONTOLOGY_NODES.find((n) => n.id === a)!;
          const nb = ONTOLOGY_NODES.find((n) => n.id === b)!;
          return (
            <line
              key={i}
              x1={na.x}
              y1={na.y}
              x2={nb.x}
              y2={nb.y}
              className="home-ontology-edge"
              style={{ animationDelay: `${i * 0.25}s` }}
            />
          );
        })}
        {ONTOLOGY_NODES.map((n) => (
          <g key={n.id} className="home-ontology-node">
            <circle cx={n.x} cy={n.y} r={3.4} />
            <text x={n.x} y={n.y - 5} textAnchor="middle">
              {n.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function ReasoningPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">Genie Deep Reasoning</div>
      <div className="home-panel-sub">
        Beyond quick answers — Genie investigates the “why”, grounded in your own data.
      </div>
      <div className="home-dr">
        <div className="home-dr-q">{REASONING_QUESTION}</div>
        <ol className="home-dr-hops">
          {REASONING_HOPS.map((hop, i) => (
            <li key={hop} style={{ animationDelay: `${0.4 + i * 0.7}s` }}>
              <span className="home-dr-dot" />
              {hop}
            </li>
          ))}
        </ol>
        <div className="home-dr-answer" style={{ animationDelay: `${0.4 + REASONING_HOPS.length * 0.7}s` }}>
          {REASONING_CONCLUSION}
        </div>
        <div className="home-dr-foot">Grounded in governed data · with citations</div>
      </div>
    </div>
  );
}
