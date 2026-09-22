const REPO = "https://github.com/suneelsunkara-db/genie-voice-agent";

const commit = (sha, label) =>
  `<a href="${REPO}/commit/${sha}" target="_blank" rel="noreferrer">${label}</a>`;

const footer = (index, source = "Grounded in “Genie for Voice Use Cases” + repository Git history") => `
  <div class="source-line">
    <span class="wordmark">Databricks</span>
    <span>${source}</span>
    <span>${String(index + 1).padStart(2, "0")}</span>
  </div>
`;

const slides = [
  {
    title: "Genie for Voice Use Cases",
    className: "dark",
    accent: "#ff3621",
    notes:
      "This is the deck's own opening. The goal we set out to achieve: power Genie's ontology — its governed business reasoning — with voice, and especially with non-English voice, using open-source models hosted on Databricks. Audience is GTM / FDE.",
    html: `
      <div class="slide-inner hero">
        <p class="eyebrow">Powering Genie's ontology with voice models</p>
        <h1>Genie for Voice Use Cases</h1>
        <p class="lede">Give Genie's governed reasoning a real-time, multilingual voice — built on open-source models, hosted on Databricks.</p>
        <p class="meta">Suneel Sunkara · Senior Partner SA · Aug 2026 · Audience: GTM / FDE</p>
        <svg class="hero-mark" viewBox="0 0 220 220" aria-hidden="true">
          <path d="M35 112c20-63 44-63 60 0s40 63 58 0 34-63 44 0" />
          <path d="M55 147c15-38 30-38 44 0s31 38 45 0" />
        </svg>
      </div>
    `,
  },
  {
    title: "The problem we set out to solve",
    className: "",
    accent: "#ff3621",
    notes:
      "The problem statement from the agenda: Genie is powerful over governed data in text, but silent on a live call — and especially unavailable in non-English. The bet was to power Genie's ontology with voice models, starting with the languages closed vendors serve least well.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Problem statement</p>
        <p class="quote">Genie reasons over your governed data.<br><strong>On a call, it has no voice — least of all in non-English.</strong></p>
        <div class="callout">The goal: power Genie's ontology with voice models, so its business reasoning is reachable live, multilingually, and under Databricks governance.</div>
      </div>
    `,
  },
  {
    title: "Voice just became the interface",
    className: "",
    accent: "#f2a900",
    notes:
      "Market context slide. OpenAI shipped GPT-Live voice into the desktop app and Codex; Anthropic plugged Claude's voice into its smarter models and connected apps. Voice is becoming the primary interface to intelligence — which is exactly why Genie needs one.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Why now</p>
        <h2>OpenAI and Anthropic shipped voice upgrades.</h2>
        <div class="grid-2">
          <article class="card" style="--card-accent:#00a972">
            <h3>OpenAI</h3>
            <p>GPT-Live voice landed in the desktop app and inside Codex — voice as a first-class developer and product surface.</p>
          </article>
          <article class="card" style="--card-accent:#f2a900">
            <h3>Anthropic</h3>
            <p>Claude's voice was plugged into its smarter models and connected apps.</p>
          </article>
        </div>
        <div class="callout">Voice is becoming the interface to intelligence. The question is which intelligence answers — a generic assistant, or Genie over your governed data.</div>
        ${footer(2)}
      </div>
    `,
  },
  {
    title: "Existing solutions — and the gap",
    className: "",
    accent: "#2272b4",
    notes:
      "This is the crux of the deck. Deepgram, ElevenLabs, OpenAI Whisper API and Gemini/Vertex do voice extremely well. What none of them do: a realtime STT/TTS API over OSS models you control, tool-calling into the Genie API and Genie Agent Mode, and reaching Genie One's deep business context. That gap is the whole opportunity.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Today's tools &amp; gaps</p>
        <h2>Great voice vendors. None can reach Genie.</h2>
        <div class="grid-2">
          <article class="card" style="--card-accent:#00a972">
            <h3>What they do well</h3>
            <p>Streaming &amp; real-time · multi-lingual · pauses, redaction, word timings · HD voices and custom tones.</p>
            <p style="margin-top:.6rem;font-size:.9em;">Deepgram · ElevenLabs · OpenAI Whisper API · Gemini / Vertex</p>
          </article>
          <article class="card" style="--card-accent:#ff3621">
            <h3>What they can't support</h3>
            <p>Realtime STT/TTS API on <strong>OSS models you host</strong>.</p>
            <p>Tool-calling into the <strong>Genie API &amp; Genie Agent Mode</strong>.</p>
            <p><strong>Genie One's</strong> deep business context and reasoning.</p>
          </article>
        </div>
        ${footer(3, "Deck: Existing Solutions &amp; Gaps")}
      </div>
    `,
  },
  {
    title: "The bet: Genie powered by voice models",
    className: "",
    accent: "#00a972",
    notes:
      "This is what we were actually trying to build. Not a voice bot — Genie with a voice. Five differentiators straight from the deck: deeper reasoning via Genie Agent Mode, 24-language multilingual support, live insights and richer context, Genie One workspace-level insights, all Databricks-hosted so it stays governed and sovereign.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Proposed approach</p>
        <h2>Genie powered by voice models.</h2>
        <div class="grid-3">
          <article class="card" style="--card-accent:#ff3621"><h3>Deeper reasoning</h3><p>Genie Agent Mode for causal, multi-step business questions.</p></article>
          <article class="card" style="--card-accent:#f2a900"><h3>24 languages</h3><p>Multilingual support, with a focus on the non-English coverage vendors under-serve.</p></article>
          <article class="card" style="--card-accent:#2272b4"><h3>Live insights</h3><p>Richer, account-aware context surfaced during the conversation.</p></article>
          <article class="card" style="--card-accent:#7d59a5"><h3>Genie One</h3><p>Workspace-level insights across your governed data.</p></article>
          <article class="card" style="--card-accent:#00a972"><h3>Databricks-hosted</h3><p>Open models on your platform — governed, sovereign, no data leaving.</p></article>
          <article class="card" style="--card-accent:#28a9c7"><h3>Tool-callable</h3><p>The voice loop calls the Genie API directly — the thing vendors can't do.</p></article>
        </div>
        ${footer(4, "Deck: Genie Powered by Voice Models")}
      </div>
    `,
  },
  {
    title: "Why OSS models on Databricks",
    className: "",
    accent: "#f2a900",
    notes:
      "The engineering choice behind the bet. Qwen3-ASR-1.7B for STT and openbmb/VoxCPM2 for TTS, both on Databricks Model Serving, covering 24 languages. Streaming TTS emits ~80ms PCM slices over SSE so audio starts mid-sentence. Silero VAD gives semantic end-of-turn. The LLM runs in a background thread to hide Agent Mode / deep-reasoning latency.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Model serving</p>
        <h2>Open models, hosted and streamed.</h2>
        <div class="grid-2">
          <article class="card" style="--card-accent:#f2a900">
            <h3>The models</h3>
            <p><strong>Qwen3-ASR-1.7B</strong> for speech-to-text and <strong>openbmb/VoxCPM2</strong> for text-to-speech — 24 languages, served on Databricks Model Serving.</p>
          </article>
          <article class="card" style="--card-accent:#00a972">
            <h3>The techniques</h3>
            <p><strong>Streaming TTS</strong> — VoxCPM2 emits ~80ms PCM slices over SSE; audio plays mid-sentence.</p>
            <p><strong>Semantic end-of-turn</strong> — Silero VAD detects real pauses.</p>
            <p><strong>Latency-hiding turn</strong> — the LLM runs in a background thread for Agent Mode / deep reasoning.</p>
          </article>
        </div>
        ${footer(5, "Deck: Model Serving · Qwen3-ASR + VoxCPM2")}
      </div>
    `,
  },
  {
    title: "How we built it: June → September",
    className: "",
    accent: "#ff3621",
    notes:
      "Now the timeline you asked for — mapped to the deck's goals, not a generic journey. June proved Genie grounding on a call. July built the realtime OSS voice loop and multilingual coverage. August exposed it as go/genie-for-voice with a Realtime API and MCP. September made it governed with AI Gateway and tracing.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">The build, mapped to the deck</p>
        <h2>Four months toward one goal.</h2>
        <div class="timeline">
          <article class="timeline-month" style="--month-color:#ff3621">
            <h3>June</h3>
            <p><strong>Ground Genie on the call.</strong><br>Live assist, resolution journey, honest Genie grounding, first ASR on Databricks.</p>
            <span class="commit-count">Problem → grounding</span>
          </article>
          <article class="timeline-month" style="--month-color:#f2a900">
            <h3>July</h3>
            <p><strong>Build the OSS voice loop.</strong><br>Realtime API, Qwen3-ASR + VoxCPM2, 24 languages, semantic end-of-turn, streaming TTS.</p>
            <span class="commit-count">Approach → runtime</span>
          </article>
          <article class="timeline-month" style="--month-color:#2272b4">
            <h3>August</h3>
            <p><strong>Ship the interface.</strong><br>go/genie-for-voice, Realtime API + MCP, Genie One, localized knowledge.</p>
            <span class="commit-count">Assets built</span>
          </article>
          <article class="timeline-month" style="--month-color:#00a972">
            <h3>September</h3>
            <p><strong>Make it governed.</strong><br>AI Gateway, fail-closed guardrails, OBO identity, end-to-end tracing.</p>
            <span class="commit-count">Guardrails → trust</span>
          </article>
        </div>
        ${footer(6, `Repo created 24 Jun · <a href="${REPO}/commits/main" target="_blank" rel="noreferrer">95 commits by 21 Sep 2026</a>`)}
      </div>
    `,
  },
  {
    title: "Realtime API architecture",
    className: "",
    accent: "#f2a900",
    notes:
      "Architect view of the deck's Realtime API Architecture slide. The live turn path is a WebSocket pipeline: browser PCM → STT → LLM with Genie tools → TTS → streamed audio. The Realtime API layer — not the models — owns VAD/endpointing, turn state, barge-in and language. The honest constraint: Databricks Model Serving does not yet support true real-time inference, so the API hides that with streaming and background LLM work. Governed serving sits on the rail below: AI Gateway for FM traffic, Model Serving for STT/TTS agent endpoints, the Genie API, and Lakebase.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Realtime API architecture</p>
        <h2>The API owns the turn; the platform serves it.</h2>
        <p class="section-note">A single WebSocket carries one voice turn end to end. The Realtime API layer owns turn-taking, barge-in and language so the model endpoints can stay stateless.</p>
        <div class="pipeline">
          <article class="pipe-stage" style="--stage:#676b75"><span class="k">Client</span><h3>Browser</h3><p>16&nbsp;kHz PCM frames over WS; plays streamed audio back.</p></article>
          <article class="pipe-stage" style="--stage:#f2a900"><span class="k">Listen</span><h3>STT · Qwen3-ASR</h3><p>Auto language detection; final utterance, not per-chunk calls.</p></article>
          <article class="pipe-stage" style="--stage:#2272b4"><span class="k">Reason</span><h3>LLM + Genie tools</h3><p>Tool-calls the Genie API; runs off-thread to hide deep-reasoning latency.</p></article>
          <article class="pipe-stage" style="--stage:#ff3621"><span class="k">Speak</span><h3>TTS · VoxCPM2</h3><p>Streaming ~80&nbsp;ms slices over SSE — audio starts mid-sentence.</p></article>
          <article class="pipe-stage" style="--stage:#00a972"><span class="k">Client</span><h3>Audio out</h3><p>Barge-in and cancellation owned server-side by the API.</p></article>
        </div>
        <div class="rail">
          <div class="rail-label">Governed serving &amp; control plane (off the live thread)</div>
          <div class="tag-row">
            <span class="tag" style="--tag:#7d59a5">Unity AI Gateway · guarded FM services</span>
            <span class="tag" style="--tag:#f2a900">Databricks Model Serving · STT/TTS agent endpoints</span>
            <span class="tag" style="--tag:#2272b4">Genie API · One / Space / Agent Mode</span>
            <span class="tag" style="--tag:#00a972">Lakebase · live call state</span>
          </div>
        </div>
        <div class="callout">Constraint (stated in the deck): Model Serving does not yet support true real-time inference — the API layer compensates with streaming TTS, semantic endpointing and background LLM work.</div>
        ${footer(7, "Deck: Realtime API Architecture")}
      </div>
    `,
  },
  {
    title: "Solution architecture",
    className: "",
    accent: "#2272b4",
    notes:
      "The deck's Solution Architecture slide, drawn as layers an architect would defend. Unity Catalog Delta is the governed source of truth (customers, billing, invoices, payments). Reverse-ETL moves it one direction into Lakebase, the sub-second serving layer the app reads on the hot path (live transcripts, call facts, history, resolution events, billing adjustments). Lakebase CDF flows back to UC gold for Genie analytics — the cold path. Point out the app never queries Delta live.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Solution architecture</p>
        <h2>Delta is truth. Lakebase serves the call.</h2>
        <div class="bands">
          <div class="band" style="--band:#2272b4">
            <div class="band-title"><h3>Unity Catalog · Delta</h3><p>Governed source of truth (OLAP)</p></div>
            <div class="tag-row">
              <span class="tag" style="--tag:#2272b4">Customers</span>
              <span class="tag" style="--tag:#2272b4">Billing</span>
              <span class="tag" style="--tag:#2272b4">Invoices</span>
              <span class="tag" style="--tag:#2272b4">Payments</span>
            </div>
          </div>
          <div class="band-sep">reverse-ETL · one direction (sync / CDF)</div>
          <div class="band" style="--band:#00a972">
            <div class="band-title"><h3>Lakebase · Postgres</h3><p>Sub-second serving on the hot path (OLTP)</p></div>
            <div class="tag-row">
              <span class="tag" style="--tag:#00a972">Live voice transcripts</span>
              <span class="tag" style="--tag:#00a972">Call facts</span>
              <span class="tag" style="--tag:#00a972">History</span>
              <span class="tag" style="--tag:#00a972">Resolution events</span>
              <span class="tag" style="--tag:#00a972">Billing adjustments</span>
            </div>
          </div>
          <div class="band-sep">Lakebase CDF → UC gold → Genie (cold analytics path)</div>
          <div class="band" style="--band:#ff3621">
            <div class="band-title"><h3>Genie for Voice app</h3><p>Reads Lakebase live; Genie validates facts</p></div>
            <div class="tag-row">
              <span class="tag" style="--tag:#ff3621">Agent assist UI</span>
              <span class="tag" style="--tag:#ff3621">Realtime voice loop</span>
              <span class="tag" style="--tag:#7d59a5">Genie fact validation</span>
            </div>
          </div>
        </div>
        ${footer(8, "Deck: Solution Architecture (Delta → Lakebase)")}
      </div>
    `,
  },
  {
    title: "How Genie is used",
    className: "",
    accent: "#7d59a5",
    notes:
      "The deck's 'How is Genie used?' detail, sequenced across one call. Pre-call: prefetch a Genie-backed account snapshot off the live path. In-turn: the LLM writes the reply, but Genie/UC facts make it trustworthy — cite the concrete example. Post-call: analytics over structured call data. The architect point is that Genie is invoked where governed facts matter, not on every syllable — that is how token cost stays sane.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">How is Genie used?</p>
        <h2>Genie where facts matter — not every syllable.</h2>
        <p class="section-note">Across a single call, Genie is called at three points. Each one grounds the conversation in governed Unity Catalog data instead of the model's imagination.</p>
        <div class="phase-grid">
          <article class="phase" style="--ph:#f2a900">
            <span class="when">Before / during — off the live path</span>
            <h3>Pre-call context</h3>
            <p>When the agent selects a customer, prefetch a Genie-backed account snapshot: overdue invoices, prior payments, disputes, likely issue areas — before the customer repeats anything.</p>
            <div class="genie-quote">Prefetched off the response path, so it never adds turn latency.</div>
          </article>
          <article class="phase" style="--ph:#00a972">
            <span class="when">In the turn — grounding</span>
            <h3>Grounded reply</h3>
            <p>The LLM writes the actual reply; Genie/UC-grounded facts make it trustworthy. Fail-closed: no validated facts, no spoken claim.</p>
            <div class="genie-quote">“I see invoice INV-2045 has a late fee, and your last payment posted on June 12.”</div>
          </article>
          <article class="phase" style="--ph:#2272b4">
            <span class="when">After — analytics</span>
            <h3>Post-call review</h3>
            <p>Over structured call data: which issues recur, which invoices drive disputes, which agents resolve billing fastest, where customers get stuck.</p>
            <div class="genie-quote">Genie Space / Agent Mode over the governed gold tables.</div>
          </article>
        </div>
        ${footer(9, "Deck: How is Genie used?")}
      </div>
    `,
  },
  {
    title: "Genie-powered voice interface — assets built",
    className: "",
    accent: "#2272b4",
    notes:
      "The deck's 'Genie Powered Voice Enabled Interface / Assets Built' slide, presented as a component inventory an architect can hand off. The app is go/genie-for-voice; the Realtime API, MCP server, multilingual support, latency-testing page and demo recording are all real, addressable surfaces. Give the endpoint paths so it reads as a system, not a slide.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Assets built · go/genie-for-voice</p>
        <h2>A voice-enabled interface, shipped as components.</h2>
        <table class="inventory">
          <thead><tr><th>Component</th><th>What it is</th><th>Surface</th></tr></thead>
          <tbody>
            <tr><td>Voice-enabled app</td><td><p>The Genie-powered cockpit — agent assist + voice.</p></td><td class="endpoint">go/genie-for-voice</td></tr>
            <tr><td>Realtime Voice API</td><td><p>WebSocket STT → LLM + Genie tools → TTS.</p></td><td class="endpoint">/realtime · WS /v1/*</td></tr>
            <tr><td>MCP server</td><td><p>The voice API exposed as MCP tools for any client.</p></td><td class="endpoint">/realtime/mcp</td></tr>
            <tr><td>Multilingual support</td><td><p>24 languages, auto-detected per turn.</p></td><td class="endpoint">/realtime/v1/languages</td></tr>
            <tr><td>Latency testing page</td><td><p>Per-stage timings (stt / llm / tts) in the browser.</p></td><td class="endpoint">/realtime-test</td></tr>
            <tr><td>Demo recording</td><td><p>End-to-end walkthrough of a grounded voice call.</p></td><td class="mono">linked from deck</td></tr>
          </tbody>
        </table>
        ${footer(10, "Deck: Genie Powered Voice Enabled Interface")}
      </div>
    `,
  },
  {
    title: "One runtime, three Genie experiences",
    className: "",
    accent: "#7d59a5",
    notes:
      "The differentiator made concrete, and the thing closed vendors cannot do. A session profile selects Genie One for governed workspace-wide knowledge, a Genie Space for scoped domain analytics, or Agent Mode for deep causal investigation with progress spoken during the wait. Same session protocol, three depths of reasoning.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Deeper reasoning · the vendor gap closed</p>
        <h2>One session protocol, three depths of Genie.</h2>
        <div class="grid-3">
          <article class="card profile-card" style="--card-accent:#7d59a5">
            <span class="badge">profile: knowledge</span><h3>Genie One</h3>
            <p>Your governed workspace — no space to choose first.</p>
            <p class="prompt">“What changed in customer risk this week?”</p>
          </article>
          <article class="card profile-card" style="--card-accent:#2272b4">
            <span class="badge">profile: billing</span><h3>Genie Space</h3>
            <p>A curated domain with its own business semantics.</p>
            <p class="prompt">“Which disputed invoices remain overdue?”</p>
          </article>
          <article class="card profile-card" style="--card-accent:#ff3621">
            <span class="badge">profile: card</span><h3>Agent Mode</h3>
            <p>Causal investigation; progress spoken during the wait.</p>
            <p class="prompt">“Why did approvals fall in this segment?”</p>
          </article>
        </div>
        <div class="callout">Genie tools run under the signed-in user's OBO identity — answers respect the permissions they already have. That is what a voice vendor cannot reach.</div>
        ${footer(11, `Deck: Genie Powered by Voice Models · <a href="${REPO}#share-and-consume-the-realtime-api" target="_blank" rel="noreferrer">README</a>`)}
      </div>
    `,
  },
  {
    title: "Languages & benchmarks",
    className: "",
    accent: "#28a9c7",
    notes:
      "The deck's Languages and Benchmarks slides, together. Coverage is the intersection of Qwen3-ASR and VoxCPM2, with the East / South / West Asia focus the deck highlights. The benchmark method matters more than a single number: compared against existing models (Deepgram) on a locked, human-verified set, judged on business-critical detail capture, not raw WER. Do not invent numbers — present the framework and the honesty boundary.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Languages supported &amp; benchmarks</p>
        <h2>24 languages, judged on business detail.</h2>
        <div class="split">
          <div class="regions">
            <div class="region" style="--reg:#ff3621"><h3>East Asia</h3><div class="langs"><span class="lang-pill">Chinese zh-CN</span><span class="lang-pill">Japanese ja-JP</span><span class="lang-pill">Korean ko-KR</span></div></div>
            <div class="region" style="--reg:#f2a900"><h3>South &amp; West Asia / SEA</h3><div class="langs"><span class="lang-pill">Hindi hi-IN</span><span class="lang-pill">Indonesian id-ID</span><span class="lang-pill">Malay ms-MY</span><span class="lang-pill">Filipino fil-PH</span><span class="lang-pill">Thai th-TH</span><span class="lang-pill">Vietnamese vi-VN</span></div></div>
            <div class="region" style="--reg:#2272b4"><h3>+ European &amp; more</h3><div class="langs"><span class="lang-pill">English</span><span class="lang-pill">French</span><span class="lang-pill">German</span><span class="lang-pill">Spanish</span><span class="lang-pill">… 24 total</span></div></div>
          </div>
          <div class="method">
            <div class="m-row"><span class="m-k">Coverage</span><p class="m-v"><strong>STT ∩ TTS</strong> — a language ships only when Qwen3-ASR and VoxCPM2 both support it.</p></div>
            <div class="m-row"><span class="m-k">Comparison</span><p class="m-v">Against existing models (<strong>Deepgram</strong>) on a locked, human-verified evaluation set.</p></div>
            <div class="m-row"><span class="m-k">Deciding metric</span><p class="m-v"><strong>Business-critical detail capture</strong> — invoice numbers, amounts, dates — over raw WER.</p></div>
          </div>
        </div>
        <div class="callout">Honesty boundary: 24 is the model-supported intersection; a subset (en, th, id, zh) was round-trip validated when documented — the rest still need a full human sweep.</div>
        ${footer(12, "Deck: Languages supported · Benchmarks")}
      </div>
    `,
  },
  {
    title: "Guardrails — AI Gateway enabled",
    className: "dark",
    accent: "#ff6b55",
    notes:
      "The deck's Guardrails slide, drawn as the enforceable Gateway contract. Foundation-model traffic runs through Unity AI Gateway with purpose-specific policy bundles ranked in order: unsafe content and jailbreak enforce, contact data redacts, credentials and government IDs block, hallucination is logged as an observer — it is not the factuality gate. Deploys fail closed if attached policies drift from the manifest.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Guardrails · AI Gateway enabled</p>
        <h2>Policy as an enforceable contract.</h2>
        <p class="section-note">Foundation-model traffic routes through Unity AI Gateway with purpose-specific bundles. Ranks are ordered; a deploy fails closed if the attached policies drift from the manifest.</p>
        <div class="ranks">
          <div class="rank"><span class="n">1</span><div><h3>Unsafe content</h3><p>Input + output, both guarded FM services.</p></div><span class="mode enforce">Enforce</span></div>
          <div class="rank"><span class="n">2</span><div><h3>Jailbreak</h3><p>Prompt-injection defense on input.</p></div><span class="mode enforce">Enforce</span></div>
          <div class="rank"><span class="n">3</span><div><h3>Contact data</h3><p>Email + phone masked before model, tools or persistence.</p></div><span class="mode redact">Redact</span></div>
          <div class="rank"><span class="n">4</span><div><h3>Credentials / government IDs</h3><p>Cards (Luhn), SSN, IBAN, passports denied.</p></div><span class="mode block">Block</span></div>
          <div class="rank"><span class="n">5</span><div><h3>Hallucination</h3><p>Observer only — not the factuality gate; Genie grounding is.</p></div><span class="mode log">Log</span></div>
        </div>
        <div class="callout">Enforcement spans the Gateway <em>and</em> the post-STT application boundary — the same PII split before browser, history, model, tool or persistence admission.</div>
        ${footer(13, "Deck: Guardrails — AI Gateway Enabled")}
      </div>
    `,
  },
  {
    title: "End-to-end tracing",
    className: "",
    accent: "#00a972",
    notes:
      "The deck's End-End Tracing slide. Every voice turn emits per-stage timings — stt_ms, llm_ms, tts_first_ms — plus tool calls, guardrail outcomes and provenance, persisted to Lakebase voice_traces and surfaced in the UI. Streaming means TTS first-audio overlaps LLM completion, which is why time-to-answer beats naive turn duration. Do not quote fabricated millisecond values; these are the measured fields, values vary per turn and language.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">End-to-end tracing</p>
        <h2>Every turn is measured and attributable.</h2>
        <p class="section-note">Each turn records the fields below to Lakebase <span class="mono" style="font-family:ui-monospace,Menlo,monospace">voice_traces</span> and the UI. Streaming overlaps TTS with the tail of the LLM, so time-to-answer beats raw turn duration.</p>
        <div class="trace-bar">
          <div class="trace-seg" style="flex:2.4;background:#f2a900">STT<span>stt_ms · Qwen3-ASR</span></div>
          <div class="trace-seg" style="flex:3.4;background:#2272b4">LLM + Genie tools<span>llm_ms · off-thread for Agent Mode</span></div>
          <div class="trace-seg" style="flex:2.2;background:#ff3621">TTS first audio<span>tts_first_ms · streamed</span></div>
        </div>
        <div class="trace-legend">
          <div><span class="mono">stt_ms</span> — final transcript latency</div>
          <div><span class="mono">llm_ms</span> — reasoning + Genie tool calls</div>
          <div><span class="mono">tts_first_ms</span> — time to first streamed audio</div>
          <div>+ tool calls · guardrail outcomes · provenance</div>
        </div>
        <div class="callout">Proportions are illustrative — values vary by turn and language. The point is that each stage is captured, so latency and policy outcomes are inspectable, not asserted.</div>
        ${footer(14, "Deck: End-End Tracing")}
      </div>
    `,
  },
  {
    title: "Prerequisites & gotchas",
    className: "",
    accent: "#ff6b55",
    notes:
      "The deck's Prerequisites & Gotchas slide — the credibility slide. Prerequisites: GPU availability in your region for the serving endpoints, and the workspace/OBO access for Genie. Gotchas: Model Serving doesn't yet support true real-time inference; STT voice guardrails aren't in AI Gateway yet, so Qwen3-ASR's own guardrails are used; and language identification emits a language tag then transcript across 30 languages plus 22 Chinese dialects. Presenting the constraints plainly is what an architect audience trusts.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Prerequisites &amp; gotchas</p>
        <h2>What it needs — and what to watch.</h2>
        <div class="duo">
          <div class="duo-col" style="--duo:#00a972">
            <h3>Prerequisites</h3>
            <div class="duo-item"><strong>GPU availability in your region</strong><span>Qwen3-ASR and VoxCPM2 run on GPU Model Serving — capacity must exist where you deploy.</span></div>
            <div class="duo-item"><strong>Workspace + OBO access to Genie</strong><span>Genie tools run as the signed-in user; they reach only spaces the user already has CAN_RUN on.</span></div>
            <div class="duo-item"><strong>Unity Catalog + Lakebase</strong><span>Governed Delta source of truth and a Lakebase project for sub-second serving.</span></div>
          </div>
          <div class="duo-col" style="--duo:#ff6b55">
            <h3>Gotchas</h3>
            <div class="duo-item"><strong>No native real-time inference yet</strong><span>Model Serving doesn't support true realtime — the API compensates with streaming + endpointing.</span></div>
            <div class="duo-item"><strong>STT guardrails not in AI Gateway</strong><span>Voice guardrails for speech-to-text aren't in Gateway yet; Qwen3-ASR's own guardrails are used.</span></div>
            <div class="duo-item"><strong>Language identification behavior</strong><span>When language isn't forced, the model emits a language tag then the transcript — 30 languages + 22 Chinese dialects.</span></div>
          </div>
        </div>
        ${footer(15, "Deck: Prerequisites &amp; Gotchas")}
      </div>
    `,
  },
  {
    title: "Future improvements",
    className: "",
    accent: "#2272b4",
    notes:
      "The deck's Future Improvements slide, split honestly into current support versus what's planned or in research. Planned: native streaming/real-time API on Databricks, better Genie Agent Mode insights, extended AI Gateway support, Genie One insights. In research: voice guardrails and CPU model hosting. Present these as direction, not shipped claims.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Future improvements</p>
        <h2>Current support → what's next.</h2>
        <div class="duo">
          <div class="duo-col" style="--duo:#00a972">
            <h3>Planned on Databricks</h3>
            <div class="duo-item"><strong>Native real-time API</strong><span>Streaming and real-time inference support for multilingual voice on Model Serving.</span></div>
            <div class="duo-item"><strong>Better Agent Mode insights</strong><span>Deeper Genie Agent Mode reasoning and Genie One workspace-level insights.</span></div>
            <div class="duo-item"><strong>Extended AI Gateway support</strong><span>Broader guardrail coverage across the voice path.</span></div>
          </div>
          <div class="duo-col" style="--duo:#f2a900">
            <h3>In research</h3>
            <div class="duo-item"><strong>Voice guardrails</strong><span>STT/TTS guardrails inside AI Gateway rather than model-native only.</span></div>
            <div class="duo-item"><strong>CPU model hosting</strong><span>Lower-cost serving to relax the regional GPU prerequisite.</span></div>
            <div class="duo-item"><strong>Full 24-language validation</strong><span>Human round-trip review across every supported language.</span></div>
          </div>
        </div>
        ${footer(16, "Deck: Future Improvements (planned / in research — not shipped)")}
      </div>
    `,
  },
  {
    title: "Genie with voice is coming",
    className: "red",
    accent: "#ffffff",
    notes:
      "Close on the deck's own promise: Genie with voice (English) support is coming soon, with multilingual right behind it. The achievement is that Genie's governed reasoning — Agent Mode and Genie One — is now reachable by voice, in 24 languages, on open models we host and govern.",
    html: `
      <div class="slide-inner hero">
        <p class="eyebrow">Genie for Voice</p>
        <h1>Genie's reasoning, now reachable by voice.</h1>
        <p class="lede">Open models on Databricks · 24 languages · Genie One, Space and Agent Mode · governed end to end. Genie with voice (English) is coming soon — multilingual right behind it.</p>
        <p class="meta">${commit("9861be7a", "24 Jun · first commit")} &nbsp;→&nbsp; ${commit("ad578361", "21 Sep · governed inference trail")} &nbsp;·&nbsp; go/genie-for-voice</p>
      </div>
    `,
  },
];

const deck = document.querySelector("#deck");
const previous = document.querySelector("#previous");
const next = document.querySelector("#next");
const counter = document.querySelector("#counter");
const picker = document.querySelector("#slide-picker");
const gridToggle = document.querySelector("#grid-toggle");
const gridDialog = document.querySelector("#grid-dialog");
const gridClose = document.querySelector("#grid-close");
const gridList = document.querySelector("#grid-list");
const notesToggle = document.querySelector("#notes-toggle");
const notesPanel = document.querySelector("#notes-panel");
const notesClose = document.querySelector("#notes-close");
const notesContent = document.querySelector("#notes-content");
const fullscreenToggle = document.querySelector("#fullscreen-toggle");

let current = 0;

function hashIndex() {
  const raw = window.location.hash.slice(1);
  const index = Number.parseInt(raw || "0", 10);
  return Number.isFinite(index) ? index : 0;
}

function clamp(index) {
  return Math.max(0, Math.min(slides.length - 1, index));
}

function show(index, updateHash = true) {
  current = clamp(index);
  const slide = slides[current];
  deck.innerHTML = `
    <section class="slide ${slide.className || ""}" style="--accent:${slide.accent}" aria-label="Slide ${current + 1}: ${slide.title}">
      ${slide.html}
    </section>
  `;
  counter.textContent = `${current + 1} / ${slides.length}`;
  picker.value = String(current);
  previous.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  notesContent.innerHTML = `<p>${slide.notes}</p>`;
  document.title = `${current + 1} · ${slide.title} — Genie for Voice`;
  if (updateHash) {
    history.replaceState(null, "", `#${current}`);
  }
}

function populateNavigation() {
  picker.innerHTML = slides
    .map((slide, index) => `<option value="${index}">${index + 1} / ${slide.title}</option>`)
    .join("");
  gridList.innerHTML = slides
    .map(
      (slide, index) => `
        <button class="grid-item" type="button" data-slide="${index}">
          <span class="grid-num">${String(index + 1).padStart(2, "0")}</span>
          <strong>${slide.title}</strong>
        </button>
      `,
    )
    .join("");
}

function move(delta) {
  show(current + delta);
}

function openGrid() {
  if (!gridDialog.open) {
    gridDialog.showModal();
  }
}

function closeGrid() {
  if (gridDialog.open) {
    gridDialog.close();
  }
}

function toggleNotes(force) {
  const shouldShow = typeof force === "boolean" ? force : notesPanel.hidden;
  notesPanel.hidden = !shouldShow;
  notesToggle.setAttribute("aria-pressed", String(shouldShow));
}

async function toggleFullscreen() {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await document.documentElement.requestFullscreen();
  }
}

previous.addEventListener("click", () => move(-1));
next.addEventListener("click", () => move(1));
counter.addEventListener("click", openGrid);
picker.addEventListener("change", (event) => show(Number(event.target.value)));
gridToggle.addEventListener("click", openGrid);
gridClose.addEventListener("click", closeGrid);
gridList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-slide]");
  if (!item) return;
  show(Number(item.dataset.slide));
  closeGrid();
});
notesToggle.addEventListener("click", () => toggleNotes());
notesClose.addEventListener("click", () => toggleNotes(false));
fullscreenToggle.addEventListener("click", toggleFullscreen);

gridDialog.addEventListener("click", (event) => {
  if (event.target === gridDialog) closeGrid();
});

window.addEventListener("hashchange", () => show(hashIndex(), false));
document.addEventListener("fullscreenchange", () => {
  fullscreenToggle.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("select, button, input, textarea")) return;
  if (event.key === "ArrowRight" || event.key === " " || event.key === "PageDown") {
    event.preventDefault();
    move(1);
  }
  if (event.key === "ArrowLeft" || event.key === "PageUp") {
    event.preventDefault();
    move(-1);
  }
  if (event.key.toLowerCase() === "g") openGrid();
  if (event.key.toLowerCase() === "n") toggleNotes();
  if (event.key.toLowerCase() === "f") toggleFullscreen();
  if (event.key === "Home") show(0);
  if (event.key === "End") show(slides.length - 1);
});

populateNavigation();
show(hashIndex());
