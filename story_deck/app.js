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
      "The promise. Genie already reasons over governed business data. This gives that intelligence a real-time, multilingual voice — using open models hosted and governed on Databricks. One thesis for the whole deck: Genie's governed intelligence, spoken, in the language people actually use.",
    html: `
      <div class="slide-inner hero">
        <p class="eyebrow">Genie · Voice · Multilingual · Databricks</p>
        <h1>Genie's intelligence, now spoken in any language.</h1>
        <p class="lede">Give Genie's governed business reasoning a real-time, multilingual voice — on open models, hosted and governed on Databricks.</p>
        <p class="meta">Suneel Sunkara · Senior Partner Solutions Architect · 2026</p>
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
      "Reframed as enterprise value. Genie already reasons over a governed ontology — permission-aware business logic over Unity Catalog data — but only as text, in English-first tools. Enterprise front lines (contact centers, branches, field) run on voice, across many languages, so those governed answers never reach the live moment. The opportunity is to give Genie's governed ontology a real-time, multilingual voice, built on OSS models on Databricks.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Problem statement</p>
        <h2>Enterprises run on a governed ontology — but it only answered in text, in English.</h2>
        <div class="grid-3 problem-grid">
          <article class="card" style="--card-accent:#2272b4">
            <h3>Governed reasoning, stuck in text</h3>
            <p>Genie reasons over Unity Catalog–governed data with permission-aware business logic — yet its answers live in dashboards and chat, not in the live conversation where decisions get made.</p>
          </article>
          <article class="card" style="--card-accent:#f2a900">
            <h3>The enterprise front line is voice</h3>
            <p>Contact centers, branches, and field teams resolve issues by speaking. Typing a query mid-call isn't an option, so the governed answer arrives too late — or never.</p>
          </article>
          <article class="card" style="--card-accent:#00a972">
            <h3>Most of the world isn't English</h3>
            <p>Customers and agents speak dozens of languages. English-first assistants leave the majority of real enterprise interactions unserved.</p>
          </article>
        </div>
        <div class="access-progression">
          <div><span>Today</span><strong>Type a question</strong><small>Dashboard or chat</small></div>
          <i>→</i>
          <div><span>Where decisions happen</span><strong>Speak during the live moment</strong><small>Call · branch · field</small></div>
          <i>→</i>
          <div><span>What changes</span><strong>Hear Genie's answer</strong><small>In the caller's language</small></div>
        </div>
        <div class="callout">The opportunity: give Genie's governed ontology a real-time, multilingual voice — so trustworthy, permission-aware answers are spoken in the caller's own language, at enterprise scale.</div>
      </div>
    `,
  },
  {
    title: "Why the market is building voice agents",
    className: "",
    accent: "#f2a900",
    notes:
      "The reason major AI companies are investing in voice, using their own stated use cases. OpenAI targets customer support, personal assistance and education. Anthropic targets hands-free planning, learning, creative thinking, preparation and work through connected tools. Google targets retail support, healthcare companions, financial advisors, education and real-time translation. Deepgram and ElevenLabs are a second category: third-party voice-agent platforms that provide production speech pipelines, turn-taking, telephony, voices, tools and knowledge retrieval. The conclusion is concrete: voice puts intelligence into moments where typing is slow, unnatural or impossible.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Why voice agents · official use cases</p>
        <h2>Why the market is building voice agents.</h2>
        <div class="voice-market">
          <section class="voice-market-group">
            <div class="voice-market-label"><strong>AI model platforms</strong><span>Put their intelligence into natural, hands-free conversations</span></div>
            <div class="voice-market-cards three">
              <article class="voice-use-card" style="--voice:#00a972">
                <h3>OpenAI</h3>
                <strong>Support · assistance · education</strong>
                <p>Built gpt-realtime for customer support, personal assistance and education, with precise tool calling and multilingual speech.</p>
                <a href="https://openai.com/index/introducing-gpt-realtime/" target="_blank" rel="noreferrer">Official source ↗</a>
              </article>
              <article class="voice-use-card" style="--voice:#d97757">
                <h3>Anthropic</h3>
                <strong>Plan · learn · think · work hands-free</strong>
                <p>Claude Voice is positioned for daily planning, learning, creative thinking and connected tools such as Gmail, Calendar and Slack.</p>
                <a href="https://support.anthropic.com/en/articles/11101966-using-voice-mode-on-claude-mobile-apps" target="_blank" rel="noreferrer">Official source ↗</a>
              </article>
              <article class="voice-use-card" style="--voice:#2272b4">
                <h3>Google</h3>
                <strong>Serve · advise · teach · translate</strong>
                <p>Gemini Live targets retail support, healthcare companions, financial advisors, education and real-time voice translation.</p>
                <a href="https://ai.google.dev/gemini-api/docs/live-api" target="_blank" rel="noreferrer">Official source ↗</a>
              </article>
            </div>
          </section>
          <section class="voice-market-group compact">
            <div class="voice-market-label"><strong>Third-party voice-agent platforms</strong><span>Provide the production speech layer around an agent</span></div>
            <div class="voice-market-cards two">
              <article class="voice-use-card platform-card" style="--voice:#28a9c7">
                <h3>Deepgram</h3>
                <strong>Listen · think · speak in one WebSocket</strong>
                <p>Managed STT, LLM, TTS, turn-taking, barge-in, function calling and telephony.</p>
              </article>
              <article class="voice-use-card platform-card" style="--voice:#7d59a5">
                <h3>ElevenLabs</h3>
                <strong>Expressive agents grounded in knowledge</strong>
                <p>Conversational voices, agent tools, telephony, knowledge bases and RAG.</p>
              </article>
            </div>
          </section>
        </div>
        <div class="market-conclusion"><strong>Why voice?</strong><span>It brings intelligence into live support, work, learning and multilingual conversations — the moments where typing is slow, unnatural or impossible.</span></div>
        ${footer(2, "Official sources: OpenAI Realtime · Claude Voice · Gemini Live · Deepgram Voice Agent · ElevenLabs Agents")}
      </div>
    `,
  },
  {
    title: "The missing layer",
    className: "",
    accent: "#2272b4",
    notes:
      "The missing layer, made visual. The market already provides complete voice pipelines: Deepgram documents listen-think-speak in one WebSocket with function calling; ElevenLabs provides voice agents with knowledge bases and RAG; OpenAI and Gemini provide realtime speech-to-speech. Those are real strengths. What they do not provide is Genie: governed business semantics, Genie Spaces and Agent Mode, Databricks-native model control, and one evidence trail across the whole answer.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">The missing layer</p>
        <h2>The market solved voice. We connect it to Genie.</h2>
        <div class="capability-bridge">
          <section class="bridge-plane voice-plane">
            <span class="bridge-kicker">Voice platforms</span>
            <h3>Listen · think · speak</h3>
            <div class="capability-pills">
              <span>Realtime audio</span><span>Turn-taking</span><span>Tool calling</span>
              <span>HD voices</span><span>Knowledge bases</span><span>Telephony</span>
            </div>
            <p>Deepgram · ElevenLabs · OpenAI Realtime · Gemini Live</p>
          </section>
          <div class="bridge-gap">
            <span>Missing</span>
            <strong>business<br>intelligence</strong>
            <i>→</i>
          </div>
          <section class="bridge-plane genie-plane">
            <span class="bridge-kicker">Genie on Databricks</span>
            <h3>Reason over the business</h3>
            <div class="capability-pills">
              <span>Business semantics</span><span>Genie Spaces</span><span>Agent Mode</span>
              <span>Governed data</span><span>Model control</span><span>Evidence trail</span>
            </div>
            <p>Enterprise context the speech layer does not own.</p>
          </section>
        </div>
        <div class="callout">This isn't a speech-quality gap. It is an <strong>intelligence-integration gap</strong>: making Genie available inside the live conversation.</div>
        ${footer(3, "Sources: Deepgram Voice Agent API · ElevenLabs Agents · OpenAI Realtime · Gemini Live")}
      </div>
    `,
  },
  {
    title: "Why Qwen3-ASR and VoxCPM2",
    className: "",
    accent: "#00a972",
    notes:
      "Why these two open models, based on their official model cards and the needs of this application. Qwen3-ASR-1.7B is Apache-2.0, supports language identification and ASR across 30 languages plus 22 Chinese dialects, and supports both streaming and offline inference. VoxCPM2 is Apache-2.0, supports TTS across 30 languages plus 9 Chinese dialects, produces 48 kHz audio, and supports voice design and controllable cloning. Together they form a complementary open STT/TTS pair; their shared application allowlist is 24 end-to-end languages. Both can be registered in Unity Catalog and hosted on GPU Model Serving.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Model choice · open multilingual speech</p>
        <h2>Why Qwen3-ASR + VoxCPM2.</h2>
        <div class="model-choice-grid">
          <article class="model-choice-card asr-choice">
            <div class="model-choice-head">
              <span>Listen · STT</span>
              <h3>Qwen3-ASR-1.7B</h3>
              <p>Open speech recognition with language identification.</p>
            </div>
            <div class="choice-facts">
              <div><strong>30</strong><span>languages</span></div>
              <div><strong>22</strong><span>Chinese dialects</span></div>
              <div><strong>1.7B</strong><span>parameters</span></div>
            </div>
            <ul>
              <li>Automatic language identification</li>
              <li>Streaming and offline inference</li>
              <li>Long-audio and timestamp support</li>
              <li>Apache-2.0 open weights</li>
            </ul>
            <a href="https://huggingface.co/Qwen/Qwen3-ASR-1.7B" target="_blank" rel="noreferrer">Official model card ↗</a>
          </article>
          <article class="model-choice-card tts-choice">
            <div class="model-choice-head">
              <span>Speak · TTS</span>
              <h3>VoxCPM2</h3>
              <p>Open multilingual speech generation with controllable voice.</p>
            </div>
            <div class="choice-facts">
              <div><strong>30</strong><span>languages</span></div>
              <div><strong>48 kHz</strong><span>audio output</span></div>
              <div><strong>2B</strong><span>parameters</span></div>
            </div>
            <ul>
              <li>Direct multilingual synthesis</li>
              <li>Voice design and controllable cloning</li>
              <li>9 Chinese dialects</li>
              <li>Apache-2.0 open weights</li>
            </ul>
            <a href="https://huggingface.co/openbmb/VoxCPM2" target="_blank" rel="noreferrer">Official model card ↗</a>
          </article>
        </div>
        <div class="selection-fit">
          <div><strong>24-language intersection</strong><span>One end-to-end allowlist supported by both STT and TTS.</span></div>
          <div><strong>Open &amp; deployable</strong><span>Register in Unity Catalog and host on Databricks GPU Model Serving.</span></div>
          <div><strong>Complementary pair</strong><span>Purpose-built ASR + TTS instead of locking the voice layer to one vendor.</span></div>
        </div>
        ${footer(4, "Sources: official Qwen3-ASR-1.7B and VoxCPM2 model cards")}
      </div>
    `,
  },
  {
    title: "Why open models on Databricks",
    className: "",
    accent: "#f2a900",
    notes:
      "Correct taxonomy: two customer-deployed open voice models and two Databricks Foundation Model APIs. Qwen3-ASR-1.7B and VoxCPM2 are downloaded, registered in Unity Catalog, packaged as MLflow ResponsesAgent models and hosted on GPU Model Serving. Qwen3-Next and GPT-5.5 are not presented as open models; they are Databricks-hosted foundation models reached through governed model services. The formatting keeps each model and role visually separate.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Two model families · one governed plane</p>
        <h2>Open voice models + Databricks foundation models.</h2>
        <div class="model-family-grid">
          <section class="model-family open-family">
            <div class="model-family-head">
              <span>Customer-deployed</span>
              <h3>2 open voice models</h3>
              <p>Registered in Unity Catalog and hosted on GPU Model Serving.</p>
            </div>
            <div class="model-row"><b>Qwen3-ASR-1.7B</b><span>Speech-to-text · language identification</span></div>
            <div class="model-row"><b>VoxCPM2</b><span>Text-to-speech · streamed audio</span></div>
            <div class="model-path">Hugging Face → Unity Catalog → MLflow ResponsesAgent → GPU endpoint</div>
          </section>
          <section class="model-family foundation-family">
            <div class="model-family-head">
              <span>Databricks-hosted</span>
              <h3>2 Foundation Model APIs</h3>
              <p>Consumed through governed Unity Catalog model services.</p>
            </div>
            <div class="model-row"><b>Qwen3-Next 80B</b><span>Intent understanding · summarization</span></div>
            <div class="model-row"><b>GPT-5.5</b><span>Translation · localized response</span></div>
            <div class="model-path">Foundation Model API → UC model service → AI Gateway controls</div>
          </section>
        </div>
        <div class="callout">The distinction matters: <strong>the speech models are open and GPU-hosted by you</strong>; the reasoning models are <strong>Databricks Foundation Model APIs</strong>.</div>
        ${footer(5, "2 open voice models · 2 Databricks Foundation Model APIs")}
      </div>
    `,
  },
  {
    title: "From models to a natural conversation",
    className: "",
    accent: "#ff3621",
    notes:
      "A positive engineering frame, not a list of problems. Capable models alone do not create a natural conversation; the runtime adds turn-taking, latency management, multilingual continuity and enterprise operation. Each capability has a concrete technique behind it.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">The engineering that makes it real</p>
        <h2>Four capabilities turn models into a live conversation.</h2>
        <div class="engineering-grid">
          <article class="engineering-card" style="--card-accent:#f2a900">
            <span class="engineering-num">01 · Conversation</span>
            <h3>Natural turn-taking</h3>
            <p>Silero VAD for semantic end-of-turn, server-side endpointing, and barge-in so callers can interrupt.</p>
            <strong>Feels like a dialogue, not push-to-talk.</strong>
          </article>
          <article class="engineering-card" style="--card-accent:#2272b4">
            <span class="engineering-num">02 · Responsiveness</span>
            <h3>Response latency</h3>
            <p>Reasoning runs off-thread; VoxCPM2 streams ~80&nbsp;ms PCM slices over SSE so audio starts mid-sentence.</p>
            <strong>The caller hears progress before the full turn finishes.</strong>
          </article>
          <article class="engineering-card" style="--card-accent:#00a972">
            <span class="engineering-num">03 · Language</span>
            <h3>Multilingual continuity</h3>
            <p>Per-turn language detection, translation when needed, and a matched voice on the way back.</p>
            <strong>One conversation can stay in the caller's language.</strong>
          </article>
          <article class="engineering-card" style="--card-accent:#7d59a5">
            <span class="engineering-num">04 · Operation</span>
            <h3>Enterprise operation</h3>
            <p>One-command deploy, readiness checks, end-to-end tracing, and model governance.</p>
            <strong>Deployable and inspectable—not just a model demo.</strong>
          </article>
        </div>
        ${footer(6, `Built &amp; running — <a href="${REPO}/commits/main" target="_blank" rel="noreferrer">see the commit history</a>`)}
      </div>
    `,
  },
  {
    title: "How it works — the voice loop",
    className: "",
    accent: "#f2a900",
    notes:
      "The only 'how it works' that matters for this deck: the voice ↔ intelligence loop. Speak → understand → reason with Genie → answer in language → speak back. Qwen3-ASR transcribes and detects language, Qwen3-Next reads intent, Genie answers from governed data, GPT-5.5 translates when needed, VoxCPM2 streams the reply. The caller can interrupt at any point. No data-plumbing here — this is the runtime experience.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">How it works</p>
        <h2>Speak → reason with Genie → speak back.</h2>
        <p class="section-note">One WebSocket carries a single turn end to end. Four models and Genie collaborate; the caller can barge in at any time.</p>
        <div class="flow">
          <article class="flow-step"><span class="num">01 · Listen</span><h3>Speak &amp; transcribe</h3><p>Browser streams audio; <strong>Qwen3-ASR</strong> returns the final utterance and detects the language.</p></article>
          <article class="flow-step"><span class="num">02 · Understand</span><h3>Read intent</h3><p><strong>Qwen3-Next</strong> interprets what the caller is actually asking.</p></article>
          <article class="flow-step"><span class="num">03 · Reason</span><h3>Ask Genie</h3><p>The request calls <strong>Genie</strong> — One, a Space, or Agent Mode — over your governed data.</p></article>
          <article class="flow-step"><span class="num">04 · Compose</span><h3>Answer in language</h3><p>The answer is composed and, when needed, translated by <strong>GPT-5.5</strong>.</p></article>
          <article class="flow-step"><span class="num">05 · Speak</span><h3>Stream back</h3><p><strong>VoxCPM2</strong> streams audio mid-sentence; barge-in is owned server-side.</p></article>
        </div>
        <div class="callout">Honest constraint: Model Serving doesn't yet offer true real-time inference — the loop compensates with streaming speech, semantic endpointing and off-thread reasoning.</div>
        ${footer(7, "How it works · the voice loop")}
      </div>
    `,
  },
  {
    title: "The Databricks architecture",
    className: "dark architecture-image-slide",
    accent: "#2272b4",
    notes:
      "The complete L200 Databricks architecture, shown directly rather than redrawn in slide components. It covers the simplified deployment lane, governed realtime runtime, AI Gateway capability tiers, Genie intelligence, operational state and the durable evidence plane.",
    html: `
      <div class="slide-inner architecture-image-frame">
        <img src="/story-assets/l200-databricks-architecture.png" alt="L200 Databricks architecture: deployment workflow, governed realtime runtime, AI Gateway, Genie intelligence, operational data and evidence plane">
      </div>
    `,
  },
  {
    title: "How Genie contributes",
    className: "",
    accent: "#7d59a5",
    notes:
      "What Genie adds that the models can't. Across a single call Genie shows up at three points — pre-call context, in-conversation grounding, and post-call analytics. Each grounds the conversation in governed business data rather than the model's imagination, and Genie is invoked where facts matter, not on every syllable, which keeps cost proportional to decisions.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">How Genie contributes</p>
        <h2>What Genie adds that the models can't.</h2>
        <p class="section-note">Across a single call, Genie is invoked at three points — each grounds the conversation in governed business data, and only where facts matter, not on every syllable.</p>
        <div class="phase-grid">
          <article class="phase" style="--ph:#f2a900">
            <span class="when">Before the turn — context</span>
            <h3>Pre-call context</h3>
            <p>Prefetch a Genie-backed account snapshot — recent activity, open items, likely issues — before the customer repeats anything.</p>
            <div class="genie-quote">Prefetched off the response path, so it never adds turn latency.</div>
          </article>
          <article class="phase" style="--ph:#00a972">
            <span class="when">In the turn — grounding</span>
            <h3>Grounded answer</h3>
            <p>The reply is grounded in Genie's reasoning over governed data, so the spoken answer reflects the business, not a guess.</p>
            <div class="genie-quote">“Invoice INV-2045 has a late fee, and your last payment posted on June&nbsp;12.”</div>
          </article>
          <article class="phase" style="--ph:#2272b4">
            <span class="when">After the turn — analytics</span>
            <h3>Post-call review</h3>
            <p>Which issues recur, which cases stall, where customers get stuck — analyzed over governed, structured data.</p>
            <div class="genie-quote">Genie Space / Agent Mode over the governed tables.</div>
          </article>
        </div>
        ${footer(9, "How Genie contributes across a call")}
      </div>
    `,
  },
  {
    title: "Experiences it enables",
    className: "",
    accent: "#2272b4",
    notes:
      "The outcomes, not the endpoint list. The same runtime enables live agent assistance, multilingual customer interaction, voice-enabled Genie exploration, contact-center resolution, voice access from external clients via the Realtime API, and MCP access for other intelligent apps. Endpoint paths stay as a small technical footer so the slide reads as value, not plumbing.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">What it enables</p>
        <h2>One runtime, many enterprise experiences.</h2>
        <div class="experience-grid">
          <article class="experience-card" style="--card-accent:#00a972"><span>For agents</span><h3>Live agent assist</h3><p>Surface Genie-grounded context while the call is happening.</p><strong>Surface · voice cockpit</strong></article>
          <article class="experience-card" style="--card-accent:#f2a900"><span>For customers</span><h3>Multilingual interaction</h3><p>Speak naturally and hear the governed answer in the same language.</p><strong>Coverage · 24-language STT/TTS intersection</strong></article>
          <article class="experience-card" style="--card-accent:#7d59a5"><span>For analysts</span><h3>Voice-enabled Genie</h3><p>Explore governed data by speaking through Genie One, Space or Agent Mode.</p><strong>Depth · discovery to multi-step reasoning</strong></article>
          <article class="experience-card" style="--card-accent:#2272b4"><span>For operations</span><h3>Resolution support</h3><p>Bring live account facts and follow-up analysis into the resolution journey.</p><strong>Value · less context switching</strong></article>
          <article class="experience-card" style="--card-accent:#ff3621"><span>For application teams</span><h3>Realtime Voice API</h3><p>Embed the complete speech → Genie → speech loop over one WebSocket.</p><strong>Surface · /realtime</strong></article>
          <article class="experience-card" style="--card-accent:#28a9c7"><span>For agent builders</span><h3>MCP access</h3><p>Expose the voice runtime as tools to other intelligent applications.</p><strong>Surface · /realtime/mcp</strong></article>
        </div>
        <div class="callout"><span class="mono" style="font-family:ui-monospace,Menlo,monospace">go/genie-for-voice</span> &nbsp;·&nbsp; <span class="mono" style="font-family:ui-monospace,Menlo,monospace">/realtime</span> &nbsp;·&nbsp; <span class="mono" style="font-family:ui-monospace,Menlo,monospace">/realtime/mcp</span> &nbsp;·&nbsp; <span class="mono" style="font-family:ui-monospace,Menlo,monospace">/realtime/v1/languages</span></div>
        ${footer(10, "Experiences enabled by the voice runtime")}
      </div>
    `,
  },
  {
    title: "One runtime, three Genie modes",
    className: "",
    accent: "#7d59a5",
    notes:
      "The same voice runtime reaches Genie at three depths, chosen by a session profile: Genie One for broad discovery across governed knowledge, a Genie Space for domain-specific analytics and semantics, and Agent Mode for deeper multi-step investigation with progress spoken during the wait. The point is business depth, not access-control mechanics.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Genie interaction modes</p>
        <h2>One voice runtime, three depths of Genie.</h2>
        <div class="grid-3">
          <article class="card profile-card" style="--card-accent:#7d59a5">
            <span class="badge">profile: knowledge</span><h3>Genie One</h3>
            <p>Broad discovery across your governed knowledge — no space to choose first.</p>
            <p class="prompt">“What changed in customer risk this week?”</p>
          </article>
          <article class="card profile-card" style="--card-accent:#2272b4">
            <span class="badge">profile: billing</span><h3>Genie Space</h3>
            <p>Domain-specific analytics with its own business semantics.</p>
            <p class="prompt">“Which disputed invoices remain overdue?”</p>
          </article>
          <article class="card profile-card" style="--card-accent:#ff3621">
            <span class="badge">profile: financial-services</span><h3>Agent Mode</h3>
            <p>Deeper multi-step investigation; progress spoken during the wait.</p>
            <p class="prompt">“Why did approvals fall in this segment?”</p>
          </article>
        </div>
        <div class="callout">Same voice runtime; the profile selects <strong>how deep Genie reasons</strong> — broad discovery, domain analytics, or multi-step investigation.</div>
        ${footer(11, `Genie interaction modes · <a href="${REPO}#share-and-consume-the-realtime-api" target="_blank" rel="noreferrer">README</a>`)}
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
        <p class="eyebrow">Languages &amp; validation</p>
        <h2>24 languages — reaching real enterprise customers.</h2>
        <div class="split">
          <div class="regions language-regions">
            <div class="region" style="--reg:#ff3621"><h3>East Asia</h3><div class="langs"><span class="lang-pill">Chinese zh-CN</span><span class="lang-pill">Japanese ja-JP</span><span class="lang-pill">Korean ko-KR</span></div></div>
            <div class="region" style="--reg:#f2a900"><h3>South &amp; Southeast Asia</h3><div class="langs"><span class="lang-pill">Hindi</span><span class="lang-pill">Indonesian</span><span class="lang-pill">Malay</span><span class="lang-pill">Filipino</span><span class="lang-pill">Thai</span><span class="lang-pill">Vietnamese</span></div></div>
            <div class="region" style="--reg:#7d59a5"><h3>West Asia</h3><div class="langs"><span class="lang-pill">Arabic</span><span class="lang-pill">Turkish</span></div></div>
            <div class="region" style="--reg:#2272b4"><h3>Europe &amp; global</h3><div class="langs"><span class="lang-pill">English</span><span class="lang-pill">Danish</span><span class="lang-pill">Dutch</span><span class="lang-pill">Finnish</span><span class="lang-pill">French</span><span class="lang-pill">German</span><span class="lang-pill">Greek</span><span class="lang-pill">Italian</span><span class="lang-pill">Polish</span><span class="lang-pill">Portuguese</span><span class="lang-pill">Russian</span><span class="lang-pill">Spanish</span><span class="lang-pill">Swedish</span></div></div>
          </div>
          <div class="method">
            <div class="m-row"><span class="m-k">Model-supported</span><p class="m-v"><strong>STT ∩ TTS</strong> — the intersection Qwen3-ASR and VoxCPM2 both support.</p></div>
            <div class="m-row"><span class="m-k">Human-validated</span><p class="m-v">A subset (<strong>en, th, id, zh</strong>) round-trip tested with real speech; the rest still need a full human sweep.</p></div>
            <div class="m-row"><span class="m-k">Deciding metric</span><p class="m-v"><strong>Business-detail capture</strong> — names, account refs, invoice IDs, dates, amounts — over raw WER.</p></div>
          </div>
        </div>
        <div class="callout">Honesty boundary: "24" is the model-supported intersection — not a claim that all 24 received identical human validation.</div>
        ${footer(12, "Languages supported &amp; how they're validated")}
      </div>
    `,
  },
  {
    title: "Measured across languages",
    className: "dark",
    accent: "#28a9c7",
    notes:
      "A screenshot of the benchmark surface in the deployed app. It is not a decorative chart: the page reads measured FLEURS results from the benchmark data, distinguishes transcription error from latency and reliability, covers 24 languages, and marks the run's status. The screenshot keeps the evidence connected to the product.",
    html: `
      <div class="slide-inner product-proof-slide">
        <div class="product-proof-copy">
          <p class="eyebrow">Benchmark evidence · live app</p>
          <h2>Measured across 24 languages — inside the product.</h2>
          <p>FLEURS transcription quality, clean-turn reliability and speech latency are shown from the deployed benchmark run, not a hand-built slide.</p>
        </div>
        <div class="product-proof-image"><img src="benchmark-page.png" alt="Deployed Voice Benchmarks page showing 24 languages, reliability and speech latency"></div>
        ${footer(13, "Screenshot captured from the deployed Voice Benchmarks page")}
      </div>
    `,
  },
  {
    title: "Conversation-proven controls",
    className: "dark",
    accent: "#00a972",
    notes:
      "A screenshot of the deployed Model Control Plane page. It proves that the controls and model metrics are sourced from this app's conversation traces, not generic workspace traffic. The page shows verified provenance, speech endpoint telemetry for Qwen3-ASR and VoxCPM2, and the split between serving-endpoint inference tables and model-service controls.",
    html: `
      <div class="slide-inner product-proof-slide">
        <div class="product-proof-copy">
          <p class="eyebrow">Model control plane · live app</p>
          <h2>Controls backed by this app's conversations.</h2>
          <p>Verified provenance, per-model calls, errors, latency and AI Gateway evidence — filtered to conversation traffic from this application.</p>
        </div>
        <div class="product-proof-image"><img src="guardrails-page.png" alt="Deployed Model Control Plane page showing verified provenance and speech model telemetry"></div>
        ${footer(14, "Screenshot captured from the deployed Model Control Plane page")}
      </div>
    `,
  },
  {
    title: "AI Gateway control plane",
    className: "dark",
    accent: "#ff6b55",
    notes:
      "AI Gateway is the model control plane, with two capability tiers in this application. The Unity Catalog model services for Qwen3-Next and GPT-5.5 support service policies, rate limits, token metering, request tags and inference tables. The Qwen3-ASR and VoxCPM2 ResponsesAgent serving endpoints support inference tables and request telemetry. The limitation is explicit at the bottom: AI Gateway does not yet support voice guardrails for these STT/TTS agent endpoints.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">AI Gateway</p>
        <h2>One control plane, two supported capability tiers.</h2>
        <p class="section-note">The capability set follows the Databricks endpoint type: Unity Catalog model services receive the full policy and metering plane; ResponsesAgent speech endpoints receive inference evidence.</p>
        <div class="gateway-tiers">
          <section class="gateway-tier full-tier">
            <div class="gateway-tier-head"><span>Full Gateway controls</span><h3>Unity Catalog model services</h3></div>
            <div class="gateway-models"><b>Qwen3-Next 80B</b><b>GPT-5.5</b></div>
            <div class="gateway-controls">
              <span>Service policies</span><span>Rate limits</span><span>Token metering</span>
              <span>Request tags</span><span>Inference tables</span>
            </div>
            <p>Intent, summarization and translation calls are authorized, metered and logged through governed model services.</p>
          </section>
          <section class="gateway-tier evidence-tier">
            <div class="gateway-tier-head"><span>Inference evidence</span><h3>ResponsesAgent serving endpoints</h3></div>
            <div class="gateway-models"><b>Qwen3-ASR-1.7B</b><b>VoxCPM2</b></div>
            <div class="gateway-controls">
              <span>Inference tables</span><span>Request / response logs</span>
              <span>Latency</span><span>Status &amp; errors</span>
            </div>
            <p>Speech endpoints expose traceable inference evidence through their supported AI Gateway integration.</p>
          </section>
        </div>
        <div class="limitation-banner">
          <strong>Current limitation</strong>
          <span>AI Gateway does not yet support voice guardrails for STT/TTS ResponsesAgent endpoints. Today, those endpoints support inference tables and telemetry — not chat safety policies, rate limits or token metering.</span>
        </div>
        ${footer(15, "AI Gateway capabilities shown exactly as supported by each endpoint type")}
      </div>
    `,
  },
  {
    title: "Future improvements",
    className: "",
    accent: "#2272b4",
    notes:
      "The roadmap, with each item explicitly labeled planned or research. Planned: native realtime serving, deeper Genie reasoning, and broader AI Gateway coverage. Research: voice guardrails, lower-cost serving, and full human validation across all 24 configured languages. These are direction, not shipped claims.",
    html: `
      <div class="slide-inner">
        <p class="eyebrow">Future improvements</p>
        <h2>What we would improve next.</h2>
        <div class="roadmap-grid">
          <article class="roadmap-card planned"><span>Planned</span><h3>Native realtime serving</h3><p>Streaming and realtime multilingual voice directly on Model Serving.</p><strong>Outcome · simpler runtime, lower turn latency</strong></article>
          <article class="roadmap-card planned"><span>Planned</span><h3>Deeper Genie reasoning</h3><p>Richer Agent Mode investigations and Genie One workspace-level insights.</p><strong>Outcome · more questions resolved by voice</strong></article>
          <article class="roadmap-card planned"><span>Planned</span><h3>Broader AI Gateway coverage</h3><p>Extend the supported control plane across more of the speech path.</p><strong>Outcome · fewer control-plane differences</strong></article>
          <article class="roadmap-card research"><span>Research</span><h3>Voice guardrails</h3><p>Native STT/TTS safety controls inside AI Gateway.</p><strong>Outcome · voice-aware policy enforcement</strong></article>
          <article class="roadmap-card research"><span>Research</span><h3>Lower-cost serving</h3><p>CPU and smaller-model options that relax regional GPU dependency.</p><strong>Outcome · broader deployment economics</strong></article>
          <article class="roadmap-card research"><span>Research</span><h3>Full language validation</h3><p>Human round-trip review across all 24 configured languages.</p><strong>Outcome · production evidence per language</strong></article>
        </div>
        ${footer(16, "Deck: Future Improvements (planned / in research — not shipped)")}
      </div>
    `,
  },
  {
    title: "Genie's intelligence, in every conversation",
    className: "red",
    accent: "#ffffff",
    notes:
      "Close by returning to the promise — not on 'coming soon'. Genie's governed intelligence is now reachable by voice, in the language people actually speak, on open models hosted and governed on Databricks. Reinforce the thesis; don't introduce new architecture or roadmap here.",
    html: `
      <div class="slide-inner hero">
        <p class="eyebrow">Genie for Voice</p>
        <h1>Bring Genie's intelligence into every conversation.</h1>
        <p class="lede">Multilingual speech · open models · Genie reasoning · governed on Databricks — one deployable solution, in the language people naturally speak.</p>
        <p class="meta">go/genie-for-voice &nbsp;·&nbsp; ${commit("9861be7a", "first commit")} &nbsp;→&nbsp; ${commit("ad578361", "governed inference trail")}</p>
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
