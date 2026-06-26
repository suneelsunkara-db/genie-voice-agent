import {
  H1,
  H2,
  H3,
  Text,
  Stack,
  Row,
  Grid,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Pill,
  Stat,
  Swatch,
  UsageBar,
  Button,
  Callout,
  Spacer,
  useCanvasState,
  useHostTheme,
  type Color,
} from "cursor/canvas";

type View = "platform" | "flow";
type ZoneId = "journey" | "ui" | "lakebase" | "setup";

type Node = {
  id: string;
  title: string;
  sub: string;
  zone: ZoneId;
  col: number;
  highlight?: boolean;
  /** Numbered step badge (journey band only). */
  step?: number;
  stepColor?: Color;
};

const ZONES: { id: ZoneId; label: string; color: Color }[] = [
  { id: "journey", label: "Agent assist journey (one customer turn)", color: "orange" },
  { id: "ui", label: "Agent Assist UI (React)", color: "pink" },
  { id: "lakebase", label: "Lakebase — live operational store", color: "cyan" },
  { id: "setup", label: "Customer Data and Genie space Setup", color: "green" },
];

const NODES: Node[] = [
  {
    id: "speaks",
    title: "Customer speaks",
    sub: "voice channel",
    zone: "journey",
    col: 0,
    step: 1,
    stepColor: "orange",
  },
  {
    id: "deepgram",
    title: "Deepgram STT",
    sub: "final utterances only",
    zone: "journey",
    col: 1,
    step: 2,
    stepColor: "orange",
    highlight: true,
  },
  {
    id: "fm_enrich",
    title: "FM enrich",
    sub: "intent · sentiment · signals",
    zone: "journey",
    col: 2,
    step: 3,
    stepColor: "blue",
    highlight: true,
  },
  {
    id: "agent_reply",
    title: "FM agent reply",
    sub: "Genie-grounded · UC-validated",
    zone: "journey",
    col: 3,
    step: 4,
    stepColor: "blue",
    highlight: true,
  },
  {
    id: "close_billing",
    title: "Issue closed",
    sub: "parameterized UC + Lakebase write",
    zone: "journey",
    col: 4,
    step: 5,
    stepColor: "yellow",
  },

  { id: "account_ctx", title: "Account context", sub: "customer_id · call_id", zone: "ui", col: 0 },
  { id: "genie_panel", title: "Genie console", sub: "account-scoped · UC NL→SQL", zone: "ui", col: 2, highlight: true },
  { id: "next_action", title: "Next best action", sub: "agent applies · FM + state machine", zone: "ui", col: 4, highlight: true },

  { id: "call_state", title: "call_state", sub: "nudge + resolution", zone: "lakebase", col: 0, highlight: true },
  { id: "utterances", title: "live_call_utterances", sub: "turn history", zone: "lakebase", col: 1 },
  { id: "lb_facts", title: "call_facts", sub: "live metrics", zone: "lakebase", col: 2 },
  { id: "resolution", title: "resolution_events", sub: "issue timeline", zone: "lakebase", col: 3 },
  { id: "billing_lb", title: "billing_adjustments", sub: "waiver / plan", zone: "lakebase", col: 4 },

  { id: "orchestration", title: "Orchestration job", sub: "ingest · gold · DQ", zone: "setup", col: 0 },
  { id: "raw_stream", title: "UC Volume", sub: "raw_streaming_data", zone: "setup", col: 1 },
  { id: "customer_data", title: "Customer Data", sub: "customers · invoices · payments", zone: "setup", col: 2 },
  { id: "gold", title: "gold_call_insights", sub: "FM-derived analytics", zone: "setup", col: 3 },
  { id: "genie_space", title: "AI/BI Genie Space", sub: "NL → SQL portfolio Q&A", zone: "setup", col: 4, highlight: true },
];

/** Horizontal arrows drawn only inside these zones. */
const INTRA_ZONE_EDGES: Record<ZoneId, [string, string][]> = {
  journey: [
    ["speaks", "deepgram"],
    ["deepgram", "fm_enrich"],
    ["fm_enrich", "agent_reply"],
    ["agent_reply", "close_billing"],
  ],
  ui: [
    ["account_ctx", "genie_panel"],
    ["genie_panel", "next_action"],
  ],
  lakebase: [],
  setup: [
    ["orchestration", "raw_stream"],
    ["raw_stream", "customer_data"],
    ["customer_data", "gold"],
    ["gold", "genie_space"],
  ],
};

const USER_STEPS: {
  n: number;
  title: string;
  detail: string;
  color: Color;
  actors: string;
}[] = [
  {
    n: 1,
    title: "Queue triage",
    detail: "Agent opens cockpit; sidebar loads GET /accounts/with-issues (at-risk, overdue, disputes). CUST-4028 pinned.",
    color: "pink",
    actors: "Agent → UI → Lakebase / UC",
  },
  {
    n: 2,
    title: "Account context + Genie insight prefetch",
    detail: "Selecting a customer hydrates account facts, invoices, and live call_state from Lakebase (no LLM). In parallel, POST /calls/{id}/genie-insight warms a Genie natural-language account snapshot OFF the live reply path and caches it in call_state.",
    color: "cyan",
    actors: "UI → Lakebase · API → Genie (async)",
  },
  {
    n: 3,
    title: "Customer speaks",
    detail: "Mic stream or typed turn → Deepgram emits a final utterance (not per-chunk FM).",
    color: "orange",
    actors: "Customer → Deepgram → API",
  },
  {
    n: 4,
    title: "FM enrich",
    detail: "POST /assist runs one structured Foundation Model call to detect intent, sentiment, customer_signal, and waiver/plan flags. Intent detection is the FM's job — not Genie's.",
    color: "blue",
    actors: "API → Foundation Model",
  },
  {
    n: 5,
    title: "Resolution journey",
    detail: "State machine advances open → in_progress → pending_close. UI strip shows describe → understand → review → offer → apply → close.",
    color: "purple",
    actors: "API → resolution_events",
  },
  {
    n: 6,
    title: "FM agent reply — Genie-grounded",
    detail: "The FM composes the prose reply. Numbers come from deterministic authoritative metrics (Lakebase/UC); the cached Genie insight grounds the opener so 'Based on Genie insights' is truthful; the reply is validated against governed UC metrics before it can close.",
    color: "blue",
    actors: "API → FM · validated vs Genie/UC",
  },
  {
    n: 7,
    title: "Issue closed",
    detail: "On confirm_proceed, billing_adjustments commit to Lakebase + UC invoices via injection-safe parameterized SQL (Statement Execution API); issue status → closed.",
    color: "yellow",
    actors: "API → Lakebase → UC",
  },
];

/** Account-scoped Genie probe inside the cockpit (decision aid, not the write path). */
const CONSOLE_CHAIN: { color: Color; title: string; desc: string }[] = [
  { color: "cyan", title: "Account context", desc: "selected customer_id + call_id" },
  { color: "pink", title: "Genie console", desc: "seeded NL question · POST /genie/ask" },
  { color: "green", title: "Genie space", desc: "NL → SQL over curated UC" },
  { color: "purple", title: "Curated UC tables", desc: "customers · invoices · payments · gold" },
  { color: "blue", title: "Facts → next action", desc: "informs agent; FM + state machine execute" },
];

const PAD = 20;
const AREA_X = PAD + 16;
const BOX_W = 150;
const BOX_H = 56;
const COL_STEP = 208;
const BAND_H = 104;
const BAND_STEP = 124;
const ARROW_GAP = 8;

const MIN_ZOOM = 0.55;
const MAX_ZOOM = 2.2;

function zoneIndex(z: ZoneId) {
  return ZONES.findIndex((zz) => zz.id === z);
}
function boxX(col: number) {
  return AREA_X + col * COL_STEP;
}
function boxY(z: ZoneId) {
  return PAD + zoneIndex(z) * BAND_STEP + BAND_H - BOX_H - 14;
}

function nodeById(id: string) {
  const n = NODES.find((node) => node.id === id);
  if (!n) throw new Error(`Unknown node: ${id}`);
  return n;
}

function horizontalAnchors(id: string) {
  const n = nodeById(id);
  const x = boxX(n.col);
  const y = boxY(n.zone);
  return {
    rightX: x + BOX_W,
    leftX: x,
    midY: y + BOX_H / 2,
    accent: n.zone,
  };
}

export default function GenieVoiceArchitectureCanvas() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<View>("gva-view", "platform");
  const [zoom, setZoom] = useCanvasState("gva-zoom", 0.85);

  const clamp = (z: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
  const zoneColor = (z: ZoneId) => theme.category[ZONES.find((zz) => zz.id === z)!.color];

  const maxCol = Math.max(...NODES.map((n) => n.col));
  const svgW = AREA_X + maxCol * COL_STEP + BOX_W + PAD;
  const svgH = PAD + ZONES.length * BAND_STEP + PAD;

  return (
    <Stack gap={18} style={{ padding: 20, background: theme.bg.editor }}>
      <Stack gap={6}>
        <H1>Genie Voice Agent — System Architecture</H1>
        <Text tone="secondary">
          Lakebase-first live assist on Databricks: streaming capture, governed Unity Catalog analytics,
          and Genie for portfolio intelligence — without token-maxing every spoken syllable.
        </Text>
      </Stack>

      <Row gap={8} align="center" wrap>
        <Pill active={view === "platform"} onClick={() => setView("platform")}>
          Databricks platform
        </Pill>
        <Pill active={view === "flow"} onClick={() => setView("flow")}>
          Live user flow
        </Pill>
        <Spacer />
        {view === "platform" && (
          <Row gap={6} align="center">
            <Button variant="secondary" onClick={() => setZoom((z) => clamp(Math.round((z - 0.1) * 100) / 100))}>
              −
            </Button>
            <Button variant="secondary" onClick={() => setZoom((z) => clamp(Math.round((z + 0.1) * 100) / 100))}>
              +
            </Button>
            <Button variant="ghost" onClick={() => setZoom(0.85)}>
              Reset
            </Button>
            <Text size="small" tone="secondary">
              {Math.round(zoom * 100)}%
            </Text>
          </Row>
        )}
      </Row>

      {view === "platform" ? (
        <Stack gap={14}>
          <Row gap={12} wrap>
            {ZONES.map((z) => (
              <div key={z.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Swatch color={z.color} style={{ width: 14, height: 14, borderRadius: 4 }} />
                <Text size="small" tone="secondary">
                  {z.label}
                </Text>
              </div>
            ))}
          </Row>

          <div
            style={{
              overflow: "auto",
              maxWidth: "100%",
              border: `1px solid ${theme.stroke.tertiary}`,
              borderRadius: 10,
            }}
          >
            <svg width={svgW * zoom} height={svgH * zoom} viewBox={`0 0 ${svgW} ${svgH}`} style={{ display: "block" }}>
              <defs>
                <marker
                  id="gva-h-arrow"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={theme.stroke.primary} />
                </marker>
              </defs>

              {ZONES.map((z, i) => {
                const y = PAD + i * BAND_STEP;
                const accent = theme.category[z.color];
                return (
                  <g key={z.id}>
                    <rect
                      x={PAD}
                      y={y}
                      width={svgW - PAD * 2}
                      height={BAND_H}
                      rx={10}
                      fill={theme.fill.tertiary}
                      stroke={accent}
                      strokeWidth={1.5}
                    />
                    <rect x={PAD} y={y} width={6} height={BAND_H} rx={3} fill={accent} />
                    <text x={PAD + 16} y={y + 22} fill={accent} fontSize={12} fontWeight={700}>
                      {z.label}
                    </text>
                  </g>
                );
              })}

              {(Object.entries(INTRA_ZONE_EDGES) as [ZoneId, [string, string][]][]).flatMap(([zone, edges]) =>
                edges.map(([from, to], i) => {
                  const a = horizontalAnchors(from);
                  const b = horizontalAnchors(to);
                  const fromNode = nodeById(from);
                  const stroke =
                    fromNode.stepColor != null
                      ? theme.category[fromNode.stepColor]
                      : theme.category[ZONES.find((z) => z.id === zone)!.color];
                  return (
                    <line
                      key={`${zone}-${i}`}
                      x1={a.rightX + ARROW_GAP}
                      y1={a.midY}
                      x2={b.leftX - ARROW_GAP}
                      y2={b.midY}
                      stroke={stroke}
                      strokeWidth={1.8}
                      markerEnd="url(#gva-h-arrow)"
                    />
                  );
                }),
              )}

              {NODES.map((n) => {
                const x = boxX(n.col);
                const y = boxY(n.zone);
                const accent = n.stepColor != null ? theme.category[n.stepColor] : zoneColor(n.zone);
                const textX = n.step != null ? x + 36 : x + 10;
                const cy = y + BOX_H / 2;
                return (
                  <g key={n.id}>
                    <rect
                      x={x}
                      y={y}
                      width={BOX_W}
                      height={BOX_H}
                      rx={8}
                      fill={n.highlight ? theme.fill.secondary : theme.fill.primary}
                      stroke={accent}
                      strokeWidth={n.highlight ? 2.2 : 1.4}
                    />
                    <rect x={x} y={y} width={BOX_W} height={4} rx={2} fill={accent} />
                    {n.step != null && (
                      <>
                        <circle cx={x + 16} cy={cy} r={13} fill={accent} />
                        <text
                          x={x + 16}
                          y={cy + 4}
                          textAnchor="middle"
                          fill={theme.text.onAccent}
                          fontSize={11}
                          fontWeight={700}
                        >
                          {n.step}
                        </text>
                      </>
                    )}
                    <text x={textX} y={y + 24} fill={theme.text.primary} fontSize={11.5} fontWeight={600}>
                      {n.title}
                    </text>
                    <text x={textX} y={y + 40} fill={theme.text.tertiary} fontSize={9.5}>
                      {n.sub}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <Grid columns={4} gap={12}>
            <Stat value="Lakebase" label="Hot-path serving" tone="info" />
            <Stat value="Foundation Model" label="Intent + prose reply" tone="info" />
            <Stat value="Genie" label="Insight + validation" tone="success" />
            <Stat value="Deepgram" label="STT per utterance" tone="info" />
          </Grid>
        </Stack>
      ) : (
        <Stack gap={16}>
          <Callout tone="info" title="Spotlight scenario">
            CUST-4028 / CALL-2028 (Omar Patel) — overdue invoice with late-fee waiver and payment-plan resolution path.
          </Callout>

          <H2>Agent assist journey (one customer turn)</H2>
          <Stack gap={10}>
            {USER_STEPS.map((step) => (
              <div key={step.n} style={{ display: "flex", gap: 14, alignItems: "stretch" }}>
                <div
                  style={{
                    minWidth: 44,
                    width: 44,
                    height: 44,
                    borderRadius: 22,
                    background: theme.category[step.color],
                    color: theme.text.onAccent,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 700,
                    fontSize: 16,
                    flexShrink: 0,
                  }}
                >
                  {step.n}
                </div>
                <div
                  style={{
                    flex: 1,
                    minWidth: 240,
                    borderLeft: `4px solid ${theme.category[step.color]}`,
                    paddingLeft: 14,
                    paddingTop: 4,
                    paddingBottom: 4,
                  }}
                >
                  <Row gap={8} align="center" wrap>
                    <Text weight="semibold">{step.title}</Text>
                    <Pill size="sm">{step.actors}</Pill>
                  </Row>
                  <Text tone="secondary" size="small">
                    {step.detail}
                  </Text>
                </div>
              </div>
            ))}
          </Stack>

          <Divider />

          <H2>Token economics vs. naive agentic voice</H2>
          <Text tone="secondary" size="small">
            Source: architecture design · per-call budget model · FM once per finalized utterance · Genie insight once per call open
          </Text>
          <UsageBar
            total={100}
            topLeftLabel="Cost / complexity budget per call"
            topRightLabel="Lakebase reads = 0 LLM tokens"
            segments={[
              { id: "lakebase", value: 30, color: "cyan" },
              { id: "stt", value: 15, color: "orange" },
              { id: "fm", value: 40, color: "blue" },
              { id: "genie", value: 15, color: "green" },
            ]}
          />
          <Grid columns={4} gap={10}>
            {[
              ["cyan", "Lakebase reads", "State, account overlay, timeline — 0 tokens"],
              ["orange", "Deepgram STT", "Audio → final utterance only"],
              ["blue", "Foundation Model", "One structured + prose call per turn (main token cost)"],
              ["green", "Genie", "Account insight off-path (per call) + fact validation"],
            ].map(([color, title, desc]) => (
              <div key={title as string} style={{ borderLeft: `3px solid ${theme.category[color as Color]}` }}>
                <Card variant="borderless">
                  <CardBody style={{ padding: "10px 12px" }}>
                    <Row gap={6} align="center">
                      <Swatch color={color as Color} style={{ width: 10, height: 10, borderRadius: 2 }} />
                      <Text weight="medium" size="small">
                        {title as string}
                      </Text>
                    </Row>
                    <Text tone="tertiary" size="small">
                      {desc as string}
                    </Text>
                  </CardBody>
                </Card>
              </div>
            ))}
          </Grid>

          <Divider />

          <H2>Genie console — account-scoped UC probe</H2>
          <Text tone="secondary" size="small">
            In the cockpit, the Genie console is seeded with the selected customer's account context and
            answers from curated UC tables via the Genie space. It informs the agent's resolution decision —
            it does not perform the billing write itself.
          </Text>
          <Row gap={8} align="center" wrap>
            {CONSOLE_CHAIN.map((c, i) => (
              <div key={c.title} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {i > 0 && <Text tone="tertiary">→</Text>}
                <div
                  style={{
                    borderLeft: `3px solid ${theme.category[c.color]}`,
                    paddingLeft: 10,
                    minWidth: 150,
                  }}
                >
                  <Text weight="medium" size="small">
                    {c.title}
                  </Text>
                  <Text tone="tertiary" size="small">
                    {c.desc}
                  </Text>
                </div>
              </div>
            ))}
          </Row>

          <Divider />

          <H2>Consumption surfaces</H2>
          <Grid columns={2} gap={14}>
            <Card>
              <CardHeader trailing={<Swatch color="pink" style={{ width: 12, height: 12, borderRadius: 3 }} />}>
                Agent Assist cockpit
              </CardHeader>
              <CardBody>
                <Stack gap={6}>
                  <Text size="small">Customers-with-issues sidebar (GET /accounts/with-issues)</Text>
                  <Text size="small">Chat + mic stream (WS /mic-stream)</Text>
                  <Text size="small">Resolution journey strip (pipeline_steps)</Text>
                  <Text size="small">Genie console: NL answer + follow-up chips (SQL hidden)</Text>
                </Stack>
              </CardBody>
            </Card>
            <Card>
              <CardHeader trailing={<Swatch color="green" style={{ width: 12, height: 12, borderRadius: 3 }} />}>
                Genie & analytics
              </CardHeader>
              <CardBody>
                <Stack gap={6}>
                  <Text size="small">Account insight prefetch grounds the reply (off critical path)</Text>
                  <Text size="small">/genie/ask: NL answer + follow-ups + conversation context</Text>
                  <Text size="small">Fact validation vs governed UC metrics</Text>
                  <Text size="small">Single-customer snapshot in space instructions</Text>
                </Stack>
              </CardBody>
            </Card>
          </Grid>

          <H3>Impact</H3>
          <Grid columns={3} gap={12}>
            <Callout tone="success" title="Streaming insights">
              Customer context refreshes on each turn while the agent stays on the call.
            </Callout>
            <Callout tone="info" title="Faster resolution">
              Pre-loaded facts and utterance-bound billing close reduce hold time.
            </Callout>
            <Callout tone="warning" title="No token maxing">
              LLM spend scales with turns, not audio frames or full-history re-prompts.
            </Callout>
          </Grid>
        </Stack>
      )}
    </Stack>
  );
}
