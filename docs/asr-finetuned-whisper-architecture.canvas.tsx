// @ts-nocheck
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
  Callout,
  useHostTheme,
  type Color,
} from "cursor/canvas";

type Phase = {
  id: string;
  step: string;
  title: string;
  script: string;
  command: string;
  output: string;
  color: Color;
};

type Artifact = {
  name: string;
  location: string;
  purpose: string;
};

type Decision = {
  label: string;
  value: string;
  note: string;
  color: Color;
};

const PHASES: Phase[] = [
  {
    id: "data",
    step: "01",
    title: "Build ASR training corpus",
    script: "scripts/asr/01_asr_model_training.sh",
    command: "prepare, synthesize, augment, manifest, baselines",
    output: "Gold JSONL manifests and audio in UC Volumes",
    color: "cyan",
  },
  {
    id: "baseline",
    step: "02",
    title: "Run fair baselines",
    script: "scripts/asr/02_asr_baseline_runs.sh",
    command: "Deepgram + base Whisper on the same manifest",
    output: "WER, CER, entity accuracy, and latency comparisons",
    color: "blue",
  },
  {
    id: "finetune",
    step: "03",
    title: "Fine-tune Whisper with LoRA",
    script: "scripts/asr/03_asr_model_finetuning.sh",
    command: "train-lora, evaluate-lora, rescore, error analysis",
    output: "LoRA adapter, processor, training metrics, evaluation reports",
    color: "purple",
  },
  {
    id: "holdout",
    step: "04",
    title: "Holdout gate",
    script: "scripts/asr/04_asr_real_audio_holdout.sh",
    command: "prepare, validate, evaluate",
    output: "Real or realistic held-out decision report",
    color: "orange",
  },
  {
    id: "register",
    step: "05",
    title: "Register UC model candidate",
    script: "scripts/asr/05_register_asr_model_candidate.sh",
    command: "register-candidate, smoke-test-candidate",
    output: "UC model version with alias candidate and pyfunc contract",
    color: "green",
  },
  {
    id: "serve",
    step: "06",
    title: "Deploy Model Serving endpoint",
    script: "scripts/asr/06_deploy_asr_model_serving_endpoint.sh",
    command: "deploy, status, smoke-test",
    output: "voice_finetuned_whisper_model endpoint for app STT",
    color: "yellow",
  },
];

const ARTIFACTS: Artifact[] = [
  {
    name: "Training manifest",
    location: "/Volumes/.../asr_model_training/datasets/gold/manifests/asr_training_gold_v1.jsonl",
    purpose: "Locked utterance-level clips used for fair baseline and LoRA evaluation.",
  },
  {
    name: "LoRA run",
    location: "/Volumes/.../model_artifacts/lora_runs/lora_20260627_041557",
    purpose: "Adapter, processor, run_config.json, and train_metrics.json.",
  },
  {
    name: "Registered UC model",
    location: "partner_demo_catalog.genie_voice_contact_center.genie_asr_whisper_lora",
    purpose: "Governed candidate model version with alias candidate.",
  },
  {
    name: "Serving endpoint",
    location: "voice_finetuned_whisper_model",
    purpose: "Warm Databricks Model Serving endpoint for app utterance transcription.",
  },
];

const DECISIONS: Decision[] = [
  {
    label: "Model identity",
    value: "voice_finetuned_whisper_model",
    note: "App-facing endpoint name avoids Genie and LoRA implementation details.",
    color: "green",
  },
  {
    label: "Runtime contract",
    value: "audio_b64, mime_type, speaker -> transcript",
    note: "Matches the browser upload path used by /calls/{call_id}/mic-transcribe.",
    color: "blue",
  },
  {
    label: "Streaming posture",
    value: "Utterance-level, not streaming",
    note: "Fine-tuned Whisper returns final transcript after Stop; Deepgram remains switchable.",
    color: "orange",
  },
  {
    label: "Serving compute",
    value: "GPU_SMALL, Small, scale_to_zero=false",
    note: "GPU redeploy moves Whisper inference from CPU to CUDA for lower final latency.",
    color: "purple",
  },
];

const APP_FLOW = [
  "config providers.stt.active selects deepgram or databricks",
  "React status poll reads stt_provider from /status",
  "Deepgram mode uses WebSocket /mic-stream for interim text",
  "Databricks mode records an utterance and posts audio_b64 to /mic-transcribe",
  "API calls voice_finetuned_whisper_model and applies invoice-ID postprocessing",
  "Assist flow reuses the returned transcript for Genie-grounded agent guidance",
];

const RISKS = [
  "No interim words for pure Databricks mode until a streaming ASR wrapper is built.",
  "Synthetic holdout is only a workflow gate; real recorded holdout is still required.",
  "Always-warm GPU serving improves latency but increases fixed serving cost.",
  "Old experimental endpoints should be removed when no longer needed to avoid spend.",
];

function Surface({ children }: { children: any }) {
  const theme = useHostTheme();
  return (
    <div
      style={{
        minHeight: "100vh",
        background: theme.background,
        color: theme.foreground,
        padding: 28,
        boxSizing: "border-box",
      }}
    >
      {children}
    </div>
  );
}

function PhaseCard({ phase }: { phase: Phase }) {
  return (
    <Card>
      <CardHeader>
        <Row align="center" justify="between">
          <Pill color={phase.color}>{phase.step}</Pill>
          <Pill>{phase.script.split("/").pop()}</Pill>
        </Row>
      </CardHeader>
      <CardBody>
        <Stack gap={8}>
          <H3>{phase.title}</H3>
          <Text size="sm">{phase.command}</Text>
          <Divider />
          <Text size="sm" muted>
            {phase.output}
          </Text>
        </Stack>
      </CardBody>
    </Card>
  );
}

function DecisionCard({ decision }: { decision: Decision }) {
  return (
    <Card>
      <CardBody>
        <Stack gap={8}>
          <Pill color={decision.color}>{decision.label}</Pill>
          <H3>{decision.value}</H3>
          <Text size="sm" muted>
            {decision.note}
          </Text>
        </Stack>
      </CardBody>
    </Card>
  );
}

function ArtifactRow({ artifact }: { artifact: Artifact }) {
  const theme = useHostTheme();
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1.7fr 2fr",
        gap: 12,
        padding: "10px 0",
        borderBottom: `1px solid ${theme.border}`,
      }}
    >
      <Text weight="semibold">{artifact.name}</Text>
      <Text size="sm">{artifact.location}</Text>
      <Text size="sm" muted>
        {artifact.purpose}
      </Text>
    </div>
  );
}

function NumberedItem({ index, text }: { index: number; text: string }) {
  return (
    <Row gap={10} align="start">
      <Pill color="cyan">{String(index + 1).padStart(2, "0")}</Pill>
      <Text size="sm">{text}</Text>
    </Row>
  );
}

export default function ASRFinetunedWhisperArchitectureCanvas() {
  return (
    <Surface>
      <Stack gap={24}>
        <Stack gap={10}>
          <Row align="center" gap={10}>
            <Pill color="purple">ASR model build</Pill>
            <Pill>scripts/asr source of truth</Pill>
            <Pill>Databricks Model Serving</Pill>
          </Row>
          <H1>Fine-tuned Whisper Voice Model Architecture</H1>
          <Text>
            This canvas documents how the Databricks-hosted fine-tuned Whisper model was
            built, registered, deployed, and wired into the app as a switchable STT
            alternative to Deepgram.
          </Text>
        </Stack>

        <Grid columns={4} gap={12}>
          <Stat label="ASR scripts" value="06" />
          <Stat label="Registered alias" value="candidate" />
          <Stat label="Serving endpoint" value="voice_finetuned_whisper_model" />
          <Stat label="Current app switch" value="providers.stt.active" />
        </Grid>

        <Callout title="Key runtime distinction">
          The fine-tuned Whisper endpoint is utterance-level ASR. It replaces Deepgram in the
          final transcription path when providers.stt.active is databricks, but it does not
          produce interim words while the customer is speaking.
        </Callout>

        <Stack gap={12}>
          <H2>Numbered Build And Deployment Flow</H2>
          <Grid columns={3} gap={12}>
            {PHASES.map((phase) => (
              <PhaseCard key={phase.id} phase={phase} />
            ))}
          </Grid>
        </Stack>

        <Grid columns={2} gap={16}>
          <Stack gap={12}>
            <H2>Important Architecture Decisions</H2>
            <Grid columns={2} gap={12}>
              {DECISIONS.map((decision) => (
                <DecisionCard key={decision.label} decision={decision} />
              ))}
            </Grid>
          </Stack>

          <Stack gap={12}>
            <H2>App Runtime Flow</H2>
            <Card>
              <CardBody>
                <Stack gap={10}>
                  {APP_FLOW.map((item, index) => (
                    <NumberedItem key={item} index={index} text={item} />
                  ))}
                </Stack>
              </CardBody>
            </Card>
          </Stack>
        </Grid>

        <Stack gap={12}>
          <H2>Artifacts And Lineage</H2>
          <Card>
            <CardBody>
              <Stack gap={0}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1.7fr 2fr",
                    gap: 12,
                    paddingBottom: 10,
                  }}
                >
                  <Text size="sm" weight="semibold">
                    Artifact
                  </Text>
                  <Text size="sm" weight="semibold">
                    Location
                  </Text>
                  <Text size="sm" weight="semibold">
                    Purpose
                  </Text>
                </div>
                {ARTIFACTS.map((artifact) => (
                  <ArtifactRow key={artifact.name} artifact={artifact} />
                ))}
              </Stack>
            </CardBody>
          </Card>
        </Stack>

        <Grid columns={2} gap={16}>
          <Card>
            <CardHeader>
              <H2>Serving Compute Posture</H2>
            </CardHeader>
            <CardBody>
              <Stack gap={10}>
                <Row gap={8}>
                  <Pill color="purple">GPU_SMALL</Pill>
                  <Pill>Small</Pill>
                  <Pill>scale_to_zero=false</Pill>
                </Row>
                <Text size="sm">
                  The endpoint script now supports workload_type and defaults this Whisper
                  endpoint to GPU-backed serving. The model wrapper uses CUDA when available,
                  so GPU serving should reduce final transcript latency after Stop.
                </Text>
                <Text size="sm" muted>
                  Source: scripts/asr/06_deploy_asr_model_serving_endpoint.sh.
                </Text>
              </Stack>
            </CardBody>
          </Card>

          <Card>
            <CardHeader>
              <H2>Remaining Risks And Gates</H2>
            </CardHeader>
            <CardBody>
              <Stack gap={10}>
                {RISKS.map((risk, index) => (
                  <NumberedItem key={risk} index={index} text={risk} />
                ))}
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>
    </Surface>
  );
}
