export const APP_HASHES = new Set([
  "#/",
  "#/card",
  "#/knowledge",
  "#/setup",
  "#/traces",
  "#/guardrails",
  "#/voice-benchmarks",
  "#/asr-benchmark",
  "#/telco",
]);

export function resolveAppHash(hash: string): string {
  const path = (hash || "#/").split("?", 1)[0];
  return APP_HASHES.has(path) ? path : "#/";
}
