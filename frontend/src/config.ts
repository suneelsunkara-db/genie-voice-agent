// All runtime config comes from Vite env vars (VITE_*). No hardcoded URLs.
//   - unset            -> local dev default (API on :8000, separate Vite server)
//   - set to a URL     -> use it verbatim
//   - set to "" (empty)-> same-origin (Databricks Apps: UI + API are one process)
// `import.meta.env` is Vite-only; guard so these modules stay importable under
// plain Node (used by the offline i18n extraction tool).
const viteEnv = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
const configuredApiBase = viteEnv.VITE_API_BASE_URL;

export const API_BASE_URL: string =
  configuredApiBase === undefined
    ? "http://localhost:8000"
    : configuredApiBase.length > 0
      ? configuredApiBase
      : typeof window !== "undefined"
        ? window.location.origin
        : "";

export const WS_BASE_URL: string = API_BASE_URL.replace(/^http/i, "ws");

export const POLL_INTERVAL_MS: number = Number(
  viteEnv.VITE_POLL_INTERVAL_MS ?? 2000,
);
