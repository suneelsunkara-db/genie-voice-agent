// All runtime config comes from Vite env vars (VITE_*). No hardcoded URLs.
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const WS_BASE_URL: string = API_BASE_URL.replace(/^http/i, "ws");

export const POLL_INTERVAL_MS: number = Number(
  import.meta.env.VITE_POLL_INTERVAL_MS ?? 2000,
);
