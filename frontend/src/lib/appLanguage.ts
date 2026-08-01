import { useEffect, useState } from "react";
import { DEFAULT_LANGUAGE } from "./languages";

/**
 * Global app language — chosen ONCE on the home page, then read (not changed) by
 * every use-case surface.
 *
 * Why this exists: language used to be picked per page, and changing it mid-call
 * tore down and reopened the voice session, which lagged and occasionally wedged
 * the reconnect. Selecting up front and locking it everywhere else removes that
 * churn entirely — the session opens in the right language from its first
 * greeting, so the choice always "reflects immediately".
 *
 * Persisted to localStorage so it survives navigation and refresh, and broadcast
 * so every mounted surface (App UI copy, telco/card/HLS pages) stays in sync.
 */
const STORAGE_KEY = "genie.appLanguage";
const CHANGE_EVENT = "genie:app-language";

export function getAppLanguage(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_LANGUAGE;
  } catch {
    return DEFAULT_LANGUAGE;
  }
}

export function setAppLanguage(code: string): void {
  if (!code || code === getAppLanguage()) return;
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch {
    // Non-persistent (private mode / blocked storage) is acceptable; the event
    // below still syncs live surfaces for this session.
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: code }));
}

/** Reactive view of the global language. `setLang` writes through to storage. */
export function useAppLanguage(): [string, (code: string) => void] {
  const [lang, setLang] = useState<string>(() => getAppLanguage());
  useEffect(() => {
    const sync = () => setLang(getAppLanguage());
    window.addEventListener(CHANGE_EVENT, sync);
    // Other tabs writing the same key.
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return [lang, setAppLanguage];
}
