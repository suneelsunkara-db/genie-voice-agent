import { useEffect, useState } from "react";

/** The two server-allowlisted agent voice references. */
export type AppVoice = "female" | "male";

const STORAGE_KEY = "genie.appVoice";
const CHANGE_EVENT = "genie:app-voice";
const DEFAULT_VOICE: AppVoice = "female";

function isAppVoice(value: string | null): value is AppVoice {
  return value === "female" || value === "male";
}

/** Read the one voice choice inherited by every voice surface. */
export function getAppVoice(): AppVoice {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return isAppVoice(value) ? value : DEFAULT_VOICE;
  } catch {
    return DEFAULT_VOICE;
  }
}

export function setAppVoice(voice: AppVoice): void {
  if (voice === getAppVoice()) return;
  try {
    localStorage.setItem(STORAGE_KEY, voice);
  } catch {
    // The current tab still receives the event when persistence is unavailable.
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: voice }));
}

/** Reactive Home-page selector. Active calls keep their session-start voice. */
export function useAppVoice(): [AppVoice, (voice: AppVoice) => void] {
  const [voice, setVoice] = useState<AppVoice>(() => getAppVoice());
  useEffect(() => {
    const sync = () => setVoice(getAppVoice());
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return [voice, setAppVoice];
}
