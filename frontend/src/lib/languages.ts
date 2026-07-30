import { API_BASE_URL } from "../config";
import { languageLabel } from "../i18n";

/**
 * Framework-level language helpers, shared by every voice surface (home
 * concierge, card, healthcare). The supported set is fetched ONCE from the
 * config-driven `/languages` endpoint (STT ∩ TTS), so no page hardcodes a list
 * and every language bar matches what the pipeline can actually speak.
 */

export type Lang = { code: string; label: string };

export const DEFAULT_LANGUAGE = "en-US";
export const DEFAULT_LANGUAGE_OPTIONS: Lang[] = [{ code: DEFAULT_LANGUAGE, label: "English" }];

/** Native endonym for a BCP-47 code (e.g. "hi-IN" -> "हिन्दी"), via Intl.DisplayNames. */
export function nativeLangLabel(code: string): string {
  if (!code) return "English";
  return languageLabel(code, code);
}

/** Flag emoji from the region subtag ("en-US" -> 🇺🇸); a globe when there's none. */
export function flagFor(code: string): string {
  const region = (code || "").split("-").find((p, i) => i > 0 && /^[A-Za-z]{2}$/.test(p));
  if (!region) return "🌐";
  return region
    .toUpperCase()
    .replace(/./g, (c) => String.fromCodePoint(127397 + c.charCodeAt(0)));
}

async function _fetchOptions(path: string): Promise<Lang[]> {
  const r = await fetch(`${API_BASE_URL}${path}`);
  if (!r.ok) return [];
  const data = (await r.json()) as { options?: Array<{ code: string }> };
  return (data.options ?? []).map((o) => ({ code: o.code, label: nativeLangLabel(o.code) }));
}

/** The config-driven supported languages, as native-labelled options (English fallback).
 *
 * Prefers the framework `/languages` endpoint but falls back to `/card/languages`
 * (same payload) so an older/partially-deployed backend still yields the full set
 * instead of collapsing to English-only. */
export async function fetchSupportedLanguages(): Promise<Lang[]> {
  for (const path of ["/languages", "/card/languages"]) {
    try {
      const langs = await _fetchOptions(path);
      if (langs.length) return langs;
    } catch {
      /* try next path */
    }
  }
  return DEFAULT_LANGUAGE_OPTIONS;
}
