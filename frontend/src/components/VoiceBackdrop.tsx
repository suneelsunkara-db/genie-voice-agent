import { useEffect, useMemo, useState } from "react";
import { fetchSupportedLanguages } from "../lib/languages";
import "../styles/voice-backdrop.css";

/**
 * Decorative fallback endonyms so the flow ALWAYS animates, even before (or
 * without) the config-driven language fetch resolving. The live set replaces
 * this once loaded; this only guarantees the visual never collapses to nothing.
 */
const FALLBACK_LABELS = [
  "English", "中文", "हिन्दी", "Español", "Français", "Deutsch", "日本語", "한국어",
  "Português", "Italiano", "Русский", "العربية", "Bahasa", "ไทย", "Tiếng Việt",
  "Nederlands", "Svenska", "Polski", "Türkçe", "Ελληνικά", "Filipino", "Dansk",
  "Suomi", "Norsk",
];

/**
 * Shared page backdrop (framework primitive): the deep-navy field with lava-red +
 * Genie-purple glows and the supported languages drifting like water. Every voice
 * surface (home, billing, card, knowledge) mounts this behind its content so the
 * background + language flow + color theme are IDENTICAL everywhere.
 *
 * Fixed + z-index:0; pages render their content in a `.gv-content` wrapper (or any
 * positioned element with z-index >= 1) so it sits above the flow. Pages should
 * make their own root background transparent so this shows through.
 */
export function VoiceBackdrop() {
  const [labels, setLabels] = useState<string[]>(FALLBACK_LABELS);

  useEffect(() => {
    let active = true;
    void (async () => {
      const langs = await fetchSupportedLanguages();
      // Only replace the fallback with the live set when it's rich enough to flow.
      if (active && langs.length >= 3) setLabels(langs.map((l) => l.label));
    })();
    return () => {
      active = false;
    };
  }, []);

  const rows = useMemo(() => {
    if (labels.length < 3) return [] as string[][];
    const doubled = [...labels, ...labels];
    const third = Math.ceil(doubled.length / 3);
    return [doubled.slice(0, third), doubled.slice(third, third * 2), doubled.slice(third * 2)];
  }, [labels]);

  return (
    <div className="gv-backdrop" aria-hidden>
      <div className="gv-langflow">
        {rows.map((row, i) => (
          <div key={i} className={`gv-langflow-row gv-langflow-row-${i % 3}`}>
            {row.map((label, j) => (
              <span key={`${label}-${j}`} className="gv-langflow-word">
                {label}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
