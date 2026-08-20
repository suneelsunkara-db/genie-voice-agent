import { InteractionLanguage } from "../api/client";
import { uiCopy } from "../i18n";
import { Lang, flagFor } from "../lib/languages";
import "../styles/langbar.css";

/**
 * Shared voice-language picker (framework primitive). Every voice surface uses
 * this SAME bar so the TTS/STT language is chosen identically everywhere. The
 * option set is the config-driven supported catalog (see lib/languages).
 *
 * Pre-call it just records the choice; in-call the parent restarts the session
 * in the new language (session config is immutable after start).
 *
 * Its own label follows the selected language, so the control a caller uses to
 * pick their language is not itself stuck in English.
 */
export function LanguageBar({
  value,
  options,
  onChange,
  disabled,
  label,
}: {
  value: string;
  options: Lang[];
  onChange: (code: string) => void;
  disabled?: boolean;
  label?: string;
}) {
  const copy = uiCopy(value as InteractionLanguage);
  return (
    <label className="langbar">
      <span className="langbar-label">
        {copy.langbarLabel(label ?? copy.voiceLanguage, String(options.length))}
      </span>
      <span className="langbar-control">
        <span className="langbar-flag" aria-hidden>
          {flagFor(value)}
        </span>
        <select
          className="langbar-select"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((o) => (
            <option key={o.code} value={o.code}>
              {o.label}
            </option>
          ))}
        </select>
      </span>
    </label>
  );
}
