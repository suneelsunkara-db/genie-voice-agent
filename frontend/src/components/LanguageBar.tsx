import { Lang, flagFor } from "../lib/languages";
import "../styles/langbar.css";

/**
 * Shared voice-language picker (framework primitive). Every voice surface uses
 * this SAME bar so the TTS/STT language is chosen identically everywhere. The
 * option set is the config-driven supported catalog (see lib/languages).
 *
 * Pre-call it just records the choice; in-call the parent restarts the session
 * in the new language (session config is immutable after start).
 */
export function LanguageBar({
  value,
  options,
  onChange,
  disabled,
  label = "Voice language",
}: {
  value: string;
  options: Lang[];
  onChange: (code: string) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <label className="langbar">
      <span className="langbar-label">
        {label} · {options.length} languages
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
