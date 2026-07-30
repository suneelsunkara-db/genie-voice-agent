import type { ReactNode } from "react";
import type { InteractionLanguage, InteractionLanguageOption } from "../../api/client";
import databricksLogo from "../../assets/databricks-logo.png";
import genieLogo from "../../assets/genie-logo.png";
import { VoiceBackdrop } from "../VoiceBackdrop";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export function letterAt(index: number) {
  return LETTERS[index] ?? String(index + 1);
}

export function SentientShell({ children }: { children: ReactNode }) {
  return (
    <div className="sentient-app">
      {/* Shared background + flowing languages — same as every other voice surface.
          No scrim here: it would sit above the backdrop and wash out the flow. */}
      <VoiceBackdrop />
      <div className="sentient-frame">{children}</div>
    </div>
  );
}

function renderProductTitle(title: string) {
  if (title.startsWith("Genie")) {
    return (
      <>
        <span className="sentient-session-title-accent">Genie</span>
        {title.slice(5)}
      </>
    );
  }
  return title;
}

export function SentientBrandLockup({
  title = "GenieCanTalk",
  className,
}: {
  title?: string;
  className?: string;
}) {
  return (
    <div className={["sentient-session-mark", className].filter(Boolean).join(" ")}>
      <div className="sentient-session-lockup">
        <div className="sentient-session-logo-bar" aria-hidden="true">
          <img src={databricksLogo} alt="" className="sentient-session-logo sentient-session-logo-dbx" />
          <span className="sentient-session-logo-divider" />
          <img src={genieLogo} alt="" className="sentient-session-logo sentient-session-logo-genie" />
        </div>
        <h1 className="sentient-session-title">{renderProductTitle(title)}</h1>
      </div>
    </div>
  );
}

export function SentientSessionHead({
  title = "GenieCanTalk",
  kicker,
  name,
  callLabel,
  issues,
}: {
  title?: string;
  kicker: string;
  name: string;
  callLabel?: string;
  issues?: Array<{ id: string; label: string; warn?: boolean }>;
}) {
  return (
    <div className="sentient-session-head">
      <SentientBrandLockup title={title} />
      <div className="sentient-session-copy">
        <span className="sentient-session-kicker">{kicker}</span>
        <strong className="sentient-session-name">{name}</strong>
        {callLabel && <span className="sentient-session-call">{callLabel}</span>}
        {issues && issues.length > 0 && (
          <div className="sentient-session-issues" aria-label={kicker}>
            {issues.map((tag) => (
              <span
                key={tag.id}
                className={`sentient-session-issue${tag.warn ? " is-warn" : ""}`}
              >
                {tag.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function SentientStep({
  step,
  title,
  description,
  children,
  compact,
}: {
  step: number;
  title: string;
  description?: string;
  children?: ReactNode;
  compact?: boolean;
}) {
  return (
    <section className={`sentient-step${compact ? " sentient-step-compact" : ""}`}>
      <div className="sentient-step-index" aria-hidden="true">
        {step}
      </div>
      <div className="sentient-step-content">
        <h1 className="sentient-title">{title}</h1>
        {description && <p className="sentient-desc">{description}</p>}
        {children}
      </div>
    </section>
  );
}

export function SentientHCol({
  step,
  title,
  description,
  children,
  className,
}: {
  step: number;
  title: string;
  description?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <section className={["sentient-h-col", className].filter(Boolean).join(" ")}>
      <header className="sentient-h-col-head">
        <span className="sentient-step-index" aria-hidden="true">
          {step}
        </span>
        <div className="sentient-h-col-titles">
          <h2 className="sentient-h-title">{title}</h2>
          {description && <p className="sentient-h-desc">{description}</p>}
        </div>
      </header>
      <div className="sentient-h-col-body">{children}</div>
    </section>
  );
}

export interface TopBarHighlight {
  label: string;
  accent: "insight" | "resolution" | "hold" | "cost" | "reasoning";
}

export function SentientTopBar({
  contextKicker,
  contextDesc,
  highlights,
  languagesNote,
  languageChips,
  languageLabel,
  options,
  value,
  onChange,
}: {
  contextKicker: string;
  contextDesc: string;
  highlights?: TopBarHighlight[];
  languagesNote?: string;
  languageChips?: string[];
  languageLabel: string;
  options: InteractionLanguageOption[];
  value: InteractionLanguage;
  onChange: (language: InteractionLanguage) => void;
}) {
  return (
    <div className="sentient-topbar">
      <SentientBrandLockup />
      <div className="sentient-topbar-context">
        <span className="sentient-topbar-kicker">{contextKicker}</span>
        <p className="sentient-topbar-desc">{contextDesc}</p>
        {highlights && highlights.length > 0 && (
          <div className="sentient-topbar-highlights">
            {highlights.map((hl) => (
              <span key={hl.label} className="sentient-hl" data-accent={hl.accent}>
                <span aria-hidden className="sentient-hl-dot" />
                {hl.label}
              </span>
            ))}
          </div>
        )}
        {languagesNote && (
          <p className="sentient-topbar-langs">
            <span className="sentient-topbar-langs-note">{languagesNote}</span>
            {(languageChips ?? []).map((name) => (
              <span key={name} className="sentient-topbar-lang-chip">
                {name}
              </span>
            ))}
          </p>
        )}
      </div>
      <label className="sentient-topbar-lang">
        <span className="sentient-topbar-lang-label">{languageLabel}</span>
        <div className="sentient-lang-select-wrap">
          <select
            className="sentient-lang-select"
            value={value}
            aria-label={languageLabel}
            onChange={(event) => onChange(event.target.value)}
          >
            {options.map((item) => (
              <option key={item.code} value={item.code}>
                {item.label ?? item.english_name ?? item.code}
              </option>
            ))}
          </select>
          <span aria-hidden className="sentient-lang-select-caret">▾</span>
        </div>
      </label>
    </div>
  );
}

export function SentientLanguagePicker({
  kicker,
  description,
  options,
  value,
  onChange,
}: {
  kicker: string;
  description: string;
  options: InteractionLanguageOption[];
  value: InteractionLanguage;
  onChange: (language: InteractionLanguage) => void;
}) {
  return (
    <div className="sentient-lang-hero">
      <div className="sentient-lang-hero-text">
        <span className="sentient-lang-hero-kicker">{kicker}</span>
        <p className="sentient-lang-hero-desc">{description}</p>
      </div>
      <div className="sentient-lang-pills" role="radiogroup" aria-label={kicker}>
        {options.map((item) => (
          <button
            key={item.code}
            type="button"
            role="radio"
            aria-checked={value === item.code}
            className={`sentient-lang-pill${value === item.code ? " is-selected" : ""}`}
            onClick={() => onChange(item.code)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function SentientField({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: ReactNode;
}) {
  return (
    <label className="sentient-field">
      <span className="sentient-field-label">
        {label}
        {required && " *"}
      </span>
      {children}
    </label>
  );
}

export function SentientInput({
  value,
  onChange,
  placeholder,
  type = "text",
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
}) {
  return (
    <input
      className="sentient-input"
      type={type}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

export function SentientChoice({
  letter,
  label,
  hint,
  selected,
  onClick,
}: {
  letter: string;
  label: string;
  hint?: string;
  selected?: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className={`sentient-choice${selected ? " selected" : ""}`} onClick={onClick}>
      <span className="sentient-choice-letter">{letter}</span>
      <span className="sentient-choice-text">
        <strong>{label}</strong>
        {hint && <small>{hint}</small>}
      </span>
    </button>
  );
}

export function SentientOk({
  onClick,
  disabled,
  children = "OK",
}: {
  onClick?: () => void;
  disabled?: boolean;
  children?: string;
}) {
  return (
    <button type="button" className="sentient-ok" onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export function SentientFloatNav({
  onUp,
  onDown,
  canUp,
  canDown,
}: {
  onUp?: () => void;
  onDown?: () => void;
  canUp?: boolean;
  canDown?: boolean;
}) {
  return (
    <div className="sentient-float-nav" aria-label="Step navigation">
      <button type="button" onClick={onUp} disabled={!canUp} aria-label="Previous">
        ↑
      </button>
      <button type="button" onClick={onDown} disabled={!canDown} aria-label="Next">
        ↓
      </button>
    </div>
  );
}

export function SentientMeta({
  items,
  variant = "inline",
}: {
  items: Array<{ label: string; value: string }>;
  variant?: "inline" | "chips";
}) {
  if (variant === "chips") {
    return (
      <div className="sentient-meta sentient-meta-chips">
        {items.map((item) => (
          <div key={item.label} className="sentient-meta-chip">
            <strong>{item.value}</strong>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className="sentient-meta">
      {items.map((item) => (
        <span key={item.label}>
          <strong>{item.value}</strong> {item.label}
        </span>
      ))}
    </div>
  );
}

export function SentientToolbar({ children }: { children: ReactNode }) {
  return <div className="sentient-toolbar">{children}</div>;
}

export function SentientWorkspace({ children }: { children: ReactNode }) {
  return <div className="sentient-workspace">{children}</div>;
}

export function SentientGlass({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`sentient-glass ${className}`.trim()}>{children}</div>;
}

export function SentientKicker({ children }: { children: ReactNode }) {
  return <div className="sentient-kicker">{children}</div>;
}

export function SentientBtn({
  children,
  onClick,
  disabled,
  variant = "primary",
  size,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  size?: "sm";
}) {
  const classes = [
    variant === "ghost" ? "sentient-btn-ghost" : "sentient-btn",
    size === "sm" ? "sentient-btn-sm" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button type="button" className={classes} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export function SentientTextarea({
  value,
  onChange,
  rows = 3,
}: {
  value: string;
  onChange: (value: string) => void;
  rows?: number;
}) {
  return (
    <textarea
      className="sentient-textarea"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      rows={rows}
    />
  );
}

export function SentientStat({ label, value, warn }: { label: string; value: ReactNode; warn?: boolean }) {
  return (
    <div className={`sentient-stat${warn ? " sentient-stat-warn" : ""}`}>
      <div className="sentient-stat-value">{value}</div>
      <div className="sentient-stat-label">{label}</div>
    </div>
  );
}

export function SentientAlert({ children }: { children: ReactNode }) {
  return <div className="sentient-alert">{children}</div>;
}

export function SentientMuted({ children }: { children: ReactNode }) {
  return <p className="sentient-muted-text">{children}</p>;
}
