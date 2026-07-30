import genieMark from "../assets/genie-mark.png";
import "../styles/voice-orb.css";

/**
 * Shared voice orb (framework primitive) — the SINGLE Genie icon used by every
 * voice surface (home, billing, card, healthcare). It carries the Genie brand
 * "voice feel": a purple gradient core with the Genie ◎ mark, pulsing halo rings,
 * and a speaking/listening glow that breathes with the live audio level. Every
 * page renders this so the on-screen presence is identical and stays true to the
 * Genie icon's theme.
 *
 * `size` is any CSS length; the whole orb (rings, core, glyph) scales from it, so
 * a page can drop in a big hero orb or a compact rail orb with one prop.
 */
export type VoiceOrbState =
  | "idle"
  | "greeting"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking";

export function VoiceOrb({
  state,
  level = 0,
  size = "112px",
  onClick,
  disabled,
  ariaLabel,
}: {
  state: VoiceOrbState;
  level?: number;
  size?: string;
  onClick?: () => void;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const scale =
    state === "speaking"
      ? 1 + Math.min(0.28, level * 0.34)
      : state === "listening"
        ? 1 + Math.min(0.18, level * 0.22)
        : 1;
  const coreStyle = { transform: `scale(${scale})` };
  const mark = <img className="gv-orb-mark" src={genieMark} alt="" draggable={false} />;

  return (
    <div className={`gv-orb gv-orb-${state}`} style={{ width: size, height: size, fontSize: size }}>
      {onClick ? (
        <button
          type="button"
          className="gv-orb-core"
          style={coreStyle}
          onClick={onClick}
          disabled={disabled}
          aria-label={ariaLabel}
        >
          {mark}
        </button>
      ) : (
        <div className="gv-orb-core" style={coreStyle} aria-label={ariaLabel} role="img">
          {mark}
        </div>
      )}
    </div>
  );
}
