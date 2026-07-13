import type { ReactNode } from "react";
import { SentientFloatNav, SentientOk, SentientStep } from "./Sentient";

export function SentientFlow({
  step,
  title,
  description,
  children,
  onOk,
  okLabel = "OK",
  okDisabled,
  onUp,
  onDown,
  canUp = true,
  canDown = false,
  wide,
}: {
  step: number;
  title: string;
  description?: string;
  children?: ReactNode;
  onOk?: () => void;
  okLabel?: string;
  okDisabled?: boolean;
  onUp?: () => void;
  onDown?: () => void;
  canUp?: boolean;
  canDown?: boolean;
  wide?: boolean;
}) {
  return (
    <div className={`sentient-flow${wide ? " sentient-flow-wide" : ""}`}>
      <SentientStep step={step} title={title} description={description}>
        {children}
      </SentientStep>
      {onOk && (
        <SentientOk onClick={onOk} disabled={okDisabled}>
          {okLabel}
        </SentientOk>
      )}
      {(onUp || onDown) && (
        <SentientFloatNav canUp={canUp} canDown={canDown} onUp={onUp} onDown={onDown} />
      )}
    </div>
  );
}
