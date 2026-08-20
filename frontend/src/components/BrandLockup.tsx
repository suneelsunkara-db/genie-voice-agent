import databricksLogo from "../assets/databricks-logo.png";
import genieLogo from "../assets/genie-logo.png";
import "../styles/brand-lockup.css";

/**
 * Shared Databricks + Genie brand lockup (framework primitive).
 *
 * The official logos have a dark wordmark, so they sit on a clean white pill that
 * reads correctly on ANY page background (home, card, knowledge). Every surface
 * uses this same component so the branding is identical everywhere. (The billing
 * cockpit has its own equivalent via the Sentient shell and is intentionally left
 * as-is.)
 */
export function BrandLockup({ product }: { product?: string }) {
  return (
    <div className="brand-lockup">
      <span className="brand-lockup-pill">
        <img src={databricksLogo} alt="Databricks" className="brand-lockup-logo dbx" />
        <span className="brand-lockup-div" />
        <img src={genieLogo} alt="Genie" className="brand-lockup-logo genie" />
      </span>
      {product && <span className="brand-lockup-product">{product}</span>}
    </div>
  );
}
