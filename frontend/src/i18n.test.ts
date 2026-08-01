import { describe, expect, it, vi } from "vitest";

import { ensureLocale, localizeResolutionNote, localizedValue } from "./i18n";
// The exact bytes translate_locales.py hashes, so the staleness check below can be
// computed here without reading the file through a Node-only API.
import EN_RAW from "./locales/en.json?raw";

const BUNDLES = import.meta.glob<Record<string, string>>("./locales/*.json", {
  eager: true,
  import: "default",
});

/** Generated bundles only — en.json is the source the translator reads. */
const GENERATED = Object.entries(BUNDLES).filter(([path]) => !path.endsWith("/en.json"));

/** Mirrors translate_locales.py: sha256 of en.json, first 16 hex chars. */
async function sourceHash(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
}

describe("localizedValue", () => {
  it("renders canonical status codes as English words", () => {
    expect(localizedValue("en-US", "overdue")).toBe("overdue");
    expect(localizedValue("en-US", "at_risk")).toBe("at risk");
    expect(localizedValue("en-US", "language_mismatch", "reason")).toBe("language mismatch");
    expect(localizedValue("en-US", "billing_dispute", "intent")).toBe("billing dispute");
  });

  it("falls back to de-underscored words for codes with no label", () => {
    expect(localizedValue("en-US", "some_unmapped_code")).toBe("some unmapped code");
    expect(localizedValue("en-US", null)).toBe("—");
  });

  it("localizes status values for a MACHINE-TRANSLATED locale, not just authored ones", async () => {
    // The regression this guards: value labels lived in a hand-written table with
    // only en/th/id/zh, so a Hindi call rendered "overdue"/"paid"/"closed" in
    // English inside an otherwise Hindi UI.
    ensureLocale("hi-IN");
    // Wait on the TRANSLATED text, not on truthiness: uiCopy falls back to English,
    // so any key is already truthy before the bundle lands.
    await vi.waitFor(() =>
      expect(localizedValue("hi-IN", "overdue")).toMatch(/\p{Script=Devanagari}/u)
    );
    for (const code of ["overdue", "paid", "closed"]) {
      const label = localizedValue("hi-IN", code);
      expect(label).not.toBe(code);
      expect(label).toMatch(/\p{Script=Devanagari}/u);
    }
  });
});

describe("localizeResolutionNote", () => {
  it("renders the backend's canonical note codes", () => {
    expect(localizeResolutionNote("en-US", "issue_closed_arrangement_waiver")).toContain(
      "payment arrangement confirmed"
    );
    expect(localizeResolutionNote("en-US", "issue_guided_by_genie", "in_progress")).toBe(
      "Issue in progress: guided by Genie and account context."
    );
  });

  it("passes through prose written before the codes existed", () => {
    const legacy = "Issue closed: legacy free text.";
    expect(localizeResolutionNote("en-US", legacy)).toBe(legacy);
    expect(localizeResolutionNote("en-US", null)).toBe("");
  });

  it("translates notes for a machine-translated locale", async () => {
    ensureLocale("hi-IN");
    await vi.waitFor(() =>
      expect(localizeResolutionNote("hi-IN", "issue_closed_arrangement_waiver")).toMatch(
        /\p{Script=Devanagari}/u
      )
    );
    // The {status} placeholder must survive translation and be filled in-language.
    expect(localizeResolutionNote("hi-IN", "issue_guided_by_genie", "in_progress")).not.toContain(
      "{status}"
    );
  });
});

describe("locale bundles", () => {
  it("covers every supported language", () => {
    expect(GENERATED.length).toBeGreaterThan(15);
  });

  it("carries every key of the English catalog", () => {
    const englishKeys = Object.keys(BUNDLES["./locales/en.json"] ?? {});
    expect(englishKeys.length).toBeGreaterThan(0);
    for (const [path, bundle] of GENERATED) {
      const missing = englishKeys.filter((key) => !(key in bundle));
      expect(missing, `${path} is missing keys`).toEqual([]);
    }
  });

  it("is regenerated whenever the English catalog changes", async () => {
    // Each bundle records the hash of the en.json it was translated from. Nothing
    // else checks it, so adding a key and forgetting to re-run
    // scripts/i18n/translate_locales.py would ship silent English fallbacks.
    const expected = await sourceHash(EN_RAW);
    for (const [path, bundle] of GENERATED) {
      expect(bundle._sourceHash, `${path} is stale — re-run translate_locales.py`).toBe(expected);
    }
  });
});
