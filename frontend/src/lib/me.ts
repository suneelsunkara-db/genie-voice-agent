import { useEffect, useState } from "react";

import { API_BASE_URL } from "../config";

/**
 * The signed-in user, from the backend `/me` endpoint (Databricks Apps forwards
 * the identity via `X-Forwarded-*` headers; a local dev fallback uses
 * `APP_DEV_USER`). Every voice surface greets and converses with THIS person, so
 * the spoken name is the real logged-in user rather than a demo persona.
 */
export interface Me {
  email: string;
  username: string;
  /** Best-effort human first name ("" when unknown → greet generically). */
  name: string;
  authenticated: boolean;
}

const EMPTY_ME: Me = { email: "", username: "", name: "", authenticated: false };

// Module-level cache so the SPA fetches `/me` once and shares it across every
// page (home concierge, card, billing, knowledge) for the whole session.
let mePromise: Promise<Me> | null = null;

export function getMe(): Promise<Me> {
  if (!mePromise) {
    mePromise = (async (): Promise<Me> => {
      try {
        const r = await fetch(`${API_BASE_URL}/me`);
        const d = (await r.json()) as Partial<Me>;
        return {
          email: typeof d.email === "string" ? d.email : "",
          username: typeof d.username === "string" ? d.username : "",
          name: typeof d.name === "string" ? d.name : "",
          authenticated: Boolean(d.authenticated),
        };
      } catch {
        return EMPTY_ME;
      }
    })();
  }
  return mePromise;
}

/** React hook: the signed-in user (resolves once, then stays cached). */
export function useMe(): Me {
  const [me, setMe] = useState<Me>(EMPTY_ME);
  useEffect(() => {
    let active = true;
    void getMe().then((m) => {
      if (active) setMe(m);
    });
    return () => {
      active = false;
    };
  }, []);
  return me;
}
