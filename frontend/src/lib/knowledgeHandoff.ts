const MAX_AGE_MS = 5 * 60 * 1000;

export interface KnowledgeHandoff {
  query: string;
  language: string;
  createdAt: number;
}

let pendingHandoff: KnowledgeHandoff | null = null;

/** Hold one in-memory continuation while this SPA changes assistant surfaces. */
export function storeKnowledgeHandoff(query: string, language: string): void {
  const clean = query.trim();
  if (!clean) return;
  pendingHandoff = {
    query: clean,
    language: language.trim(),
    createdAt: Date.now(),
  };
}

function readKnowledgeHandoff(remove: boolean): KnowledgeHandoff | null {
  const handoff = pendingHandoff;
  if (remove) pendingHandoff = null;
  if (!handoff) return null;
  if (Date.now() - handoff.createdAt > MAX_AGE_MS) {
    pendingHandoff = null;
    return null;
  }
  return { ...handoff };
}

/** Read without consuming; safe during React StrictMode's development remount. */
export function peekKnowledgeHandoff(): KnowledgeHandoff | null {
  return readKnowledgeHandoff(false);
}

/** Consume exactly once so refresh/reconnect cannot submit a duplicate question. */
export function consumeKnowledgeHandoff(): KnowledgeHandoff | null {
  return readKnowledgeHandoff(true);
}
