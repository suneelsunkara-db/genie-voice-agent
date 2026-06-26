import { useState } from "react";
import { api, GenieResponse } from "../api/client";

export function GeniePanel() {
  const [q, setQ] = useState("How many billing_dispute calls were there?");
  const [resp, setResp] = useState<GenieResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [showSql, setShowSql] = useState(false);

  // Keep the conversation open within this panel session so follow-ups retain
  // context. Reset when the user edits the box and asks a fresh question.
  const run = async (question: string, conversationId?: string) => {
    setLoading(true);
    try {
      setResp(await api.askGenie(question, conversationId));
    } finally {
      setLoading(false);
    }
  };

  const ask = () => run(q);
  const askFollowup = (question: string) => run(question, resp?.conversation_id);

  return (
    <div className="genie">
      <div className="genie-input">
        <input value={q} onChange={(e) => setQ(e.target.value)} />
        <button onClick={ask} disabled={loading}>
          {loading ? "Asking…" : "Ask Genie"}
        </button>
      </div>
      {resp && (
        <div className="genie-resp">
          {/* Lead with Genie's NL answer (real facts); never lead with SQL. */}
          {(resp.answer || resp.description) && (
            <div className="answer">{resp.answer ?? resp.description}</div>
          )}
          {resp.description && resp.answer && resp.description !== resp.answer && (
            <div className="genie-hint">{resp.description}</div>
          )}
          {resp.rows && (
            <table>
              {resp.columns && resp.columns.length > 0 && (
                <thead>
                  <tr>
                    {resp.columns.map((c, i) => (
                      <th key={i}>{c}</th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody>
                {resp.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j}>{String(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {resp.suggested_followups && resp.suggested_followups.length > 0 && (
            <div className="genie-followups">
              {resp.suggested_followups.map((f, i) => (
                <button
                  key={i}
                  className="followup-chip"
                  disabled={loading}
                  onClick={() => askFollowup(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          )}
          {resp.sql && (
            <div className="genie-sql">
              <button className="sql-toggle" onClick={() => setShowSql((v) => !v)}>
                {showSql ? "Hide query" : "Show query"}
              </button>
              {showSql && <pre className="sql">{resp.sql}</pre>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
