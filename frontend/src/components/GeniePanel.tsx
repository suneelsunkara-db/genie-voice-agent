import { useState } from "react";
import { api, GenieResponse } from "../api/client";

export function GeniePanel() {
  const [q, setQ] = useState("How many billing_dispute calls were there?");
  const [resp, setResp] = useState<GenieResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    setLoading(true);
    try {
      setResp(await api.askGenie(q));
    } finally {
      setLoading(false);
    }
  };

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
          <div className="answer">{resp.answer}</div>
          {resp.sql && <pre className="sql">{resp.sql}</pre>}
          {resp.rows && (
            <table>
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
        </div>
      )}
    </div>
  );
}
