"use client";

import { picksMessage, type PPFigure } from "@/lib/pedigreePersona";

const CSS = `
.pp-intro{font-size:12.5px;line-height:1.7;margin:8px 0 4px}
.pp-note{font-size:11.5px;color:#fbbf24;margin:4px 0 8px}
.pp-facts{display:inline-block;font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 10px;margin:4px 0;color:var(--text-dim)}
.pp-list{list-style:none;padding:0;margin:8px 0 0}
.pp-list li{border:1px solid var(--border);border-radius:8px;padding:8px;margin:6px 0}
.pp-num{display:inline-block;min-width:26px;text-align:center;font-weight:700;margin-right:6px}
.pp-name{font-weight:600;font-size:13px}
.pp-reason{font-size:12px;line-height:1.6;margin:4px 0 0;color:var(--text)}
.pp-dim{font-size:12px;color:var(--text-dim);margin:8px 0 0}
.pp-sub{font-size:12px;font-weight:600;margin:12px 0 0}
`;

/** 血統予想家の欄。印は出さない。並べるのは理由の文と事実だけで、点・人気・成績の数字は出さない。 */
export function PedigreePersona({ fig, notes }: { fig: PPFigure; notes?: string[] }) {
  const msg = picksMessage(fig.picks_state);
  return (
    <div>
      <style>{CSS}</style>
      <p className="pp-intro">{fig.intro}</p>
      <p className="pp-note">{fig.note}</p>
      <span className="pp-facts">このレース: {fig.facts}</span>
      <p className="pp-sub">
        血統の上位の人気薄（{fig.surface}）{fig.confidence ? `・自信は${fig.confidence}` : ""}
      </p>
      {fig.picks_state === "ready" && fig.picks.length > 0 ? (
        <ul className="pp-list">
          {fig.picks.map((p) => (
            <li key={p.horse_number}>
              <span className="pp-num">{p.horse_number}</span>
              <span className="pp-name">{p.horse_name}</span>
              <p className="pp-reason">{p.reason}</p>
            </li>
          ))}
        </ul>
      ) : (
        msg && <p className="pp-dim">{msg}</p>
      )}
      {notes?.length ? (
        <ul style={{ margin: "10px 0 0", paddingLeft: 18, fontSize: 11, color: "var(--text-dim)" }}>
          {notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
