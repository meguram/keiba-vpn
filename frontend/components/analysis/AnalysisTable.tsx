"use client";

import { useState } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import { AnalysisEntry, AnalysisModuleMeta, ColumnBlock, columnBlocks, formatCell, heatColor, isThin, runColor, shortLabel, sortValue, PastRun } from "@/lib/raceAnalysis";

const CSS = `
.ra-wide{display:block}.ra-cards{display:none}
@media (max-width:768px){.ra-wide{display:none}.ra-cards{display:block}}
.ra-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.ra-tbl{width:100%;border-collapse:collapse;font-size:12px}
.ra-tbl th,.ra-tbl td{padding:7px 4px;text-align:center;white-space:nowrap;vertical-align:middle}
.ra-tbl td{border-bottom:1px solid rgba(36,48,73,.4)}
.ra-tbl thead th{color:var(--text-dim);font-weight:600;border-bottom:1px solid var(--border)}
.ra-tbl thead tr.ra-g th{border-bottom:none;padding-bottom:2px}
.ra-tbl thead tr.ra-g th.gh{border-left:1px solid rgba(36,48,73,.6);color:var(--text);font-weight:700}
.ra-tbl thead tr.ra-g th.gh small{display:block;font-size:10px;font-weight:400;color:var(--text-dim)}
.ra-tbl .l{text-align:left}
.ra-tbl td.w{white-space:normal;min-width:6em;max-width:11em;overflow-wrap:anywhere;line-height:1.3}
.ra-tbl .gs{border-left:1px solid rgba(36,48,73,.6)}
.ra-tbl .thin{opacity:.45}
.ra-tbl .nm{font-weight:700;color:#fff}
.ra-tbl .dim{color:var(--text-dim)}
.ra-mbar{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:var(--text-dim)}
.ra-mbar select{flex:1 1 auto;min-width:0;min-height:36px;padding:0 8px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:14px}
.ra-m{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;table-layout:auto}
.ra-m th,.ra-m td{padding:5px 2px;text-align:center;vertical-align:middle;white-space:nowrap}
.ra-m td{border-bottom:1px solid rgba(36,48,73,.4)}
.ra-m thead th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border);padding-top:7px;padding-bottom:7px}
.ra-m .l{text-align:left}
.ra-m td.mn{width:100%;max-width:0;text-align:left;padding-left:4px}
.ra-m td.mn b{display:block;font-size:13px;color:#fff;overflow:hidden;text-overflow:ellipsis}
.ra-m td.mn small{display:block;font-size:10.5px;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;margin-top:1px}
.ra-m td.mn .ra-runs{margin-top:3px}
.ra-m .thin{opacity:.45}
.ra-m .ra-run{min-width:21px;padding:1px 2px}
.ra-m .ra-run b{font-size:12px}
.ra-m .ra-run small{font-size:8px}
.ra-runs{display:inline-flex;gap:3px;vertical-align:middle}
.ra-run{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;min-width:26px;padding:2px 3px;border-radius:5px;color:#fff;line-height:1.05;font-variant-numeric:tabular-nums;border:1px solid transparent}
.ra-run b{font-size:14px;font-weight:800}
.ra-run small{font-size:9px;opacity:.8}
.ra-run.x{background:transparent!important;border:1px dashed rgba(139,148,158,.7);color:var(--text-dim);opacity:.7}
.ra-run.x b{text-decoration:line-through;font-weight:600}
`;

function RunChips({ v }: { v: unknown }) {
  if (!Array.isArray(v) || v.length === 0) return <span className="dim">–</span>;
  return (
    <span className="ra-runs">
      {(v as PastRun[]).map(([p, x, ex], i) => (
        <span key={i} className={`ra-run${ex ? " x" : ""}`} style={{ background: ex ? undefined : runColor(p, x) }} title={`${p}番手 / ${x}頭${ex ? "（外れ値として平均から除外）" : ""}`}>
          <b>{p}</b><small>/{x}</small>
        </span>
      ))}
    </span>
  );
}

function NumChip({ e, size = 24 }: { e: AnalysisEntry; size?: number }) {
  return (
    <span style={{ ...bracketChipStyle(e.bracket_number, size), fontSize: 12 }} title={e.bracket_number ? `${e.bracket_number}枠` : undefined}>
      {e.horse_number ?? "–"}
    </span>
  );
}

/** スマホ用の表。群は 1 つずつプルダウンで選び、数値の列だけを並べて見出しで並べ替えられるようにする（並べ替えは TableSorter）。
    成績（1着-2着-3着-着外）は走数に、文字の列（系統名など）と直近の走は馬名の下に寄せて、横幅に収める。 */
function MobileTable({ entries, module, blocks, keyHeader }: { entries: AnalysisEntry[]; module: AnalysisModuleMeta; blocks: ColumnBlock[]; keyHeader: string | null }) {
  const [sel, setSel] = useState(0);
  const b = blocks[Math.min(sel, blocks.length - 1)];
  if (!b) return null;
  const sub = b.columns.filter((c) => (c.fmt === "text" && !c.palette) || c.fmt === "runs");
  const cols = b.columns.filter((c) => !sub.includes(c));
  const nKey = b.group?.n_key ?? null;
  return (
    <>
      {blocks.length > 1 && (
        <label className="ra-mbar">
          表示
          <select value={Math.min(sel, blocks.length - 1)} onChange={(ev) => setSel(Number(ev.target.value))} aria-label="表示する集計">
            {blocks.map((x, i) => (
              <option key={x.group?.key ?? `u${i}`} value={i}>{x.group ? `${x.group.label}${x.group.sub ? `（${x.group.sub}）` : ""}` : "その他"}</option>
            ))}
          </select>
        </label>
      )}
      <table className="ra-m">
        <thead>
          <tr>
            <th>番</th>
            <th className="l">馬名{keyHeader && !sub.some((c) => c.fmt === "text") ? ` / ${keyHeader}` : ""}</th>
            {cols.map((c) => <th key={c.key} title={c.title ?? undefined}>{c.fmt === "record" && nKey ? "走数" : shortLabel(c.label)}</th>)}
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const cells = e.analysis?.[module.id];
            const thin = isThin(b.group, cells);
            const texts = sub.filter((c) => c.fmt === "text").map((c) => cells?.[c.key]).filter((v) => v != null && v !== "");
            const line = texts.length ? texts.join(" / ") : keyHeader ? ((cells?._key as string | null | undefined) ?? "不明") : null;
            return (
              <tr key={e.key}>
                <td><NumChip e={e} size={22} /></td>
                <td className="mn">
                  <b>{e.horse_name ?? "–"}</b>
                  {line && <small>{line}</small>}
                  {sub.filter((c) => c.fmt === "runs").map((c) => <RunChips key={c.key} v={cells?.[c.key]} />)}
                </td>
                {cols.map((c) => {
                  const v = cells?.[c.key];
                  if (c.fmt === "record" && nKey) {
                    const n = cells?.[nKey];
                    return <td key={c.key} className={thin ? "thin" : undefined} title={formatCell(v, c.fmt)}>{typeof n === "number" ? n : "–"}</td>;
                  }
                  const bg = thin ? undefined : (heatColor(v, c.scale) ?? (c.palette && typeof v === "string" ? c.palette[v] : undefined));
                  return (
                    <td key={c.key} className={thin ? "thin" : undefined} style={{ background: bg, fontWeight: c.palette && bg ? 700 : undefined }} data-sort={sortValue(v, c.fmt, c.palette)}>
                      {formatCell(v, c.fmt)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

export function AnalysisTable({ entries, module }: { entries: AnalysisEntry[]; module: AnalysisModuleMeta }) {
  const columns = module.columns ?? [];
  const blocks = columnBlocks(columns, module.groups ?? []);
  const ordered = blocks.flatMap((b) => b.columns.map((c) => ({ c, g: b.group, first: c === b.columns[0] })));
  const cellsOf = (e: AnalysisEntry) => e.analysis?.[module.id];
  const keyHeader = module.key_header ?? null;
  const keyOf = (e: AnalysisEntry) => (cellsOf(e)?._key as string | null | undefined) ?? null;

  return (
    <div style={{ marginTop: 8 }}>
      <style>{CSS}</style>

      <div className="ra-wide ra-scroll">
        <table className="ra-tbl">
          <thead>
            <tr className="ra-g">
              <th colSpan={keyHeader ? 3 : 2} />
              {blocks.map((b, i) => (
                <th key={b.group?.key ?? `u${i}`} className="gh" colSpan={b.columns.length}>
                  {b.group ? <>{b.group.label}{b.group.sub && <small>{b.group.sub}</small>}</> : null}
                </th>
              ))}
            </tr>
            <tr>
              <th>馬番</th><th className="l">馬名</th>{keyHeader && <th className="l">{keyHeader}</th>}
              {ordered.map(({ c, first }) => (
                <th key={c.key} className={first ? "gs" : undefined} title={c.title ?? undefined}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const cells = cellsOf(e);
              return (
                <tr key={e.key}>
                  <td><NumChip e={e} /></td>
                  <td className="l nm">{e.horse_name ?? "–"}</td>
                  {keyHeader && <td className="l w">{keyOf(e) ?? <span className="dim">不明</span>}</td>}
                  {ordered.map(({ c, g, first }) => {
                    const v = cells?.[c.key];
                    const thin = isThin(g, cells);
                    const bg = thin ? undefined : (heatColor(v, c.scale) ?? (c.palette && typeof v === "string" ? c.palette[v] : undefined));
                    const left = (c.fmt === "text" && !c.palette) || c.fmt === "runs";
                    return (
                      <td
                        key={c.key}
                        className={[first ? "gs" : "", thin ? "thin" : "", left ? "l" : "", c.fmt === "text" && !c.palette ? "w" : ""].filter(Boolean).join(" ") || undefined}
                        style={{ background: bg, fontWeight: c.palette && bg ? 700 : undefined }}
                        data-sort={sortValue(v, c.fmt, c.palette)}
                      >
                        {c.fmt === "runs" ? <RunChips v={v} /> : formatCell(v, c.fmt)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="ra-cards">
        <MobileTable entries={entries} module={module} blocks={blocks} keyHeader={keyHeader} />
      </div>
    </div>
  );
}
