"use client";

import { bracketChipStyle } from "@/lib/bracket";
import { AnalysisEntry, AnalysisModuleMeta, columnBlocks, formatCell, heatColor, isThin, runColor, sortValue, PastRun } from "@/lib/raceAnalysis";

const CSS = `
.ra-wide{display:block}.ra-cards{display:none}
@media (max-width:768px){.ra-wide{display:none}.ra-cards{display:block}}
.ra-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.ra-tbl{width:100%;border-collapse:collapse;font-size:12px;min-width:max-content}
.ra-tbl th,.ra-tbl td{padding:7px 4px;text-align:center;white-space:nowrap;vertical-align:middle}
.ra-tbl td{border-bottom:1px solid rgba(36,48,73,.4)}
.ra-tbl thead th{color:var(--text-dim);font-weight:600;border-bottom:1px solid var(--border)}
.ra-tbl thead tr.ra-g th{border-bottom:none;padding-bottom:2px}
.ra-tbl thead tr.ra-g th.gh{border-left:1px solid rgba(36,48,73,.6);color:var(--text);font-weight:700}
.ra-tbl thead tr.ra-g th.gh small{display:block;font-size:10px;font-weight:400;color:var(--text-dim)}
.ra-tbl .l{text-align:left}
.ra-tbl .gs{border-left:1px solid rgba(36,48,73,.6)}
.ra-tbl .thin{opacity:.45}
.ra-tbl .nm{font-weight:700;color:#fff}
.ra-tbl .dim{color:var(--text-dim)}
.ra-card{padding:10px 12px;border-bottom:1px solid rgba(36,48,73,.5);display:grid;gap:6px}
.ra-card .r1{display:flex;align-items:center;gap:8px;min-width:0}
.ra-card .nm{font-weight:700;color:#fff;font-size:15px;flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ra-card .r2{font-size:12px;color:var(--text-dim)}
.ra-card .grp{display:grid;gap:2px;font-size:12px}
.ra-card .grp .gl{color:var(--text-dim);font-size:11px}
.ra-card .grp .gl b{color:var(--text);font-weight:600;margin-right:6px}
.ra-card .grp .gv{display:flex;flex-wrap:wrap;gap:2px 14px;font-variant-numeric:tabular-nums}
.ra-card .grp .gv span i{font-style:normal;color:var(--text-dim);font-size:10px;margin-right:3px}
.ra-card .thin{opacity:.45}
.ra-card .hm{padding:1px 6px;border-radius:5px}
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

function NumChip({ e }: { e: AnalysisEntry }) {
  return (
    <span style={{ ...bracketChipStyle(e.bracket_number, 24), fontSize: 12 }} title={e.bracket_number ? `${e.bracket_number}枠` : undefined}>
      {e.horse_number ?? "–"}
    </span>
  );
}

function Jockey({ e }: { e: AnalysisEntry }) {
  return e.jockey_name ? <>{e.jockey_name}</> : <span className="dim">未定</span>;
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
              <th colSpan={keyHeader ? 5 : 4} />
              {blocks.map((b, i) => (
                <th key={b.group?.key ?? `u${i}`} className="gh" colSpan={b.columns.length}>
                  {b.group ? <>{b.group.label}{b.group.sub && <small>{b.group.sub}</small>}</> : null}
                </th>
              ))}
            </tr>
            <tr>
              <th>馬番</th><th className="l">馬名</th><th>性齢</th><th className="l">騎手</th>{keyHeader && <th className="l">{keyHeader}</th>}
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
                  <td className="dim">{e.sex_age ?? "–"}</td>
                  <td className="l"><Jockey e={e} /></td>
                  {keyHeader && <td className="l">{keyOf(e) ?? <span className="dim">不明</span>}</td>}
                  {ordered.map(({ c, g, first }) => {
                    const v = cells?.[c.key];
                    const thin = isThin(g, cells);
                    const bg = thin ? undefined : (heatColor(v, c.scale) ?? (c.palette && typeof v === "string" ? c.palette[v] : undefined));
                    const left = (c.fmt === "text" && !c.palette) || c.fmt === "runs";
                    return (
                      <td
                        key={c.key}
                        className={[first ? "gs" : "", thin ? "thin" : "", left ? "l" : ""].filter(Boolean).join(" ") || undefined}
                        style={{ background: bg, fontWeight: c.palette && bg ? 700 : undefined }}
                        data-sort={sortValue(v, c.fmt)}
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
        {entries.map((e) => {
          const cells = cellsOf(e);
          return (
            <div key={e.key} className="ra-card">
              <div className="r1"><NumChip e={e} /><span className="nm">{e.horse_name ?? "–"}</span></div>
              <div className="r2">{[e.sex_age, e.jockey_name ?? "騎手未定", e.jockey_weight != null ? `${e.jockey_weight}kg` : null, e.trainer_name, keyHeader ? `${keyHeader}: ${keyOf(e) ?? "不明"}` : null].filter(Boolean).join(" ・ ")}</div>
              {blocks.map((b, i) => (
                <div key={b.group?.key ?? `u${i}`} className={`grp${isThin(b.group, cells) ? " thin" : ""}`}>
                  <div className="gl">{b.group ? <><b>{b.group.label}</b>{b.group.sub}</> : null}</div>
                  <div className="gv">
                    {b.columns.map((c) => {
                      const v = cells?.[c.key];
                      const bg = isThin(b.group, cells) ? undefined : (heatColor(v, c.scale) ?? (c.palette && typeof v === "string" ? c.palette[v] : undefined));
                      return <span key={c.key} className={bg ? "hm" : undefined} style={{ background: bg }}><i>{c.label}</i>{c.fmt === "runs" ? <RunChips v={v} /> : formatCell(v, c.fmt)}</span>;
                    })}
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
