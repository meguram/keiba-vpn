"use client";

import { bracketChipStyle } from "@/lib/bracket";
import { AnalysisEntry, AnalysisModuleMeta, columnBlocks, formatCell, heatColor, isThin, sortValue } from "@/lib/raceAnalysis";

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
`;

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

  return (
    <div style={{ marginTop: 8 }}>
      <style>{CSS}</style>

      <div className="ra-wide ra-scroll">
        <table className="ra-tbl">
          <thead>
            <tr className="ra-g">
              <th colSpan={4} />
              {blocks.map((b, i) => (
                <th key={b.group?.key ?? `u${i}`} className="gh" colSpan={b.columns.length}>
                  {b.group ? <>{b.group.label}{b.group.sub && <small>{b.group.sub}</small>}</> : null}
                </th>
              ))}
            </tr>
            <tr>
              <th>馬番</th><th className="l">馬名</th><th>性齢</th><th className="l">騎手</th>
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
                  {ordered.map(({ c, g, first }) => {
                    const v = cells?.[c.key];
                    const thin = isThin(g, cells);
                    return (
                      <td
                        key={c.key}
                        className={[first ? "gs" : "", thin ? "thin" : ""].filter(Boolean).join(" ") || undefined}
                        style={{ background: thin ? undefined : heatColor(v, c.scale) }}
                        data-sort={sortValue(v, c.fmt)}
                      >
                        {formatCell(v, c.fmt)}
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
              <div className="r2">{[e.sex_age, e.jockey_name ?? "騎手未定", e.jockey_weight != null ? `${e.jockey_weight}kg` : null, e.trainer_name].filter(Boolean).join(" ・ ")}</div>
              {blocks.map((b, i) => (
                <div key={b.group?.key ?? `u${i}`} className={`grp${isThin(b.group, cells) ? " thin" : ""}`}>
                  <div className="gl">{b.group ? <><b>{b.group.label}</b>{b.group.sub}</> : null}</div>
                  <div className="gv">
                    {b.columns.map((c) => {
                      const bg = isThin(b.group, cells) ? undefined : heatColor(cells?.[c.key], c.scale);
                      return <span key={c.key} className={bg ? "hm" : undefined} style={{ background: bg }}><i>{c.label}</i>{formatCell(cells?.[c.key], c.fmt)}</span>;
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
