"use client";

import { useEffect, useMemo, useState } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import { AnalysisEntry, AnalysisModuleMeta, ColumnBlock, columnBlocks, formatCell, heatColor, isThin, runColor, shortLabel, sortValue, PastRun, RangeCell } from "@/lib/raceAnalysis";

/* th.rn: 通過順位の列はチップ 5 個分（.ra-run 1 個 = 最小 26px＋左右の余白 6px＋枠 2px = 34px、すき間 3px × 4）= 182px に固定し、見出しは折り返す */
const CSS = `
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
.ra-m td.mn .ra-runs{display:flex;flex-wrap:wrap;row-gap:2px;margin-top:3px}
.ra-m td.mn small .ra-st{display:inline-block;padding:0 4px;margin-right:4px;border-radius:4px;color:#fff;font-weight:700}
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
.ra-rw{display:inline-flex;flex-direction:column;align-items:center;gap:1px}
.ra-rk{font-size:9px;line-height:1;color:var(--text-dim);white-space:nowrap}
.ra-rw.p0 .ra-rk{color:var(--text);font-weight:800}
.ra-rw.p0 .ra-run{box-shadow:0 0 0 1.5px var(--text)}
.ra-m .ra-rk{font-size:8px}
.ra-rg{display:inline-flex;flex-direction:column;align-items:center;gap:2px}
.ra-rg svg{display:block}
.ra-rg span{font-size:11px;font-variant-numeric:tabular-nums}
.ra-tbl th.rg{white-space:normal;width:72px;min-width:72px;max-width:72px;line-height:1.25}
.ra-tbl th.rn{white-space:normal;width:182px;min-width:182px;line-height:1.25}
`;

/** 直近の位置取りの幅。左が先頭・右が最後方。点は各走（新しい走ほど濃い）、縦線は加重平均 */
function RangeBar({ v }: { v: unknown }) {
  const r = v as RangeCell | null;
  if (!r || typeof r.lo !== "number") return <span className="dim">–</span>;
  const W = 64, H = 14, x = (f: number) => 4 + f * (W - 8);
  return (
    <span className="ra-rg" title={`前から ${Math.round(r.lo * 100)}〜${Math.round(r.hi * 100)}%（平均 ${Math.round(r.mean * 100)}%）。新しい走ほど濃い点`}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="直近の位置取りの幅">
        <rect x={0} y={H / 2 - 1} width={W} height={2} rx={1} fill="rgba(139,148,158,.25)" />
        <rect x={x(r.lo)} y={H / 2 - 3} width={Math.max(2, x(r.hi) - x(r.lo))} height={6} rx={3} fill="rgba(96,165,250,.28)" />
        {r.pts.map(([f, w], i) => <circle key={i} cx={x(f)} cy={H / 2} r={3.2} fill={runColor(f * 100 + 1, 101)} opacity={0.2 + 0.8 * w} />)}
        <line x1={x(r.mean)} x2={x(r.mean)} y1={1} y2={H - 1} stroke="#fff" strokeWidth={1.5} />
      </svg>
      <span>{formatCell(v, "range")}</span>
    </span>
  );
}

/** 直近の走のチップ。左が前走で、各チップの上に「前走・2走前…」を出す（前走は枠で強調） */
const ago = (i: number) => (i === 0 ? "前走" : `${i + 1}走前`);
function RunChips({ v }: { v: unknown }) {
  if (!Array.isArray(v) || v.length === 0) return <span className="dim">–</span>;
  return (
    <span className="ra-runs" aria-label="左が前走">
      {(v as PastRun[]).map(([p, x, ex], i) => (
        <span key={i} className={`ra-rw${i === 0 ? " p0" : ""}`}>
          <span className="ra-rk">{ago(i)}</span>
          <span className={`ra-run${ex ? " x" : ""}`} style={{ background: ex ? undefined : runColor(p, x) }} title={`${ago(i)}: ${p}番手 / ${x}頭${ex ? "（外れ値として平均から除外）" : ""}`}>
            <b>{p}</b><small>/{x}</small>
          </span>
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
    成績（1着-2着-3着-着外）は走数に、文字の列（系統名・色分けのある脚質など）・直近の走・位置取りの幅は馬名の下に寄せて、横幅に収める（390px で横にはみ出さない）。
    直近の走のチップは馬名の列の幅で折り返す。 */
function MobileTable({ entries, module, blocks, keyHeader }: { entries: AnalysisEntry[]; module: AnalysisModuleMeta; blocks: ColumnBlock[]; keyHeader: string | null }) {
  const [sel, setSel] = useState(0);
  const b = blocks[Math.min(sel, blocks.length - 1)];
  if (!b) return null;
  const sub = b.columns.filter((c) => c.fmt === "text" || c.fmt === "runs" || c.fmt === "range");
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
            <th className="l">馬名{keyHeader && !sub.some((c) => c.fmt === "text" && !c.palette) ? ` / ${keyHeader}` : ""}</th>
            {cols.map((c) => <th key={c.key} title={c.title ?? undefined}>{c.fmt === "record" && nKey ? "走数" : shortLabel(c.label)}</th>)}
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const cells = e.analysis?.[module.id];
            const thin = isThin(b.group, cells);
            const texts = sub.filter((c) => c.fmt === "text" && !c.palette).map((c) => cells?.[c.key]).filter((v) => v != null && v !== "");
            const line = texts.length ? texts.join(" / ") : keyHeader ? ((cells?._key as string | null | undefined) ?? "不明") : null;
            return (
              <tr key={e.key}>
                <td><NumChip e={e} size={22} /></td>
                <td className="mn">
                  <b>{e.horse_name ?? "–"}</b>
                  {line && <small>{line}</small>}
                  {sub.filter((c) => c.fmt === "text" && c.palette).map((c) => {
                    const v = cells?.[c.key];
                    return typeof v === "string" && v ? <small key={c.key}><span className="ra-st" style={{ background: c.palette?.[v] ?? "rgba(139,148,158,.35)" }} title={c.label}>{v}</span></small> : null;
                  })}
                  {sub.filter((c) => c.fmt === "range" && typeof (cells?.[c.key] as RangeCell | null | undefined)?.lo === "number").map((c) => <small key={c.key} title={c.label}>前から{formatCell(cells?.[c.key], c.fmt)}</small>)}
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

/** 768px 以下ならスマホ表示。片方の DOM だけを作る（初回描画は PC 表示で、直後に合わせる） */
function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width:768px)");
    const update = () => setNarrow(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return narrow;
}

export function AnalysisTable({ entries, module }: { entries: AnalysisEntry[]; module: AnalysisModuleMeta }) {
  const narrow = useNarrow();
  const blocks = useMemo(() => columnBlocks(module.columns ?? [], module.groups ?? []), [module.columns, module.groups]);
  const ordered = useMemo(() => blocks.flatMap((b) => b.columns.map((c) => ({ c, g: b.group, first: c === b.columns[0] }))), [blocks]);
  const cellsOf = (e: AnalysisEntry) => e.analysis?.[module.id];
  const keyHeader = module.key_header ?? null;
  const keyOf = (e: AnalysisEntry) => (cellsOf(e)?._key as string | null | undefined) ?? null;

  return (
    <div style={{ marginTop: 8 }}>
      <style>{CSS}</style>

      {!narrow && <div className="ra-wide ra-scroll">
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
                <th key={c.key} className={[first ? "gs" : "", c.fmt === "range" ? "rg" : "", c.fmt === "runs" ? "rn" : ""].filter(Boolean).join(" ") || undefined} title={c.title ?? undefined}>{c.label}</th>
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
                        {c.fmt === "runs" ? <RunChips v={v} /> : c.fmt === "range" ? <RangeBar v={v} /> : formatCell(v, c.fmt)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>}

      {narrow && <div className="ra-cards">
        <MobileTable entries={entries} module={module} blocks={blocks} keyHeader={keyHeader} />
      </div>}
    </div>
  );
}
