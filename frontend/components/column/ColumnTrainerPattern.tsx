"use client";

import { Fragment, useMemo, useState } from "react";
import { OddsDist } from "@/components/OddsDistPopup";
import type { ColumnTrainerPatternBlock } from "@/lib/columns";
import { levelsOf, pct, rowStats, splitPat, sumRows, type TPRow, type TPTable } from "@/lib/trainerPattern";

type SortKey = "name" | "n" | "heads" | "win" | "place" | "winRoi" | "placeRoi";
type Node = { id: string; name: string; sub?: string; row: TPRow; kids?: Node[] };
type Flat = { id: string; trainer: string; place: string; pat: string; row: TPRow };

const COLS: { key: SortKey; label: string; left?: boolean }[] = [
  { key: "n", label: "出走" }, { key: "heads", label: "頭数" }, { key: "win", label: "勝率" },
  { key: "place", label: "複勝率" }, { key: "winRoi", label: "単勝回収率" }, { key: "placeRoi", label: "複勝回収率" },
];
const PAGE = 200;

const CSS = `
.tpc-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tpc{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.tpc th,.tpc td{padding:5px 6px;text-align:right;white-space:nowrap;vertical-align:middle}
.tpc th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border);cursor:pointer;user-select:none}
.tpc th.l,.tpc td.l{text-align:left}
.tpc td{border-bottom:1px solid rgba(36,48,73,.35)}
.tpc tr.t1 td{font-weight:700;background:rgba(148,163,184,.06)}
.tpc tr.t1{cursor:pointer}
.tpc tr.t2 td.l{padding-left:22px}
.tpc tr.t3 td.l{padding-left:40px}
.tpc .up{color:#4ade80;font-weight:700}
.tpc .thin{opacity:.5}
.tpc-ctl{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin:8px 0;font-size:12px;color:var(--text-dim)}
.tpc-ctl select,.tpc-ctl input[type=search]{min-height:34px;padding:0 8px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:14px}
.tpc-ctl input[type=search]{flex:1 1 9em;min-width:0}
.tpc-ctl label{display:inline-flex;align-items:center;gap:4px}
`;

function stat(r: TPRow, k: SortKey): number {
  const s = rowStats(r);
  switch (k) {
    case "n": return s.n;
    case "heads": return s.heads ?? -1;
    case "win": return s.win ?? -1;
    case "place": return s.place ?? -1;
    case "winRoi": return s.winRoi ?? -1;
    default: return s.placeRoi ?? -1;
  }
}

function Cells({ r, t, edges, maxOdds }: { r: TPRow; t: string; edges: number[]; maxOdds: number }) {
  const s = rowStats(r);
  const roi = (v: number | null) => <td className={v != null && v > 1 ? "up" : ""}>{pct(v)}</td>;
  return (
    <>
      <td>{s.n}</td><td>{s.heads ?? "–"}</td>
      <td>{s.c1}-{s.c2}-{s.c3}-{s.c4p}</td>
      <td>{pct(s.win)}</td><td>{pct(s.place)}</td>
      {roi(s.winRoi)}{roi(s.placeRoi)}
      <td style={{ textAlign: "center" }}><OddsDist row={r} edges={edges} maxOdds={maxOdds} title={t} /></td>
    </>
  );
}

function build(t: TPTable): { tree: Node[]; flat: Flat[] } {
  const tree: Node[] = [];
  const flat: Flat[] = [];
  for (const [tid, tr] of Object.entries(t.trainers)) {
    const byPlace = new Map<string, [string, TPRow][]>();
    for (const [k, r] of Object.entries(tr.p)) {
      const { place, pat } = splitPat(k);
      if (!byPlace.has(place)) byPlace.set(place, []);
      byPlace.get(place)!.push([pat, r]);
      flat.push({ id: `${tid}|${k}`, trainer: tr.n, place, pat, row: r });
    }
    const places: Node[] = [...byPlace].map(([place, ps]) => ({
      id: `${tid}|${place}`, name: place, row: sumRows(ps.map(([, r]) => r))!,
      kids: ps.map(([pat, r]) => ({ id: `${tid}|${place}|${pat}`, name: pat, row: r })),
    }));
    const all = sumRows(places.map((p) => p.row));
    if (all) tree.push({ id: tid, name: tr.n, row: all, kids: places });
  }
  return { tree, flat };
}

export function ColumnTrainerPattern({ block }: { block: ColumnTrainerPatternBlock }) {
  const [table, setTable] = useState<TPTable | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");
  const [mode, setMode] = useState<"trainer" | "flat">("trainer");
  const [q, setQ] = useState("");
  const [minN, setMinN] = useState(30);
  const [onlyUp, setOnlyUp] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "n", dir: -1 });
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [limit, setLimit] = useState(PAGE);

  const load = () => {
    if (table || state === "loading") return;
    setState("loading");
    fetch(block.src, { cache: "force-cache" })
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((j: TPTable) => { setTable(j); setState("idle"); })
      .catch(() => setState("error"));
  };

  const data = useMemo(() => (table ? build(table) : null), [table]);
  const up = (r: TPRow) => { const s = rowStats(r); return (s.winRoi ?? 0) > 1 || (s.placeRoi ?? 0) > 1; };
  const cmp = (a: TPRow, b: TPRow, an: string, bn: string) => {
    if (sort.key === "name") return an.localeCompare(bn, "ja") * sort.dir;
    return (stat(a, sort.key) - stat(b, sort.key)) * sort.dir || an.localeCompare(bn, "ja");
  };
  const head = (key: SortKey, label: string, left?: boolean) => (
    <th key={key} className={left ? "l" : ""} onClick={() => { setLimit(PAGE); setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : (key === "name" ? 1 : -1) })); }}
      aria-sort={sort.key === key ? (sort.dir > 0 ? "ascending" : "descending") : "none"}>
      {label}{sort.key === key ? (sort.dir > 0 ? " ▲" : " ▼") : ""}
    </th>
  );

  const edges = table?.edges ?? [];
  const maxOdds = table?.max_odds ?? 50;
  const needle = q.trim();

  let body: React.ReactNode = null;
  let count = 0;
  if (data && table) {
    if (mode === "trainer") {
      const ts = data.tree.filter((t) => t.row[0] >= minN && (!needle || t.name.includes(needle)) && (!onlyUp || up(t.row)))
        .sort((a, b) => cmp(a.row, b.row, a.name, b.name));
      count = ts.length;
      const sub = (ns: Node[]) => ns.filter((n) => n.row[0] >= minN && (!onlyUp || up(n.row))).sort((a, b) => cmp(a.row, b.row, a.name, b.name));
      body = ts.map((t) => (
        <Fragment key={t.id}>
          <tr className="t1" onClick={() => setOpen((o) => ({ ...o, [t.id]: !o[t.id] }))} aria-expanded={!!open[t.id]}>
            <td className="l">{open[t.id] ? "▼" : "▶"} {t.name}</td>
            <Cells r={t.row} t={`${t.name} 全体`} edges={edges} maxOdds={maxOdds} />
          </tr>
          {open[t.id] && sub(t.kids ?? []).map((p) => (
            <Fragment key={p.id}>
              <tr className="t2"><td className="l">{p.name}</td><Cells r={p.row} t={`${t.name} ${p.name}`} edges={edges} maxOdds={maxOdds} /></tr>
              {sub(p.kids ?? []).map((k) => (
                <tr key={k.id} className="t3"><td className="l">{k.name}</td><Cells r={k.row} t={`${t.name} ${p.name} ${k.name}`} edges={edges} maxOdds={maxOdds} /></tr>
              ))}
            </Fragment>
          ))}
        </Fragment>
      ));
    } else {
      const fs = data.flat.filter((f) => f.row[0] >= minN && (!needle || f.trainer.includes(needle)) && (!onlyUp || up(f.row)))
        .sort((a, b) => cmp(a.row, b.row, a.trainer + a.place + a.pat, b.trainer + b.place + b.pat));
      count = fs.length;
      body = fs.slice(0, limit).map((f) => (
        <tr key={f.id}>
          <td className="l">{f.trainer}</td><td className="l">{f.place}</td><td className="l">{f.pat}</td>
          <Cells r={f.row} t={`${f.trainer} ${f.place} ${f.pat}`} edges={edges} maxOdds={maxOdds} />
        </tr>
      ));
    }
  }

  const flat = mode === "flat";
  return (
    <details className="card" style={{ padding: 0 }} onToggle={(e) => { if ((e.currentTarget as HTMLDetailsElement).open) load(); }}>
      <summary style={{ padding: "10px 14px", cursor: "pointer", fontWeight: 600, fontSize: 14 }}>{block.title}（開くと表を読み込みます）</summary>
      <div style={{ padding: "0 12px 12px", borderTop: "1px solid var(--border)" }}>
        <style>{CSS}</style>
        {state === "loading" && <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>読み込み中…</p>}
        {state === "error" && <p style={{ fontSize: 13, color: "var(--err)", marginTop: 8 }}>表を読み込めませんでした。</p>}
        {table && data && (
          <>
            <div className="tpc-ctl">
              <label>表示
                <select value={mode} onChange={(e) => { setMode(e.target.value as "trainer" | "flat"); setLimit(PAGE); }}>
                  <option value="trainer">調教師ごと（開閉）</option>
                  <option value="flat">全行を一覧</option>
                </select>
              </label>
              <input type="search" placeholder="調教師名で絞る" value={q} onChange={(e) => { setQ(e.target.value); setLimit(PAGE); }} aria-label="調教師名で絞る" />
              <label>最低出走
                <select value={minN} onChange={(e) => { setMinN(Number(e.target.value)); setLimit(PAGE); }}>
                  {[0, 10, 30, 100, 300].map((n) => <option key={n} value={n}>{n}走以上</option>)}
                </select>
              </label>
              <label><input type="checkbox" checked={onlyUp} onChange={(e) => { setOnlyUp(e.target.checked); setLimit(PAGE); }} />回収率100%超だけ</label>
              <span>{count.toLocaleString("en-US")} 件</span>
            </div>
            <div className="tpc-scroll">
              <table className="tpc">
                <thead>
                  <tr>
                    {flat ? <>{head("name", "調教師", true)}<th className="l" style={{ cursor: "default" }}>場所</th><th className="l" style={{ cursor: "default" }}>パターン</th></> : head("name", "調教師 / 場所 / パターン", true)}
                    {COLS.slice(0, 2).map((c) => head(c.key, c.label))}
                    <th style={{ cursor: "default" }}>着順内訳</th>
                    {COLS.slice(2).map((c) => head(c.key, c.label))}
                    <th style={{ cursor: "default" }} title="オッズ区間別の出走数。カーソルを合わせる／タップで詳細">オッズ分布</th>
                  </tr>
                </thead>
                <tbody>{body}</tbody>
              </table>
            </div>
            {flat && count > limit && (
              <button onClick={() => setLimit((l) => l + PAGE)} style={{ marginTop: 8, padding: "6px 14px", borderRadius: 999, border: "1px solid var(--border)", background: "transparent", color: "var(--text)", cursor: "pointer", fontSize: 13 }}>
                さらに {Math.min(PAGE, count - limit)} 件を表示
              </button>
            )}
            <p style={{ margin: "8px 0 0", fontSize: 11, color: "var(--text-dim)" }}>
              {block.caption ?? ""} 集計 {table.y0}〜{table.y1} 年、単勝 {table.max_odds} 倍以上は除く。見出しをクリックで並べ替え。
            </p>
          </>
        )}
      </div>
    </details>
  );
}
