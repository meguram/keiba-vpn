"use client";

import { MIN_N, combine, type CRFigure, type CRPeriods, type CRRule, type CRStat } from "@/lib/courseRules";
import { pct, rec } from "@/lib/gradedPicks";

const CSS = `
.cr-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.cr{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.cr th,.cr td{padding:3px 6px;text-align:right;white-space:nowrap}
.cr thead th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border)}
.cr tbody td{border-bottom:1px solid rgba(36,48,73,.35)}
.cr .l{text-align:left}
.cr .up{color:#4ade80;font-weight:700}
.cr tr.dim td{color:var(--text-dim)}
.cr .sub{display:block;font-size:10.5px;color:var(--text-dim);font-weight:400}
.cr-rule{border:1px solid var(--border);border-radius:8px;padding:8px;margin:8px 0}
.cr-rule h4{font-size:13px;margin:0 0 6px;font-weight:600;line-height:1.5}
.cr-sec{margin-top:12px}
.cr-sec h3{font-size:13.5px;margin:0 0 4px}
.cr-note{font-size:11.5px;color:var(--text-dim);margin:4px 0}
.cr-warn{font-size:12px;color:#fbbf24;margin:6px 0}
.cr-more>summary{cursor:pointer;font-size:12px;color:var(--accent);margin-top:4px}
`;

/** 値と、同じ人気の馬の値（小さく下に）。good = この向きなら強調（件数が MIN_N 以上のときだけ） */
function Cell({ v, base, good, n }: { v: number | null | undefined; base: number | null | undefined; good: "higher" | null; n: number }) {
  const up = good === "higher" && n >= MIN_N && v != null && base != null && v > base;
  return (
    <td className={up ? "up" : ""}>
      {pct(v)}
      <span className="sub">{base == null ? "" : `同じ人気 ${pct(base)}`}</span>
    </td>
  );
}

function Row({ label, x, kind, dim }: { label: string; x: CRStat | null; kind: "buy" | "kill" | "all"; dim?: boolean }) {
  const b = x?.base_same_pop;
  const n = x?.n ?? 0;
  const good = kind === "buy" && !dim ? "higher" : null;       // 消しの条件・抽出の期間・全馬の行は強調しない
  return (
    <tr className={dim ? "dim" : ""}>
      <td className="l">{label}{x && n > 0 && n < MIN_N ? <span className="sub">件数が少なく参考になりません</span> : null}</td>
      <td>{x ? n.toLocaleString() : "—"}</td>
      <td>{rec(x?.rec)}</td>
      <Cell v={x?.place} base={b?.place} good={good} n={n} />
      <Cell v={x?.roi_t} base={b?.roi_t} good={good} n={n} />
      <Cell v={x?.roi_f} base={b?.roi_f} good={good} n={n} />
    </tr>
  );
}

function Table({ s, kind }: { s: CRPeriods; kind: "buy" | "kill" | "all" }) {
  return (
    <div className="cr-scroll">
      <table className="cr">
        <thead>
          <tr><th className="l">期間</th><th>件数</th><th>1-2-3-着外</th><th>複勝率</th><th>単勝の払戻率</th><th>複勝の払戻率</th></tr>
        </thead>
        <tbody>
          <Row label="2025〜26 年" x={combine(s["2025"], s["2026"])} kind={kind} />
          <Row label="抽出（2016〜24）・条件を選んだ期間" x={s.train} kind={kind} dim />
        </tbody>
      </table>
      <details className="cr-more">
        <summary>年ごとに見る</summary>
        <table className="cr">
          <tbody>
            <Row label="2025 年" x={s["2025"]} kind={kind} />
            <Row label="2026 年" x={s["2026"]} kind={kind} />
          </tbody>
        </table>
      </details>
    </div>
  );
}

function Rule({ r, kind, i }: { r: CRRule; kind: "buy" | "kill"; i: number }) {
  return (
    <div className="cr-rule">
      <h4>{i + 1}. {r.text}</h4>
      <Table s={r.stats} kind={kind} />
    </div>
  );
}

export function CourseRules({ fig, notes }: { fig: CRFigure; notes?: string[] }) {
  const nb = fig.rules.buy.length;
  const nk = fig.rules.kill.length;
  return (
    <div>
      <style>{CSS}</style>
      <div className="cr-warn">参考です。払戻率は過去の値で、今後を保証しません。このレースのどの馬が条件に当てはまるかは、まだ表示していません。</div>
      {!fig.track_ok && <div className="cr-warn">当日の馬場が「{fig.track}」です。これらは良馬場の条件なので、このレースには当てはまりません。</div>}
      {fig.includes_after && <div className="cr-warn">このレースは結果の最終日（{fig.data_until}）より前です。2025〜26 年の値には、このレース自身とそれより後の結果が入っています（振り返りの参考）。</div>}
      <div className="cr-note">
        単位: {fig.unit_label ?? fig.unit}{fig.layer === "副" ? "（件数が少ないため、コース全体で探した条件）" : ""}。
        条件ごとに、主に 2025〜26 年の値を「同じ人気の馬」の値と比べてください（上回り、件数が {MIN_N} 以上のときだけ緑）。
      </div>
      <div className="cr-sec">
        <h3>過去に 3 着内が多かった条件</h3>
        <div className="cr-note">{fig.n_buy} 個のうち、上位 {nb} 個（市場より強かった度合いの順。この単位で抽出の期間に当たる例が無い条件は除く）。</div>
        {nb ? fig.rules.buy.map((r, i) => <Rule key={`b${i}`} r={r} kind="buy" i={i} />) : <div className="cr-note">この単位には、表示できる条件がありません。</div>}
      </div>
      <div className="cr-sec">
        <h3>過去に 3 着内が少なかった条件</h3>
        <div className="cr-note">{fig.n_kill} 個のうち、上位 {nk} 個（抽出の期間の件数の多い順）。抽出の期間は定義上 3 着内が 0 です。</div>
        {nk ? fig.rules.kill.map((r, i) => <Rule key={`k${i}`} r={r} kind="kill" i={i} />) : <div className="cr-note">この単位には、表示できる条件がありません。</div>}
      </div>
      <div className="cr-sec">
        <h3>この単位の全馬</h3>
        <Table s={fig.unit_stats} kind="all" />
      </div>
      {notes?.length ? (
        <div className="cr-sec">
          {notes.map((n, i) => <div key={i} className="cr-note">・{n}</div>)}
          {fig.data_until ? <div className="cr-note">・結果は {fig.data_until} まで。</div> : null}
        </div>
      ) : null}
    </div>
  );
}
