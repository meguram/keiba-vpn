"use client";

import { PERIOD_LABEL, SKIP_LABEL, fin, killedTop3, odds, pct, periodsOf, rec, yen, type GPFigure, type GPHorse, type GPPicked, type GPRace, type GPRule } from "@/lib/gradedPicks";

const CSS = `
.gp-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.gp{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.gp th,.gp td{padding:4px 6px;text-align:right;white-space:nowrap;vertical-align:middle}
.gp thead th{color:var(--text-dim);font-weight:600;font-size:11px;border-bottom:1px solid var(--border)}
.gp tbody td{border-bottom:1px solid rgba(36,48,73,.35)}
.gp .l{text-align:left}
.gp .up{color:#4ade80;font-weight:700}
.gp tr.hit td{background:rgba(239,68,68,.14)}
.gp tr.hit td.f{color:var(--err);font-weight:700}
.gp-sec{margin-top:14px}
.gp-sec h3{font-size:13.5px;margin:0 0 6px}
.gp-note{font-size:11.5px;color:var(--text-dim);margin:4px 0}
.gp-warn{font-size:12px;color:#fbbf24;margin:6px 0}
.gp-rule{margin:2px 0 6px;padding-left:18px;font-size:12px}
.gp-race{border:1px solid var(--border);border-radius:8px;padding:8px;margin:8px 0}
.gp-race h4{font-size:12.5px;margin:0 0 4px;font-weight:600}
.gp-tag{display:inline-block;font-size:10.5px;padding:0 6px;border:1px solid var(--border);border-radius:8px;color:var(--text-dim);margin-left:6px}
`;

function RuleText({ r, kind }: { r: GPRule; kind: "honmei" | "kill" }) {
  return (
    <div>
      <div className="gp-note">
        {kind === "honmei" ? "次の点を足し合わせ、一番高い馬を ◎ にする。" : `次の点で順位を付け、消す側から ${r.k ?? "—"} 頭を消す（頭数が少ないレースは 3 頭を残す）。`}
        {r.rule.filter ? <b> {r.rule.filter}</b> : null}
      </div>
      <ol className="gp-rule">
        {r.rule.terms.map((t) => (
          <li key={t.key}>{t.text}{t.small ? <span style={{ color: "var(--text-dim)" }}>（ほぼ効かない）</span> : null}</li>
        ))}
      </ol>
      <div className="gp-note">{r.rule.tie}</div>
      {r.dayof && <div className="gp-warn">この規則は当日に決まる値（馬体重・馬場など）を使うので、前日には該当馬を出しません（結果の後に記録します）。</div>}
      {r.default && <div className="gp-warn">2018〜2024 年にこの重賞の回が無く、規則を決められていません（既定の候補が残っただけ）。</div>}
    </div>
  );
}

function HorseRow({ label, h }: { label: string; h: GPHorse | null }) {
  if (!h) return <tr><td className="l">{label}</td><td className="l" colSpan={7} style={{ color: "var(--text-dim)" }}>規則なし</td></tr>;
  return (
    <tr>
      <td className="l">{label}</td>
      <td>{h.hn}</td>
      <td className="l">{h.name ?? "（馬名なし）"}</td>
      <td>{odds(h)}</td>
      <td>{h.pop ?? "—"}</td>
      <td className={h.fin <= 3 ? "up" : ""}>{fin(h)}</td>
      <td>{yen(h.tan)}</td>
      <td>{yen(h.fuku)}</td>
    </tr>
  );
}

function RaceBlock({ x }: { x: GPRace }) {
  const win = x.winners.map((w) => `${w.name ?? "—"}（${w.hn} 番・単勝 ${odds(w)} 倍・${w.pop ?? "—"} 番人気）`).join("・") || "—";
  const kill = [...(x.B.kill ?? [])].sort((a, b) => a.hn - b.hn);
  return (
    <div className="gp-race">
      <h4>
        {x.date} {x.venue}・{x.field} 頭立て・勝ち馬 {win}
        {!x.pay_ok && <span className="gp-tag">払戻のデータなし（回収率の分母に入れていない）</span>}
      </h4>
      <div className="gp-scroll">
        <table className="gp">
          <thead><tr><th className="l">印</th><th>馬番</th><th className="l">馬名</th><th>単勝オッズ</th><th>人気</th><th>着順</th><th>単勝の払戻</th><th>複勝の払戻</th></tr></thead>
          <tbody>
            <HorseRow label="◎（B）" h={x.B.honmei} />
            <HorseRow label="前日版の点の ◎（参考）" h={x.base} />
          </tbody>
        </table>
      </div>
      {x.B.kill == null ? (
        <div className="gp-note">消しの規則なし</div>
      ) : kill.length === 0 ? (
        <div className="gp-note">この回は消した馬なし（頭数が少なく 3 頭を残した）</div>
      ) : (
        <>
          <div className="gp-note">消した馬 {kill.length} 頭のうち 3 着以内 {killedTop3(kill)} 頭（3 着以内に来た馬は赤）</div>
          <div className="gp-scroll">
            <table className="gp">
              <thead><tr><th>馬番</th><th className="l">馬名</th><th>単勝オッズ</th><th>人気</th><th>着順</th></tr></thead>
              <tbody>
                {kill.map((h) => (
                  <tr key={h.hn} className={h.fin <= 3 ? "hit" : ""}>
                    <td>{h.hn}</td><td className="l">{h.name ?? "（馬名なし）"}</td><td>{odds(h)}</td><td>{h.pop ?? "—"}</td><td className="f">{fin(h)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function StatsTable({ fig }: { fig: GPFigure }) {
  const ps = periodsOf(fig.stats);
  if (!ps.length) return <div className="gp-note">このレースより前の回の記録がありません。</div>;
  return (
    <div className="gp-scroll">
      <table className="gp">
        <thead>
          <tr><th className="l">期間</th><th className="l">選び方</th><th>回</th><th>着（1-2-3-外）</th><th>勝率</th><th>複勝率</th><th>単回収</th><th>複回収</th><th>消し馬の 3 着以内</th></tr>
        </thead>
        <tbody>
          {ps.flatMap((p) => {
            const s = fig.stats[p]!;
            const rows = [
              { who: "◎（B）", h: s.B_honmei, k: s.B_kill },
              { who: "前日版の点の ◎（参考）", h: s.base_honmei, k: null },
            ];
            return rows.map((r, i) => (
              <tr key={`${p}-${i}`}>
                {i === 0 && <td className="l" rowSpan={rows.length} style={{ whiteSpace: "normal", minWidth: "10em" }}>{PERIOD_LABEL[p]}</td>}
                <td className="l">{r.who}</td>
                <td>{r.h?.races ?? "—"}</td>
                <td>{rec(r.h?.rec)}</td>
                <td>{pct(r.h?.win)}</td>
                <td>{pct(r.h?.place)}</td>
                <td className={r.h?.roi_t != null && r.h.roi_t > 1 ? "up" : ""}>{pct(r.h?.roi_t)}</td>
                <td className={r.h?.roi_f != null && r.h.roi_f > 1 ? "up" : ""}>{pct(r.h?.roi_f)}</td>
                <td>{r.k ? `${pct(r.k.top3_rate)}（${r.k.killed} 頭）` : "—"}</td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}

function PickedBlock({ p }: { p: GPPicked }) {
  const kill = [...(p.B.kill ?? [])].sort((a, b) => a.hn - b.hn);
  const sk = p.skipped ?? {};
  return (
    <div className="gp-race">
      <h4>{p.date}・{p.field} 頭（出馬表の確定後に計算。結果はまだありません）</h4>
      <div className="gp-scroll">
        <table className="gp">
          <thead><tr><th className="l">印</th><th>馬番</th><th className="l">馬名</th></tr></thead>
          <tbody>
            <tr><td className="l">◎（B）</td>{p.B.honmei ? <><td>{p.B.honmei.hn}</td><td className="l">{p.B.honmei.name ?? "（馬名なし）"}</td></> : <td className="l" colSpan={2} style={{ color: "var(--text-dim)", whiteSpace: "normal" }}>{SKIP_LABEL[sk.honmei ?? "no_rule"]}</td>}</tr>
            <tr><td className="l">前日版の点の ◎（参考）</td>{p.base ? <><td>{p.base.hn}</td><td className="l">{p.base.name ?? "（馬名なし）"}</td></> : <td colSpan={2}>—</td>}</tr>
          </tbody>
        </table>
      </div>
      {p.B.kill == null ? (
        <div className="gp-note">消し: {SKIP_LABEL[sk.kill ?? "no_rule"]}</div>
      ) : kill.length === 0 ? (
        <div className="gp-note">消した馬なし（頭数が少なく 3 頭を残した）</div>
      ) : (
        <div className="gp-note">消し（{kill.length} 頭）: {kill.map((h) => `${h.hn} ${h.name ?? "（馬名なし）"}`).join("・")}</div>
      )}
      <div className="gp-note">
        使った結果は {p.data_until} まで（レースの前日以前。前日の結果は翌朝まで手元に無いので、日曜の重賞は金曜までの結果になることがあります）。
        当日に決まる値（馬体重・馬場など）と、出馬表の段階で分からないブリンカーは使っていません。計算 {p.computed_at.slice(0, 16).replace("T", " ")}。
      </div>
    </div>
  );
}

const THIS_MSG: Record<string, string> = {
  pending: "出馬表の確定後に表示します（前日に計算）。結果が出た後は、記録の更新後に結果つきで出ます。",
  dayof: "この重賞の規則は当日に決まる値（馬体重・馬場など）を使うので、前日には該当馬を出しません。結果の後に「これまで」に記録します。",
  default: "2018〜2024 年にこの重賞の回が無く、規則を決められていないため、該当馬を出しません。",
};

/** 重賞の「データで予想」: このレースの該当馬・規則・これまでの成績・回ごとの該当馬。 */
export function GradedPicks({ fig, notes }: { fig: GPFigure; notes?: string[] }) {
  const t = fig.this;
  return (
    <div style={{ marginTop: 8 }}>
      <style>{CSS}</style>
      {/* 注意書き（モジュールの notes）。先頭の 2 つ（2024 年までで決めた・まっさらな検証ではない）は目立たせる */}
      {(notes ?? []).slice(0, 2).map((n) => <div key={n} className="gp-warn">{n}</div>)}
      {(notes ?? []).length > 2 && (
        <ul style={{ margin: "4px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--text-dim)" }}>
          {(notes ?? []).slice(2).map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      <div className="gp-sec">
        <h3>
          このレースの該当馬（{fig.rkey}・{fig.grade}・{fig.course}）
          {t.race && t.period === "dev" && <span className="gp-tag">規則を選んだ期間の回（当てはめ）</span>}
        </h3>
        {t.race ? <RaceBlock x={t.race} /> : t.picked ? <PickedBlock p={t.picked} /> : <div className="gp-note">{THIS_MSG[t.status] ?? THIS_MSG.pending}</div>}
      </div>
      <div className="gp-sec">
        <h3>成績・回収率（このレースより前の回）</h3>
        <StatsTable fig={fig} />
        <div className="gp-note">回収率は 100 円ずつ買ったときの払戻の割合（100% が収支ゼロ）で、払戻の解けた回だけで割っています。2018〜2024 年は規則を選んだ期間なので良く見えます。</div>
      </div>
      <div className="gp-sec">
        <h3>選び方（B: 2018〜2024 年の中で決めて凍結）</h3>
        <div className="gp-note"><b>◎</b></div>
        <RuleText r={fig.rules.honmei} kind="honmei" />
        {fig.rules.kill && (<><div className="gp-note"><b>消し</b></div><RuleText r={fig.rules.kill} kind="kill" /></>)}
      </div>
      <div className="gp-sec">
        <h3>これまでの該当馬（新しい順）</h3>
        {fig.past.length === 0 && <div className="gp-note">このレースより前の回の記録がありません。</div>}
        {fig.past.filter((x) => x.date.slice(0, 4) >= "2025").map((x) => <RaceBlock key={x.race_id} x={x} />)}
        {fig.past.some((x) => x.date.slice(0, 4) < "2025") && (
          <details>
            <summary style={{ fontSize: 12, cursor: "pointer" }}>2018〜2024 年の回（規則を選んだ期間・当てはめ）</summary>
            {fig.past.filter((x) => x.date.slice(0, 4) < "2025").map((x) => <RaceBlock key={x.race_id} x={x} />)}
          </details>
        )}
      </div>
      {fig.frozen_at && <div className="gp-note">規則の凍結: {fig.frozen_at.slice(0, 10)}・記録の最終日: {fig.data_until ?? "—"}</div>}
    </div>
  );
}
