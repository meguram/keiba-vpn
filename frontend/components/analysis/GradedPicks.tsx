"use client";

import { SKIP_LABEL, fin, horseShort, newVersion, odds, pct, rec, yen, type GPFigure, type GPHorse, type GPPicked, type GPRace, type GPRule, type GPSimple } from "@/lib/gradedPicks";

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
.gp-sec{margin-top:12px}
.gp-sec h3{font-size:13.5px;margin:0 0 6px}
.gp-note{font-size:11.5px;color:var(--text-dim);margin:4px 0}
.gp-warn{font-size:12px;color:#fbbf24;margin:6px 0}
.gp-rule{margin:2px 0 6px;padding-left:18px;font-size:12px}
.gp-race{border:1px solid var(--border);border-radius:8px;padding:8px;margin:8px 0}
.gp-race h4{font-size:12.5px;margin:0 0 4px;font-weight:600}
.gp-tag{display:inline-block;font-size:10.5px;padding:0 6px;border:1px solid var(--border);border-radius:8px;color:var(--text-dim);margin-left:6px}
.gp-pick{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:baseline;font-size:14px}
.gp-pick .k{font-size:12px;color:var(--text-dim);white-space:nowrap}
.gp-pick .hm{font-size:16px;font-weight:700}
.gp-pick .bad{color:var(--err);font-weight:700}
.gp-brief{font-size:12.5px;margin:8px 0 0;padding-left:16px}
.gp-brief li{margin:2px 0}
.gp-more{margin-top:12px;border-top:1px solid var(--border);padding-top:6px}
.gp-more>summary{cursor:pointer;font-size:13px;color:var(--accent)}
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
      {r.dayof && <div className="gp-warn">当日の値（馬体重・馬場など）を使う条件なので、前日には出しません。</div>}
      {r.default && <div className="gp-warn">2018〜2024 年にこの重賞の回が無く、条件を決められていません（既定の候補が残っただけ）。</div>}
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

/** このレースの詳細（確定の単勝オッズ・人気・払戻つき）。「詳しく」の中だけで出す */
function RaceDetail({ x }: { x: GPRace }) {
  const win = x.winners.map((w) => `${w.name ?? "—"}（${w.hn} 番・単勝 ${odds(w)} 倍・${w.pop ?? "—"} 番人気）`).join("・") || "—";
  const kill = [...(x.B.kill ?? [])].sort((a, b) => a.hn - b.hn);
  return (
    <div className="gp-race">
      <h4>
        {x.date} {x.venue}・{x.field} 頭立て・勝ち馬 {win}
        {!x.pay_ok && <span className="gp-tag">払戻のデータなし</span>}
      </h4>
      <div className="gp-scroll">
        <table className="gp">
          <thead><tr><th className="l">印</th><th>馬番</th><th className="l">馬名</th><th>単勝オッズ</th><th>人気</th><th>着順</th><th>単勝の払戻</th><th>複勝の払戻</th></tr></thead>
          <tbody>
            <HorseRow label="◎" h={x.B.honmei} />
            <HorseRow label="前日版の ◎（参考）" h={x.base} />
            {kill.map((h) => <HorseRow key={h.hn} label="消し" h={h} />)}
          </tbody>
        </table>
      </div>
      <div className="gp-note">単勝オッズ・人気は確定の値で、結果を見せるためだけに使っています（選び方には使っていません）。</div>
    </div>
  );
}

/** 選ばれた馬（最初に見せる）。結果があれば着順、前日の計算なら馬番・馬名だけ */
function Picks({ fig }: { fig: GPFigure }) {
  const t = fig.this;
  const r = t.race;
  const p: GPPicked | null | undefined = t.picked;
  const hm = r ? r.B.honmei : p?.B.honmei ?? null;
  const kill: { hn: number; name: string | null; fin?: number | null }[] | null = r ? r.B.kill : p?.B.kill ?? null;
  // 出さない理由: 前日の計算の skipped、なければ規則が既定（決められていない）か
  const sk = { ...(fig.rules.honmei.default ? { honmei: "default" as const } : {}), ...(fig.rules.kill?.default ? { kill: "default" as const } : {}), ...(p?.skipped ?? {}) };
  if (!r && !p) return <div className="gp-note" style={{ fontSize: 13 }}>{THIS_MSG[t.status] ?? THIS_MSG.pending}</div>;
  return (
    <div className="gp-pick">
      <span className="k">◎</span>
      <span className="hm">{hm ? horseShort(hm) : <span style={{ fontSize: 12, fontWeight: 400, color: "var(--text-dim)" }}>{SKIP_LABEL[sk.honmei ?? "no_rule"]}</span>}</span>
      <span className="k">消し</span>
      <span>
        {kill == null ? <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{SKIP_LABEL[sk.kill ?? "no_rule"]}</span>
          : kill.length === 0 ? <span style={{ fontSize: 12, color: "var(--text-dim)" }}>なし（頭数が少なく 3 頭を残した）</span>
          : [...kill].sort((a, b) => a.hn - b.hn).map((h, i) => (
            <span key={h.hn} className={h.fin != null && h.fin <= 3 ? "bad" : ""}>{i ? "・" : ""}{horseShort(h)}</span>
          ))}
      </span>
    </div>
  );
}

const THIS_MSG: Record<string, string> = {
  pending: "出馬表の確定後に表示します（前日に計算）。",
  dayof: "この重賞の条件は当日の値（馬体重・馬場など）を使うため、前日には出しません。",
  default: "2018〜2024 年にこの重賞の回が無く、条件を決められていません。",
};

/** これ未満の回の数の回収率は強調しない（1 回の高配当で大きく動く）。条件を決めた期間（〜2024）の行も強調しない */
const MIN_RACES = 10;

/** 過去該当馬: ◎ と消しの着順の成績・◎ の回収率を、期間ごとに 1 行で */
function PastSimple({ fig }: { fig: GPFigure }) {
  // 古い値（simple_stats が無い）は、期間別の成績から作る（2025〜 は凍結の前の回だけ）
  const ss: { recent: GPSimple | null; dev: GPSimple | null } = fig.simple_stats ?? { recent: fig.stats.eval ?? fig.stats.forward ?? null, dev: fig.stats.dev ?? null };
  const rows = ([["2025〜（条件を決めた後）", ss.recent], ["〜2024（条件を決めた期間）", ss.dev]] as const).filter(([, v]) => v);
  if (!rows.length) return <div className="gp-note">このレースより前の回の記録がありません。</div>;
  const after = fig.same_pop_includes_after;
  return (
    <div className="gp-scroll">
      <table className="gp">
        <thead><tr><th className="l">期間</th><th>回</th><th>◎ の着</th><th>単回収</th><th>複回収</th><th className="l">同じ人気の馬（複勝率・単回収・複回収）</th><th>消し馬の着（頭）</th></tr></thead>
        <tbody>
          {rows.map(([lab, v]) => (
            <tr key={lab}>
              <td className="l">{lab}</td>
              <td>{v!.B_honmei?.races ?? v!.B_kill?.races ?? "—"}</td>
              <td>{rec(v!.B_honmei?.rec)}</td>
              <td className={lab.startsWith("2025") && (v!.B_honmei?.races ?? 0) >= MIN_RACES && v!.B_honmei?.roi_t != null && v!.same_pop?.roi_t != null && v!.B_honmei.roi_t > v!.same_pop.roi_t ? "up" : ""}>{pct(v!.B_honmei?.roi_t)}</td>
              <td className={lab.startsWith("2025") && (v!.B_honmei?.races ?? 0) >= MIN_RACES && v!.B_honmei?.roi_f != null && v!.same_pop?.roi_f != null && v!.B_honmei.roi_f > v!.same_pop.roi_f ? "up" : ""}>{pct(v!.B_honmei?.roi_f)}</td>
              <td className="l" style={{ color: "var(--text-dim)" }}>{v!.same_pop ? `${pct(v!.same_pop.place)}・${pct(v!.same_pop.roi_t)}・${pct(v!.same_pop.roi_f)}` : "—"}</td>
              <td>{v!.B_kill ? `${rec(v!.B_kill.rec)}（${v!.B_kill.killed}）` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="gp-note">着外は 4 着以下。「同じ人気の馬」は、◎ と同じ人気の馬の、同じ期間の全重賞（G1〜G3）での複勝率・単回収・複回収です。{after ? "この回は結果の最終日より前なので、2025〜 の比べる相手には、この回とそれより後の結果も入っています。" : ""}</div>
    </div>
  );
}

/** 重賞の「好走データ予想」: 選ばれた馬と条件の要約だけを先に見せ、ほかは「詳しく」に入れる。 */
export function GradedPicks({ fig, notes }: { fig: GPFigure; notes?: string[] }) {
  const t = fig.this;
  const b = fig.brief;
  const base = t.race?.base ?? t.picked?.base ?? null;
  const nv = newVersion(t.picked);
  const rt = nv ? t.picked!.version!.rule_text : null;
  return (
    <div style={{ marginTop: 8 }}>
      <style>{CSS}</style>
      <div className="gp-warn">参考です。回収率は保証しません。これまでの確かめでは、この ◎ はオッズに織り込み済みでした。「同じ人気の馬」の値と比べて読んでください。</div>
      <div className="gp-note">
        {nv ? `${(t.picked!.version!.discovery.to ?? "").slice(0, 4)} 年までのこの重賞の回で決めた条件です（直近の回は検証に使用）。` : "2024 年までのデータで決めた条件です（2025 年以降は、その条件で選んだ結果）。"}
        {fig.rkey}・{fig.grade}・{fig.course}
        {t.race && t.period === "dev" && <span className="gp-tag">この回は条件を決めた期間の中</span>}
        {t.picked?.version?.flag === "val_weak" && <span className="gp-tag" title="発見の期間で決めた条件が、検証の 2 回では ◎ の単勝の回収率 100% 未満でした">検証では振るわず</span>}</div>
      <div className="gp-sec">
        <Picks fig={fig} />
        {b && (b.honmei || b.kill) && (
          <ul className="gp-brief">
            {b.honmei && <li>{b.honmei}</li>}
            {b.kill && <li>{b.kill}</li>}
          </ul>
        )}
      </div>
      <details className="gp-more">
        <summary>詳しく</summary>
        <div className="gp-sec">
          <h3>過去該当馬（このレースより前の回）</h3>
          <PastSimple fig={fig} />
          <div className="gp-note">回収率は 100 円ずつ買ったときの払戻の割合（100% が収支ゼロ）。〜2024 は条件を決めた期間なので良く見えます。100% を超えていても、儲かる根拠にはなりません。このタブの ◎ とは別の版（重賞の買いの条件）の診断でも、単勝の回収率は、条件を探した期間の中で 147.7%、2016〜21 年で探して 2022〜24 年に当てると 72.8%、2025〜26 年は 73.0% でした。消し馬の着は、消した馬の頭数で数えます。</div>
        </div>
        {base && !t.race && <div className="gp-note">前日版の点の ◎（参考）: {horseShort(base)}</div>}
        {t.race && <div className="gp-sec"><h3>このレースの結果</h3><RaceDetail x={t.race} /></div>}
        {t.picked && <div className="gp-note">使った結果は {t.picked.data_until} まで。当日の値とブリンカーは使っていません。計算 {t.picked.computed_at.slice(0, 16).replace("T", " ")}。</div>}
        {t.picked?.version && (
          <div className="gp-note">
            条件の版: {t.picked.version.id}（発見 {t.picked.version.discovery.from ?? "—"}〜{t.picked.version.discovery.to ?? "—"}・検証 {t.picked.version.validation.from ?? "—"}〜{t.picked.version.validation.to ?? "—"}）
            {t.picked.version.val_stats?.honmei?.races ? `。検証の ◎ ${rec(t.picked.version.val_stats.honmei.rec)}・単回収 ${pct(t.picked.version.val_stats.honmei.roi_t)}` : ""}
          </div>
        )}
        {(t.picked?.warnings ?? []).map((w) => <div key={w} className="gp-warn">{w}</div>)}
        <div className="gp-sec">
          {rt ? (
            <>
              <h3>条件の詳細（{t.picked!.version!.discovery.from}〜{t.picked!.version!.discovery.to} の回で決めた版）</h3>
              {rt.honmei && (<><div className="gp-note"><b>◎</b></div><RuleText r={{ rule: rt.honmei, label: "", default: false, dayof: false }} kind="honmei" /></>)}
              {rt.kill && (<><div className="gp-note"><b>消し</b></div><RuleText r={{ rule: rt.kill, label: "", default: false, dayof: false, k: rt.kill.k }} kind="kill" /></>)}
            </>
          ) : (
            <>
              <h3>条件の詳細（2018〜2024 年の中で決めて凍結）</h3>
              <div className="gp-note"><b>◎</b></div>
              <RuleText r={fig.rules.honmei} kind="honmei" />
              {fig.rules.kill && (<><div className="gp-note"><b>消し</b></div><RuleText r={fig.rules.kill} kind="kill" /></>)}
            </>
          )}
        </div>
        {(notes ?? []).length > 0 && (
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--text-dim)" }}>
            {(notes ?? []).map((n) => <li key={n}>{n}</li>)}
          </ul>
        )}
        {fig.frozen_at && <div className="gp-note">条件の凍結: {fig.frozen_at.slice(0, 10)}・記録の最終日: {fig.data_until ?? "—"}</div>}
      </details>
    </div>
  );
}
