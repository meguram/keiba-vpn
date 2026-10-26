"use client";

import { useEffect, useState } from "react";
import { PendingLink } from "@/components/PendingLink";
import { getJson } from "@/lib/httpCache";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";

type LevelRow = { level: number; label: string; races: number; hit_races: number; hit_rate: number | null; invested: number; returned: number; roi: number | null };
type LevelPeriod = {
  key: string; label: string; from: string; to: string; data_from: string | null; data_to: string | null; days: number;
  levels: LevelRow[]; total: { races: number; hit_races: number; invested: number; returned: number; roi: number | null; hit_rate: number | null };
  skipped_races: number; unrated_races: number;
};
type LevelCard = { periods: LevelPeriod[]; note?: string };

const SMALL_SAMPLE = 30;   // これ未満のレース数では回収率を目安にしない注記を出す
const yen = (n: number) => `${Math.round(n).toLocaleString("ja-JP")}円`;
// 回収率の色: 100%以上=緑、80%以上=黄、それ未満=灰
const roiColor = (r: number | null) => (r == null ? "var(--text-dim)" : r >= 1 ? "#4ade80" : r >= 0.8 ? "#fbbf24" : "var(--text-dim)");
const md = (ymd: string) => `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}`;
const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

const CSS = `
.mrc{max-width:1100px;margin:0 auto 16px;padding:0 20px}
.mrc-panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.mrc-head{display:flex;align-items:center;gap:8px;padding:10px 12px 0}
.mrc-head h2{margin:0;font-size:14px;font-weight:700;color:var(--text)}
.mrc-head .mrc-link{margin-left:auto;font-size:11px;color:var(--accent);text-decoration:none;white-space:nowrap}
.mrc-tabs{display:flex;gap:4px;padding:8px 12px 0}
.mrc-tab{appearance:none;flex:1;border:1px solid var(--border);background:transparent;color:var(--text-dim);font-size:12px;font-weight:600;padding:4px 6px;border-radius:999px;cursor:pointer;min-height:28px;white-space:nowrap;transition:all .15s}
.mrc-tab:hover{color:var(--text);border-color:rgba(148,163,184,.5)}
.mrc-tab[aria-pressed="true"]{background:rgba(88,166,255,.14);border-color:rgba(88,166,255,.55);color:#9ecbff}
.mrc-meta{padding:6px 12px 8px;font-size:11px;color:var(--text-dim)}
.mrc-meta b{color:var(--text);font-weight:700}
.mrc-empty{padding:14px 12px;font-size:12px;color:var(--text-dim)}
.mrc-bets{border-top:1px solid var(--border);padding:8px 12px 4px}
.mrc-bets h3{margin:0 0 4px;font-size:13px;font-weight:700;color:var(--text)}
.mrc-bets-note{font-size:10px;color:var(--text-dim);line-height:1.5;padding:4px 0 6px}
.mrc-lv{display:grid;grid-template-columns:52px 1fr auto;column-gap:8px;align-items:baseline;padding:5px 0;border-top:1px solid var(--border);font-variant-numeric:tabular-nums}
.mrc-lv:first-of-type{border-top:none}
.mrc-lv-name{font-size:12px;font-weight:700;color:var(--text);white-space:nowrap}
.mrc-lv-sub{font-size:11px;color:var(--text-dim);line-height:1.4}
.mrc-lv-roi{font-size:14px;font-weight:800;white-space:nowrap}
@media (max-width:767px){.mrc{padding:0 12px}}
`;

/** 推奨馬券の回収率（推奨度別）。見送りは除く。同じ期間タブに連動。会員のみ（API が会員限定）。 */
function LevelSection({ p, note }: { p: LevelPeriod; note?: string }) {
  const bought = p.total.races;
  return (
    <div className="mrc-bets" aria-label="推奨馬券の回収率（推奨度別）">
      <h3>推奨度別の回収率</h3>
      {bought === 0 ? (
        <div className="mrc-lv-sub" style={{ padding: "4px 0 6px" }}>この期間に、見送り以外で集計できる推奨馬券がありません{p.skipped_races ? `（見送り ${p.skipped_races}R）` : ""}</div>
      ) : (
        <div role="list">
          {p.levels.map((l) => (
            <div key={l.level} role="listitem" className="mrc-lv">
              <span className="mrc-lv-name">{l.label}</span>
              <span className="mrc-lv-sub">
                {l.races > 0 ? <>{l.races}R・的中 {l.hit_races}R（{pct(l.hit_rate)}）<br />{yen(l.invested)} → {yen(l.returned)}</> : "該当レースなし"}
              </span>
              <span className="mrc-lv-roi" style={{ color: roiColor(l.roi) }}>{l.roi == null ? "—" : pct(l.roi)}</span>
            </div>
          ))}
          <div className="mrc-lv" style={{ borderTop: "1px solid var(--border)" }}>
            <span className="mrc-lv-name">合計</span>
            <span className="mrc-lv-sub">{bought}R・的中 {p.total.hit_races}R（{pct(p.total.hit_rate)}）<br />{yen(p.total.invested)} → {yen(p.total.returned)}</span>
            <span className="mrc-lv-roi" style={{ color: roiColor(p.total.roi) }}>{pct(p.total.roi)}</span>
          </div>
        </div>
      )}
      {bought > 0 && bought < SMALL_SAMPLE && (
        <div className="mrc-bets-note" style={{ color: "#fbbf24" }}>対象が{bought}レースと少なく、1回の的中で回収率が大きく動きます。目安にはなりません（{SMALL_SAMPLE}レース以上で参考になります）。</div>
      )}
      <div className="mrc-bets-note">
        見送り{p.skipped_races}Rは除く{p.unrated_races > 0 ? `・推奨度の記録が無い${p.unrated_races}Rも除く` : ""}。{note ?? ""}
      </div>
    </div>
  );
}

/** ホームの『推奨馬券の成績』: 先週・直近5週間・年の切り替えで、推奨度（見送りを除く）別の購入レース数・的中・回収率を見せる。会員のみ（API が会員限定）。集計が無ければ何も出さない。 */
export function BetResultCard() {
  const { isMember } = useAuthStatus();
  const [lv, setLv] = useState<LevelCard | null>(null);
  const [sel, setSel] = useState("recent_5w");

  useEffect(() => {
    if (!isMember) { setLv(null); return; }
    let alive = true;
    getJson<LevelCard>("/api/prediction-results/bets-by-level", { ttlMs: 60_000 })
      .then((d) => { if (alive) setLv(d); })
      .catch(() => { if (alive) setLv(null); });
    return () => { alive = false; };
  }, [isMember]);

  if (!lv || !lv.periods.some((p) => p.days > 0)) return null;
  const cur = lv.periods.find((p) => p.key === sel && p.days > 0) ?? lv.periods.find((p) => p.days > 0) ?? lv.periods[0];

  const allRaces = cur.total.races + cur.skipped_races + cur.unrated_races;   // 集計できた全レース（購入＋見送り＋推奨度の記録なし）

  return (
    <section aria-label="推奨馬券の成績" className="mrc">
      <style>{CSS}</style>
      <div className="mrc-panel">
        <div className="mrc-head">
          <h2>推奨馬券の成績</h2>
          <PendingLink href="/prediction-results" className="mrc-link">成績をすべて見る →</PendingLink>
        </div>
        <div className="mrc-tabs" role="group" aria-label="期間">
          {lv.periods.map((p) => (
            <button key={p.key} type="button" className="mrc-tab" aria-pressed={p.key === cur.key} disabled={p.days === 0} style={p.days === 0 ? { opacity: 0.4, cursor: "default" } : undefined} onClick={() => setSel(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="mrc-meta">
          <b>{cur.data_from && cur.data_to ? `${md(cur.data_from)}〜${md(cur.data_to)}` : `${md(cur.from)}〜${md(cur.to)}`}</b>・開催 {cur.days}日
          {allRaces > 0 && (
            <> ・全 {allRaces}R のうち <b>{cur.total.races}R に参戦</b>
              {cur.total.races > 0 ? `（約 ${(allRaces / cur.total.races).toFixed(1)} レースに1回・${pct(cur.total.races / allRaces)}）` : "（見送りのみ）"}</>
          )}
        </div>
        <LevelSection p={cur} note={lv.note} />
      </div>
    </section>
  );
}
