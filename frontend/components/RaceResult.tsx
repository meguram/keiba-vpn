"use client";

import { useState, type CSSProperties } from "react";
import type { RaceLevel, RaceResultData, TrackBias } from "@/lib/hooks/useRaceResult";
import { MARK_ORDER } from "@/lib/tableSort";
import { TONE_COLOR, earlySortKey, headline, labelBadges, positionShort, positionText, type BiasBadge } from "@/lib/trackBias";

const TH: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12, color: "var(--text-dim)", fontWeight: 600, whiteSpace: "nowrap" };
const TD: CSSProperties = { padding: "5px 8px", textAlign: "right", fontSize: 12.5, borderTop: "1px solid var(--border)", verticalAlign: "middle", whiteSpace: "nowrap" };
const POS_COLOR: Record<number, string> = { 1: "#fbbf24", 2: "#cbd5e1", 3: "#d97706" };
const yen = (n: number) => `${n.toLocaleString("ja-JP")}円`;

/** そのレースのめぐAI印（馬番 → ◎○▲★△消。無印は入れない）。予測の表と同じ印（確定したレースは発走前の最後のオッズで付けた印） */
export type MarkMap = Record<number, string>;


/** レース結果の上の欄（レースレベル・トラックバイアス）の共通の枠。2 つの欄の見た目をそろえる */
export const NOTE_BOX: CSSProperties = { margin: "0 0 10px", padding: "8px 10px", border: "1px solid var(--border)", borderRadius: 8, background: "var(--surface)", fontSize: 12.5, lineHeight: 1.7 };

/** 秒の値（負なら速い）を「−0.8 秒」の形に */
const secText = (v: number | null | undefined) => (v == null ? null : `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(1)} 秒`);

/** レース結果の上の 3 行: レースレベル（ペース補正後と補正前・ペース）・開催日レベル・メンバーレベル。値は事前に計算したものを読むだけ */
export function RaceLevelNote({ lv }: { lv?: RaceLevel }) {
  if (!lv) return null;
  const box = NOTE_BOX;
  if (!lv.available) return <div style={box}><b>レースレベル</b>：算出なし</div>;
  const r = lv.race, d = lv.day, m = lv.member;
  const withWord = (sec: number | null | undefined, w: string | null | undefined) => (sec == null ? "算出なし" : `${secText(sec)}${w ? `・${w}` : ""}`);
  const memberText = m?.sec == null ? "算出なし"
    : Math.abs(m.sec) < 0.05 ? `同じクラスの基準と同じくらい${m.word ? `（${m.word}）` : ""}`
    : `同じクラスの基準より ${Math.abs(m.sec).toFixed(1)} 秒${m.sec < 0 ? "速い" : "遅い"}${m.word ? `（${m.word}）` : ""}`;
  return (
    <div style={box} data-testid="race-level">
      <div><b>レースレベル（ペース補正後）</b>：{withWord(r?.sec, r?.word)}
        {r?.raw_sec != null && <span style={{ color: "var(--text-dim)" }}> ／ 補正前 {secText(r.raw_sec)}</span>}
        {r?.pace && <span style={{ color: "var(--text-dim)" }}> ／ ペース：{r.pace}</span>}</div>
      <div><b>開催日レベル{d?.venue ? `（${d.venue}${d.surf ?? ""}）` : ""}</b>：{withWord(d?.sec, d?.word)}</div>
      <div><b>メンバーレベル（馬場・ペース補正後）</b>：{memberText}</div>
      <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-dim)" }}>
        秒はこのレースの距離の分で、負の値は基準より速い。{lv.formula} 基準の時計の関数は、従来の馬場の速さの物差しより一つ抜きのばらつきで 0.02 秒だけ良かった（同じ日の一貫性は従来の方が高い）。採否に使っていない 2025 年でも一つ抜きのばらつきの勝ちは続くが、土日の相関の勝ちは続かない。
      </div>
    </div>
  );
}

/** 符号つきの数（−0.53 の形） */
const signed = (v: number | null | undefined, nd = 2) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(nd)}`);

/** 前後・内外の小さな札（色つき）。強いの段は太字 */
function Badge({ b }: { b: BiasBadge }) {
  const c = TONE_COLOR[b.tone];
  return (
    <span style={{ display: "inline-block", padding: "0 8px", borderRadius: 999, fontSize: 11.5, lineHeight: "20px", border: `1px solid ${c}`, color: c, background: `${c}1a`, fontWeight: b.strong ? 800 : 600, whiteSpace: "nowrap" }}>
      {b.text}
    </span>
  );
}

/** トラックバイアス（その日・その場・その面）。1 行目に名札と前後・内外の札、数値と通過順は「詳しく」の中、長い説明は「？」の中。値は事前に計算したものを読むだけ */
export function TrackBiasNote({ tb }: { tb?: TrackBias }) {
  const [help, setHelp] = useState(false);
  if (!tb) return null;
  if (!tb.available) return <div style={NOTE_BOX}><b>トラックバイアス</b>：{tb.note ?? "算出前"}</div>;
  const d = tb.day, r = tb.race;
  const badges = labelBadges(d?.front_label, d?.inner_label, d?.label);
  const pos = (t: NonNullable<TrackBias["top3"]>[number]) => (t.passing_order == null || t.early == null ? "通過なし" : `通過 ${t.passing_order}`);
  const dim: CSSProperties = { color: "var(--text-dim)" };
  return (
    <div style={NOTE_BOX} data-testid="track-bias">
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ ...dim, fontSize: 11.5 }}>トラックバイアス</span>
        <b style={{ fontSize: 15 }}>{headline(d?.venue, d?.surf)}</b>
        {badges.map((b) => <Badge key={b.text} b={b} />)}
      </div>
      <div style={{ ...dim, fontSize: 11, display: "flex", alignItems: "center", gap: 6 }}>
        その日の結果の振り返り・偶然の幅が大きい
        <button type="button" aria-label="トラックバイアスの説明" aria-expanded={help} onClick={() => setHelp((v) => !v)}
          style={{ width: 24, height: 24, lineHeight: "22px", padding: 0, borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text-dim)", fontSize: 11, cursor: "pointer" }}>？</button>
      </div>
      {help && (
        <div role="note" style={{ margin: "4px 0", padding: "6px 8px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--surface2)", fontSize: 11.5 }}>
          {tb.note} 正の値は前・内の馬が見込みより来た向き。{tb.formula}
        </div>
      )}
      <details style={{ marginTop: 2 }}>
        <summary style={{ cursor: "pointer", fontSize: 11.5, color: "var(--accent, #60a5fa)" }}>詳しく</summary>
        <div style={{ fontSize: 12 }}>
          <div>その日: 前後 z {signed(d?.z_front, 1)}・内外 z {signed(d?.z_inner, 1)}・{d?.n_races ?? "—"} レース</div>
          <div>このレースの効き: 前後 {signed(r?.front_se)}（日の z への寄与 {signed(r?.front_zc)}）・内外 {signed(r?.inner_se)}（寄与 {signed(r?.inner_zc)}）・4 角の位置で測ると前後 {signed(r?.front4c_se)}</div>
          {(tb.top3 ?? []).length > 0 && (
            <div>1〜3 着の通過順: {(tb.top3 ?? []).map((t) => `${t.pos}着 ${t.horse_number}番（${pos(t)}）`).join(" ／ ")}</div>
          )}
        </div>
      </details>
    </div>
  );
}

/** 週間予測の『レース結果』タブ。着順表（めぐAI印つき）と払戻表。速報は払戻が揃うまで『速報』と出す。 */
export function RaceResult({ result, marks }: { result: RaceResultData; marks?: MarkMap }) {
  const finish = result.finish ?? [];
  const payoff = result.payoff ?? [];
  const provisional = result.status === "provisional" || !result.official;
  const hasMarks = !!marks && Object.keys(marks).length > 0;
  // 位置の列（序盤 → 4 角の 番手/頭数）は、トラックバイアスが出ているときだけ。頭数はトラックバイアスの 1〜3 着の頭数（発走した馬）。
  // 1〜3 着が無いときだけ、通過順のある行の数で代える（通過順の無い中止の馬の分だけ少ない）
  const tb = !provisional && result.track_bias?.available ? result.track_bias : undefined;
  const field = tb ? (tb.top3?.[0]?.field ?? finish.filter((f) => f.passing_order).length) : 0;
  return (
    <div style={{ padding: "12px 0" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8, fontSize: 12.5 }}>
        <b style={{ color: "var(--text)" }}>レース結果</b>
        <span style={{ padding: "1px 8px", borderRadius: 999, fontSize: 11, border: `1px solid ${provisional ? "#fbbf24" : "#34d399"}`, color: provisional ? "#fbbf24" : "#34d399" }}>
          {provisional ? "速報（払戻・確定は取得後に更新）" : "確定"}
        </span>
        {result.fetched_at && <span style={{ color: "var(--text-dim)", fontSize: 11.5 }}>取得 {result.fetched_at}</span>}
      </div>

      {!provisional && <RaceLevelNote lv={result.race_level} />}
      {!provisional && <TrackBiasNote key={result.race_id} tb={result.track_bias} />}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", background: "var(--surface)" }}>
          <thead style={{ background: "var(--surface2)" }}>
            <tr>
              <th style={TH}>着順</th>
              <th style={TH}>馬番</th>
              <th style={{ ...TH, textAlign: "left" }}>馬名</th>
              <th style={{ ...TH, textAlign: "left" }}>騎手</th>
              <th style={TH}>タイム</th>
              <th style={TH}>着差</th>
              <th style={TH}>上がり3F</th>
              {tb && <th style={TH} title="通過順の最初（序盤）と最後の角（4 角と書く）の番手 ÷ 頭数。トラックバイアスの前後は序盤の位置で決まる">位置</th>}
              <th style={TH}>人気</th>
              <th style={TH}>単勝</th>
              {hasMarks && <th style={{ ...TH, textAlign: "center" }} title="このレースで出しためぐAI印">めぐ印</th>}
            </tr>
          </thead>
          <tbody>
            {finish.map((f) => {
              const mk = (f.horse_number != null ? marks?.[f.horse_number] : undefined) ?? "";
              return (
                <tr key={`${f.pos}-${f.horse_number}`} style={{ background: f.pos <= 3 ? "rgba(251,191,36,0.05)" : undefined }}>
                  <td style={{ ...TD, fontWeight: 800, color: POS_COLOR[f.pos] }} data-sort={f.pos}>{f.pos}</td>
                  <td style={TD} data-sort={f.horse_number ?? 0}>{f.horse_number ?? "-"}</td>
                  <td style={{ ...TD, textAlign: "left", fontWeight: 700 }}>{f.horse_name}</td>
                  <td style={{ ...TD, textAlign: "left" }}>{f.jockey_name}</td>
                  <td style={TD}>{f.finish_time || "—"}</td>
                  <td style={TD}>{f.margin || "—"}</td>
                  <td style={TD} data-sort={f.last_3f ?? 999}>{f.last_3f != null ? f.last_3f.toFixed(1) : "—"}</td>
                  {tb && <td style={{ ...TD, fontSize: 11.5, color: "var(--text-dim)" }} data-sort={earlySortKey(f.passing_order)}>
                    <span className="m-hide">{positionText(f.passing_order, field)}</span><span className="m-only">{positionShort(f.passing_order, field)}</span></td>}
                  <td style={TD} data-sort={f.popularity ?? 999}>{f.popularity ?? "—"}</td>
                  <td style={TD} data-sort={f.odds ?? 0}>{f.odds != null ? `${f.odds.toFixed(1)}倍` : "—"}</td>
                  {hasMarks && <td style={{ ...TD, textAlign: "center", fontWeight: 800, color: "#fbbf24" }} data-sort={MARK_ORDER[mk] ?? MARK_ORDER[""]} title={mk ? "めぐAI印" : undefined}>{mk}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h4 style={{ margin: "16px 0 6px", fontSize: 13, color: "var(--text)" }}>払戻（100円あたり）</h4>
      {payoff.length === 0 ? (
        <div style={{ padding: 12, border: "1px dashed var(--border)", borderRadius: 8, color: "var(--text-dim)", fontSize: 12.5 }}>
          払戻はまだ取得できていません（確定の結果が公開されると表示します）。
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", background: "var(--surface)" }} data-no-sort>
            <thead style={{ background: "var(--surface2)" }}>
              <tr>
                <th style={{ ...TH, textAlign: "left" }}>券種</th>
                <th style={{ ...TH, textAlign: "left" }}>組合せ</th>
                <th style={TH}>払戻</th>
                <th style={TH}>人気</th>
              </tr>
            </thead>
            <tbody>
              {payoff.flatMap((g) =>
                g.rows.map((r, i) => (
                  <tr key={`${g.bet_label}-${r.numbers}-${i}`}>
                    <td style={{ ...TD, textAlign: "left", fontWeight: 700 }}>{i === 0 ? g.bet_label : ""}</td>
                    <td style={{ ...TD, textAlign: "left" }}>{r.numbers}</td>
                    <td style={{ ...TD, fontWeight: 700 }}>{yen(r.payout)}</td>
                    <td style={TD}>{r.popularity != null ? `${r.popularity}人気` : "—"}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
