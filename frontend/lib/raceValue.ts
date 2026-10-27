"use client";

import { useEffect, useState } from "react";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockBetRecommendation } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { getJson, prefetchJson, readStored } from "@/lib/httpCache";

/** レースの「価値」= 推奨買い目の期待回収率（バックエンド delegates.race_value_summary と同じ区分） */
export type ValueLevel = "hot" | "good" | "ok" | "pass" | "none";

export type RaceValue = {
  race_id: string;
  level: ValueLevel;
  expected_roi?: number | null;
  /** 会員向け（有料ゾーン）: 期待回収率の数値の代わりに、レース単位の推奨度の段階が来る */
  tier?: { level: number; label: string } | null;
  /** "target" = 既定方策（目標払戻つきの 3連複）。推奨度は『どれかが当たる確率の大きさ』で、回収率の高さではない */
  basis?: string;
  max_ev?: number | null;
  n_bets?: number;
  total_bet?: number;
  rank?: number | null;
  top_bet?: { bet_type: string; label: string; ev: number } | null;
  reason?: string;
  /** 確率を市場補正（実オッズあり）したレース。補正したレースとしないレースは基準（段の閾値・順位）が別 */
  prob_calibrated?: boolean;
};

export const LEVEL_STYLE: Record<ValueLevel, { label: string; hint: string; color: string; bg: string; border: string; glow?: string }> = {
  hot: { label: "★ 相対的に高い", hint: "較正後の期待回収率が高水準で、その日の上位 20%（目安。年によって実現回収率は大きく動く）", color: "#fff", bg: "linear-gradient(135deg,#f97316,#ef4444)", border: "#fb923c", glow: "0 0 10px rgba(249,115,22,.65)" },
  good: { label: "やや高い", hint: "較正後の期待回収率が高め（目安）", color: "#bbf7d0", bg: "rgba(34,197,94,0.22)", border: "#22c55e" },
  ok: { label: "平均並み", hint: "較正後の期待回収率が損益分岐の近く（目安）", color: "#bfdbfe", bg: "rgba(96,165,250,0.14)", border: "#60a5fa" },
  pass: { label: "見送り水準", hint: "較正後の期待回収率が基準未満、または買い目なし", color: "#8b9bb4", bg: "transparent", border: "#35445f" },
  none: { label: "未計算", hint: "予測またはオッズが無い", color: "#8b9bb4", bg: "transparent", border: "#35445f" },
};

/** 有料会員向けの5段階の推奨度の凡例（API の tier.label と同じ並び）。表示の色は4段階の色分けに対応させる */
export const TIER_LEGEND: { label: string; level: ValueLevel }[] = [
  { label: "最注目", level: "hot" }, { label: "高注目", level: "good" }, { label: "注目", level: "ok" }, { label: "標準", level: "ok" }, { label: "見送り", level: "pass" },
];

export const TARGET_TIER_HINT: Record<string, string> = {
  "最注目": "選んだ買い目のどれかが当たる確率が高い（それでも約10〜15%）", "高注目": "どれかが当たる確率がやや高い", "注目": "どれかが当たる確率は標準",
  "標準": "どれかが当たる確率は低め", "見送り": "条件を満たす買い目がない",
};

export function isTargetValues(values: Map<string, RaceValue>): boolean {
  for (const v of values.values()) if (v.basis === "target") return true;
  return false;
}

export function hasTier(values: Map<string, RaceValue>): boolean {
  for (const v of values.values()) if (v.tier?.label) return true;
  return false;
}

/** チップの2行目の文言: 5段階の推奨度があればそのラベル、無ければ期待回収率（開発者）、見送りは「見送」 */
export function chipLabel(v: RaceValue | undefined): string {
  if (v?.tier?.label) return v.tier.label;
  const lv = v?.level ?? "none";
  if (lv === "none" && v?.reason === "予測待ち") return "予測前";
  return v?.expected_roi != null && lv !== "pass" ? `${(v.expected_roi * 100).toFixed(0)}%` : lv === "pass" ? "見送" : "–";
}

const BET_JA: Record<string, string> = { tansho: "単勝", fukusho: "複勝", umaren: "馬連", wide: "ワイド", umatan: "馬単", sanrenpuku: "3連複" };

export function valueTitle(v: RaceValue | undefined): string {
  if (!v) return "";
  const s = LEVEL_STYLE[v.level];
  if (v.level === "none") return `${s.label}${v.reason ? `（${v.reason}）` : ""}`;
  if (v.tier) return `推奨度: ${v.tier.label}${v.n_bets ? `｜買い目 ${v.n_bets} 点` : ""}`;
  const roi = v.expected_roi != null ? `推奨の期待回収率（較正後） ${(v.expected_roi * 100).toFixed(0)}%` : "買い目なし";
  const top = v.top_bet ? ` ／ 最有力 ${BET_JA[v.top_bet.bet_type] ?? v.top_bet.bet_type} ${v.top_bet.label}（期待値 ${v.top_bet.ev.toFixed(2)}）` : "";
  const rank = v.rank ? ` ／ その日 ${v.rank} 位${v.prob_calibrated ? "（市場補正したレースの中）" : ""}` : "";
  const pc = v.prob_calibrated ? " ／ 確率は市場補正後" : "";
  return `${s.label}｜${roi}${top}${rank}${pc}`;
}

function levelOf(roi: number | null, n: number): ValueLevel {
  if (!n || roi == null) return "pass";
  return roi >= 1.5 ? "hot" : roi >= 1.25 ? "good" : roi >= 1.05 ? "ok" : "pass";
}

function mockValues(): RaceValue[] {
  const rows: RaceValue[] = MOCK_WEEKLY_RACES.map((r) => {
    const rec = getMockBetRecommendation(r.race_id, 100000, "auto", "latest", 2000);
    const roi = rec.total_bet > 0 ? rec.expected_return / rec.total_bet : null;
    const top = [...rec.candidates].sort((a, b) => b.ev - a.ev)[0];
    return {
      race_id: r.race_id, level: levelOf(roi, rec.candidates.length), expected_roi: roi, n_bets: rec.candidates.length, total_bet: rec.total_bet,
      top_bet: top ? { bet_type: top.bet_type, label: top.pair_label, ev: top.ev } : null,
    };
  });
  const ranked = rows.filter((r) => r.expected_roi != null).sort((a, b) => (b.expected_roi ?? 0) - (a.expected_roi ?? 0));
  const cutoff = Math.max(1, Math.ceil(ranked.length * 0.2));
  ranked.forEach((r, i) => { r.rank = i + 1; if (r.level === "hot" && i + 1 > cutoff) r.level = "good"; });
  return rows;
}

const TTL_MS = 30_000;
const lastGood = new Map<string, Map<string, RaceValue>>();

export function raceValuesUrl(date: string): string {
  return `/api/v1/betting/race-values?date=${date}`;
}

function toMap(rows: RaceValue[]): Map<string, RaceValue> {
  return new Map(rows.map((r) => [r.race_id, r]));
}

/** 失敗（401・通信エラー）は空で上書きしない: 取得に失敗したら例外にし、呼ぶ側が前回の表示を残す。 */
async function fetchValues(date: string): Promise<Map<string, RaceValue>> {
  if (USE_MOCK) return toMap(mockValues());
  const doc = await getJson<{ races?: RaceValue[] }>(raceValuesUrl(date), { ttlMs: TTL_MS, credentials: "include", persist: true });
  const map = toMap(doc.races ?? []);
  lastGood.set(date, map);
  return map;
}

/** 先に取り始めておく（日付の確定や認証の確認を待たない）。ログイン前なら 401 で失敗するだけで何も残らない。 */
export function prefetchRaceValues(date: string): void {
  if (USE_MOCK || !date) return;
  prefetchJson(raceValuesUrl(date), { ttlMs: TTL_MS, credentials: "include", persist: true });
}

function initialValues(date: string): Map<string, RaceValue> | null {
  if (USE_MOCK || !date) return null;
  const g = lastGood.get(date);
  if (g) return g;
  const st = readStored<{ races?: RaceValue[] }>(raceValuesUrl(date));
  return st?.races ? toMap(st.races) : null;
}

/** 開催日（YYYYMMDD）の各レースの価値。前回の内容があれば先に出し、裏で更新する。認証の確認とは並行して取りに行く。 */
export function useRaceValues(date: string): Map<string, RaceValue> {
  const [map, setMap] = useState<Map<string, RaceValue>>(new Map());
  const { loading, isMember: loggedIn } = useAuthStatus();
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (!date && !USE_MOCK) return;
    if (!USE_MOCK && !loading && !loggedIn) { setMap(new Map()); return; }   // ログイン必須の API。未ログインと分かったら呼ばない
    const first = initialValues(date);
    if (first) setMap(first);
    const run = (attempt: number) => {
      fetchValues(date).then((m) => alive && setMap(m)).catch(() => {
        if (alive && attempt < 3) timer = setTimeout(() => run(attempt + 1), 1500 * (attempt + 1));   // 一時的な失敗は前回の表示を残して再取得
      });
    };
    run(0);
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [date, loading, loggedIn]);
  return map;
}

/** 市場補正したレースとしないレースが同じ日に混ざっているか（混ざっていると段の基準と順位が別になるので凡例に注記する） */
export function mixedProbCalibration(values: Map<string, RaceValue>): boolean {
  let pc = false;
  let plain = false;
  values.forEach((v) => { if (v.level === "none") return; if (v.prob_calibrated) pc = true; else plain = true; });
  return pc && plain;
}
