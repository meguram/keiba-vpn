"use client";

import { PRED_TTL_MS, getJson, readStored } from "@/lib/httpCache";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { USE_MOCK, getMockRaceDetail, getMockPredictions, getMockTdData, getMockMeguPredicted } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberLockCard, MemberLockNote, PublicModeNote } from "@/components/upgrade/MemberLock";
import { featureEnabled } from "@/lib/features";
import { PredictionSourceBadge } from "@/components/PredictionSourceBadge";
import ProbCalNote from "@/components/ProbCalNote";
import type { ProbCalibration } from "@/lib/probCalibration";
import { EntriesStatusBadge } from "@/components/EntriesStatus";
import type { EntriesStatus } from "@/lib/entriesStatus";
import { BetRecommend } from "@/components/BetRecommend";
import { PipelineStatus, type PipelineStatusData } from "@/components/PipelineStatus";
import { RaceNav } from "@/components/RaceNav";
import { bracketChipStyle, bracketOf as wakuOf } from "@/lib/bracket";
import { useNarrow } from "@/components/RaceStrip";
import { raceExists } from "@/lib/raceExists";
import { LoadingNotice } from "@/components/LoadingNotice";
import { useAutoRetry } from "@/lib/hooks/useAutoRetry";
import { firstCornerPos } from "@/lib/resultCheck";
import { MarkNote } from "@/components/MarkNote";
import { RaceLevelNote, TrackBiasNote } from "@/components/RaceResult";
import type { RaceLevel, RaceResultData, TrackBias } from "@/lib/hooks/useRaceResult";

const SHOW_MEGU = featureEnabled("megu-index");
const SHOW_TRACKING = featureEnabled("tracking-difficulty");
const SHOW_HORSE_LINK = featureEnabled("horse-detail");

function HorseName({ id, name }: { id?: string; name?: string }) {
  const label = name ?? "—";
  if (!id || !SHOW_HORSE_LINK) return <>{label}</>;
  return <a href={`/horse/${id}`} target="_blank" rel="noreferrer" style={{ color: "#fff", textDecoration: "none" }}>{label}</a>;
}

/* ── 型 ── */
type RaceEntry = {
  horse_number?: number | string;
  bracket_number?: number | string;
  horse_name?: string;
  horse_id?: string;
  sex_age?: string;
  sex?: string;
  age?: string;
  jockey_name?: string;
  jockey?: string;
  trainer_name?: string;
  trainer?: string;
  jockey_weight?: number | string;
  weight?: number | string;
  finish_position?: number | string;
  position?: number | string;
  time?: string;
  passing_order?: string;
  last3f?: number | string;
  win_odds?: number;
  /** スクレイピング実データの項目名（フロント内蔵モックは time / last3f / win_odds） */
  finish_time?: string;
  last_3f?: number | string;
  odds?: number;
  [k: string]: unknown;
};

type RaceOddsEntry = { horse_number?: number | string; odds?: number; win_odds?: number };

/** 正の数値だけを返す（スクレイピング直後の出馬表は odds=0.0 で入っていることがある） */
const positive = (v: unknown): number | undefined => (typeof v === "number" && v > 0 ? v : undefined);
type MeguHorseData = { megu_final: number | null; actual_megu: number | null };

/** 枠番バッジ（枠色の背景に枠番）。馬番は NumBadge で同じ枠色に塗る */
function BracketBadge({ bn }: { bn: number | null | undefined }) {
  if (!bn) return <span style={{ color: "var(--text-dim)" }}>—</span>;
  return <span title={`${bn}枠`} style={bracketChipStyle(bn, 24)}>{bn}</span>;
}

function NumBadge({ no, bn }: { no: number | string | null | undefined; bn: number | null | undefined }) {
  if (no == null || no === "") return <span style={{ color: "var(--text-dim)" }}>—</span>;
  return <span title={bn ? `${bn}枠 ${no}番` : `${no}番`} style={bracketChipStyle(bn, 24)}>{no}</span>;
}

function indexMeguHorses(
  horses: { horse_id?: string; horse_number?: number | null; megu_final?: number | null; actual_megu?: number | null }[],
): { byId: Record<string, MeguHorseData>; byNumber: Record<string, MeguHorseData> } {
  const byId: Record<string, MeguHorseData> = {};
  const byNumber: Record<string, MeguHorseData> = {};
  for (const h of horses) {
    const row: MeguHorseData = {
      megu_final: h.megu_final ?? null,
      actual_megu: h.actual_megu ?? null,
    };
    if (h.horse_id) byId[String(h.horse_id)] = row;
    if (h.horse_number != null) byNumber[String(h.horse_number)] = row;
  }
  return { byId, byNumber };
}

function lookupMegu(
  entry: RaceEntry,
  byId: Record<string, MeguHorseData>,
  byNumber: Record<string, MeguHorseData>,
): MeguHorseData | null {
  const horseId = entry.horse_id != null ? String(entry.horse_id) : "";
  if (horseId && byId[horseId]) return byId[horseId];
  const num = entry.horse_number ?? entry.number;
  if (num != null && byNumber[String(num)]) return byNumber[String(num)];
  return null;
}
type RaceData = {
  race_id?: string;
  race_name?: string;
  venue?: string;
  round?: number | string;
  date?: string;
  surface?: string;
  distance?: number | string;
  direction?: string;
  weather?: string;
  track_condition?: string;
  start_time?: string;
  grade?: string;
  race_shutuba?: { entries?: RaceEntry[] };
  race_result?: { entries?: RaceEntry[]; payoffs?: unknown[]; pace?: { first_half_3f?: number | null; second_half_3f?: number | null } };
  race_odds?: { win?: RaceOddsEntry[]; entries?: RaceOddsEntry[] };
  /** 匿名・無料会員の応答。result_confirmed=確定の結果（確定オッズ・人気・払戻はこの時だけ来る） */
  access?: { tier?: string; result_confirmed?: boolean };
  [k: string]: unknown;
};

const MARK_SYMBOL: Record<string, string> = { honmei: "◎", pair: "○", show_val: "▲", anchor: "△", star: "★", kesi: "消" };
const markOf = (p?: { mark?: string; mark_type?: string } | null) => (p?.mark || MARK_SYMBOL[p?.mark_type ?? ""] || "");

type PredEntry = {
  bracket_number?: number | string | null;
  horse_number?: number;
  horse_name?: string;
  horse_id?: string;
  recommendation?: string;
  /** 印の語（honmei/pair/show_val/anchor/star/kesi/none）と記号。匿名・無料会員には予測の数値の代わりにこれだけが来る */
  mark_type?: string;
  mark?: string;
  pred_rank?: number;
  composite_rank?: number;
  pred_score?: number | null;
  composite_score?: number | null;
  win_prob?: number | null;
  top2_prob?: number | null;
  top3_prob?: number | null;
  win_prob_model?: number | null;
  top2_prob_model?: number | null;
  top3_prob_model?: number | null;
  win_odds?: number | null;
  place_odds_min?: number | null;
  place_odds_max?: number | null;
  redacted?: boolean;
  [k: string]: unknown;
};

type PredData = {
  access?: { public_mode?: boolean; withheld?: string[] };
  status?: string;
  has_prediction?: boolean;
  /** "bets" = 印をサンプル買い目（買い目）から決めている */
  model_type?: string;
  model_description?: string;
  prediction_source?: string;
  prediction_note?: string;
  total_horses?: number;
  elapsed_sec?: number;
  predictions?: PredEntry[];
  feature_highlights?: { label?: string; direction?: string; magnitude?: string }[];
  bet_suggestion?: unknown;
  prob_calibration?: ProbCalibration | null;
  /** 出馬表の公示の段階（特別登録／出走馬決定／出馬表確定／馬体重発表済み）と最終更新 */
  entries_status?: EntriesStatus | null;
  /** 予測パイプライン（出馬表→馬体重→AI予測→オッズ→サンプル買い目）の進み具合 */
  pipeline_status?: PipelineStatusData | null;
  provisional_entries?: boolean;
  [k: string]: unknown;
};

/* ── ユーティリティ ── */
function pick(e: RaceEntry, ...keys: string[]): string {
  for (const k of keys) {
    const v = (e as Record<string, unknown>)[k];
    if (v !== undefined && v !== null && v !== "") return String(v);
  }
  return "";
}

function fmtOdds(v: number | null | undefined): { text: string; cls: string } {
  if (v == null) return { text: "—", cls: "" };
  const n = Number(v);
  if (n < 5) return { text: n.toFixed(1), cls: "text-red-400" };
  if (n < 15) return { text: n.toFixed(1), cls: "text-yellow-400" };
  return { text: n.toFixed(1), cls: "text-gray-400" };
}

function gradeStyle(grade: string | undefined): React.CSSProperties {
  if (!grade) return {};
  if (grade === "G1") return { background: "rgba(239,68,68,0.2)", color: "#ef4444" };
  if (grade === "G2") return { background: "rgba(59,130,246,0.2)", color: "#60a5fa" };
  if (grade === "G3") return { background: "rgba(34,197,94,0.2)", color: "#22c55e" };
  return { background: "rgba(245,158,11,0.2)", color: "var(--warn)" };
}

/* ── タブ ── */
type Tab = "shutuba" | "result" | "predict" | "bet" | "horses";

const TABS: { id: Tab; label: string }[] = [
  { id: "predict", label: "🤖 AI予測" },
  { id: "bet", label: "💴 サンプル買い目" },
  { id: "shutuba", label: "出馬表" },
  { id: "result", label: "レース結果" },
  ...(SHOW_TRACKING ? [{ id: "horses" as Tab, label: "出走馬詳細" }] : []),
];

const TH: React.CSSProperties = {
  padding: "10px 12px", fontSize: 12, fontWeight: 600, color: "var(--text-dim)",
  textAlign: "center", borderBottom: "1px solid var(--border)", whiteSpace: "nowrap",
};
const TD: React.CSSProperties = {
  padding: "10px 12px", borderBottom: "1px solid rgba(36,48,73,0.5)",
  textAlign: "center", verticalAlign: "middle", whiteSpace: "nowrap",
};

/* ── めぐ指数（モック） ── */
/** 時刻文字列 "1:23.4" や "83.4" を秒に変換 */
function parseTimeSec(t: string | undefined | null): number | null {
  if (!t) return null;
  const s = String(t).trim();
  const m = s.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (m) return parseInt(m[1]) * 60 + parseFloat(m[2]);
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

/**
 * めぐ指数（モック）
 * 走破タイムとレース内最速タイムの差から暫定的な指数を算出。
 * ※ 正式なロジックは後日置き換え予定
 */
function calcMeguIndex(timeSec: number | null, bestSec: number | null, fieldSize: number): number | null {
  if (timeSec == null || bestSec == null || timeSec <= 0) return null;
  const diff = timeSec - bestSec; // 秒差（0=1着タイム）
  // ベース: 勝ちタイム=100, 頭数・距離にかかわらず1秒差≒-4pt（暫定）
  const raw = 100 - diff * 4.0;
  // フィールドサイズが大きいほど上位の評価を引き上げる（暫定補正）
  const fieldBonus = diff < 0.3 ? (fieldSize - 8) * 0.3 : 0;
  return Math.max(50, Math.min(120, Math.round((raw + fieldBonus) * 10) / 10));
}

function meguColor(v: number | null): { color: string; bg: string } {
  if (v == null) return { color: "var(--text-dim)", bg: "transparent" };
  if (v >= 105) return { color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (v >= 98)  return { color: "#4ade80", bg: "rgba(74,222,128,0.07)" };
  if (v >= 92)  return { color: "#60a5fa", bg: "rgba(96,165,250,0.07)" };
  if (v >= 82)  return { color: "var(--text-dim)", bg: "transparent" };
  return { color: "#f87171", bg: "rgba(239,68,68,0.07)" };
}

/* ── コンポーネント ── */
export default function RaceDetailPage() {
  const { id: raceId } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>("predict");
  const [resSort, setResSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);      // 結果表の列ソート（見出しを押すと 昇順→降順→元の順）
  const [tabInitialized, setTabInitialized] = useState(false);
  const [raceData, setRaceData] = useState<RaceData | null>(null);
  const [predData, setPredData] = useState<PredData | null>(null);
  const [tdData, setTdData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  // 前回の内容を出したまま、最新の取得に失敗している（L1）。本体と予測は並行で取るので、印は別々に持つ
  const [staleBody, setStaleBody] = useState(false);
  const [stalePred, setStalePred] = useState(false);
  const staleNotice = staleBody || stalePred;
  const [notFound, setNotFound] = useState(false);
  const [navDate, setNavDate] = useState("");
  const [meguById, setMeguById] = useState<Record<string, MeguHorseData>>({});
  const [meguByNumber, setMeguByNumber] = useState<Record<string, MeguHorseData>>({});
  const [meguSource, setMeguSource] = useState<"db" | "mock">("mock");
  const { isMember, isAdmin, loading: authLoading } = useAuthStatus();
  const lock = !authLoading && !isMember;      // 匿名は勝率・複勝率まで（それ以外はサーバが落としている）
  const [raceLevel, setRaceLevel] = useState<RaceLevel | undefined>(undefined);   // レースレベル・開催日レベル・メンバーレベル（会員だけ。確定の結果の API）
  const [trackBias, setTrackBias] = useState<TrackBias | undefined>(undefined);   // トラックバイアス（会員だけ。同じ API）
  const narrow = useNarrow();

  const loadSeq = useRef(0);
  const tdRaceRef = useRef<string | null>(null);
  // いま本体を出しているレース。取り直しが失敗し続けて保存の期限（30 分）を過ぎても、出した内容を「取得中」・エラーで消さないため
  const shownRaceRef = useRef<string | null>(null);
  const loadPrediction = useCallback(async (seq?: number) => {
    if (!raceId) return;
    if (USE_MOCK) {
      setPredData(getMockPredictions(raceId) as PredData);
      return;
    }
    const url = `/api/race/${raceId}/predictions`;
    const current = () => seq === undefined || seq === loadSeq.current;
    // 前回このタブで見た予測があれば先に出し、最新を取れたら差し替える（L1。週の予想の画面と同じ扱い）
    const stored = readStored<PredData>(url);
    if (stored && current()) setPredData((p) => p ?? stored);
    try {
      const d = await getJson<PredData>(url, { ttlMs: PRED_TTL_MS, persist: true });
      if (current()) { setPredData(d); setStalePred(false); }
    } catch {
      if (stored && current()) setStalePred(true);   // 前回の予測を出したまま最新が取れない（オッズが古いままになりうる）ことを出す。取り直しは下の useAutoRetry（予測だけ）
    }
  }, [raceId]);
  const loadRace = useCallback(async () => {
    if (!raceId) return;
    const shown = shownRaceRef.current === raceId;
    if (!shown) setLoading(true);
    setLoadError("");
    setNotFound(false);
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 300));
      setRaceData(getMockRaceDetail(raceId) as RaceData);
      setTdData(getMockTdData(raceId) as unknown as Record<string, unknown>);
      setLoading(false);
      return;
    }
    // レースを素早く切り替えたとき、遅れて届いた古い応答が新しい画面を上書きしないようにする
    const seq = ++loadSeq.current;
    const fresh = () => seq === loadSeq.current;
    // 前回このタブで見た本体があれば先に出して（取得中を出さない）、最新を取れたら差し替える（L1）
    const raceUrl = `/api/race/${raceId}`;
    const storedRace = readStored<RaceData>(raceUrl);
    if (storedRace && raceExists(storedRace)) { setRaceData(storedRace); setLoading(false); shownRaceRef.current = raceId; }
    // 別のレースに移ったときだけ、予測・走りやすさ・印を消す（同じレースの取り直しでは一度消さない＝ちらつかせない）
    if (tdRaceRef.current !== raceId) { setPredData(null); setTdData(null); setStaleBody(false); setStalePred(false); tdRaceRef.current = raceId; }
    try {
      // 本体・予測・走りやすさを同時に取りに行く（予測は本体の後ではなく並行。重い走りやすさは本体を待たせない）
      const racePromise = getJson<RaceData>(raceUrl, { ttlMs: 20_000, persist: true });
      loadPrediction(seq);
      if (SHOW_TRACKING) {
        getJson<Record<string, unknown>>(`/api/race/${raceId}/tracking-difficulty`, { ttlMs: 60_000 })
          .then((td) => { if (fresh() && !td.error) setTdData(td); })
          .catch(() => { /* 走りやすさが無くても画面は出す */ });
      }
      let rd: RaceData;
      try {
        rd = await racePromise;
      } catch (e) {
        if (!fresh()) return;
        if ((e as { status?: number }).status === 404) { setNotFound(true); return; }
        if ((storedRace && raceExists(storedRace)) || shown) { setStaleBody(true); return; }   // 最新が取れなくても、先に出した前回の内容は消さない（取れなかったことは出す）
        throw new Error("レースデータの取得に失敗しました。通信状況を確認して、もう一度お試しください。");
      }
      if (!fresh()) return;
      if (!raceExists(rd)) { setNotFound(true); return; }
      setRaceData(rd);
      shownRaceRef.current = raceId;
      setStaleBody(false);          // 印は最新が取れるまで残す（取り直しのたびに消さない）
    } catch (e: unknown) {
      if (fresh()) setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      if (fresh()) setLoading(false);
    }
  }, [raceId, loadPrediction]);



  useEffect(() => {
    loadRace();
  }, [loadRace]);

  // 予測パイプラインが途中（馬体重待ち・予測待ちなど）の間は 60 秒ごとに予測を取り直し、完了表示に切り替える
  const pipelineOpen = !!predData?.pipeline_status && !predData.pipeline_status.complete && !predData.pipeline_status.race_over;
  useEffect(() => {
    if (!pipelineOpen || USE_MOCK) return;
    const t = setInterval(() => { if (!document.hidden) loadPrediction(); }, 60_000);
    return () => clearInterval(t);
  }, [pipelineOpen, loadPrediction]);

  // レース移動ナビ用の開催日（YYYYMMDD）。読み込み中も直前の値を保つのでナビは消えない
  useEffect(() => {
    const d = (raceData?.date ?? "").replace(/-/g, "");
    if (/^\d{8}$/.test(d)) setNavDate(d);
  }, [raceData]);

  useEffect(() => {
    if (!raceData || tabInitialized) return;
    setTabInitialized(true);
  }, [raceData, tabInitialized]);

  // レースレベル・開催日レベル・メンバーレベル（会員だけ。確定の結果の API が事前に計算した値を返す）
  const hasResult = (raceData?.race_result?.entries?.length ?? 0) > 0;
  useEffect(() => {
    setRaceLevel(undefined);                                       // 別のレースへ移ったら前の値を消す
    setTrackBias(undefined);
    if (!raceId || !hasResult || lock || authLoading || USE_MOCK) return;
    let alive = true;
    getJson<RaceResultData>(`/api/race/${raceId}/result`, { ttlMs: PRED_TTL_MS })
      .then((d) => { if (alive && d?.status === "final") { setRaceLevel(d.race_level); setTrackBias(d.track_bias); } })
      .catch(() => { /* 取れなければ出さない */ });
    return () => { alive = false; };
  }, [raceId, hasResult, lock, authLoading]);

  // めぐ指数（実測。レース結果の表に出す）を API から取得
  useEffect(() => {
    if (!raceId || !SHOW_MEGU) return;
    if (USE_MOCK) {
      const mock = getMockMeguPredicted(raceId) as { horses?: { horse_id: string; horse_number?: number | null; megu_final?: number | null; actual_megu?: number | null }[] };
      const { byId, byNumber } = indexMeguHorses(mock.horses ?? []);
      setMeguById(byId);
      setMeguByNumber(byNumber);
      setMeguSource("mock");
      return;
    }
    fetch(`/api/v1/races/${raceId}/megu-index-predicted`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.horses?.length) {
          const { byId, byNumber } = indexMeguHorses(data.horses);
          setMeguById(byId);
          setMeguByNumber(byNumber);
          setMeguSource("db");
        }
      })
      .catch(() => { /* APIなければモックフォールバック */ });
  }, [raceId]);

  // 自動の取り直しは本体の失敗だけ（予測だけの失敗で本体・走りやすさまで 8 秒ごとに取り直さない）
  useAutoRetry(!!loadError || staleBody, () => { void loadRace(); }, 8000);
  // 予測だけが古いときは、予測だけを取り直す（完了したレースでは 60 秒ごとの予測の更新が動かないため）
  useAutoRetry(stalePred && !staleBody && !loadError, () => { void loadPrediction(); }, 8000);

  /* ── ローディング ── */
  if (loading) {
    return (
      <div style={{ padding: 24 }}>
        {navDate && <RaceNav raceId={raceId} date={navDate} />}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <LoadingNotice what="レースデータ" size={40} onRetry={() => { void loadRace(); }} />
      </div>
      </div>
    );
  }

  if (notFound) {
    return (
      <div style={{ padding: 24, textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>🔍</div>
        <h1 style={{ fontSize: 18, fontWeight: 700, color: "#f0f6fc", marginBottom: 8 }}>レースが見つかりません</h1>
        <p style={{ color: "var(--text-dim)", fontSize: 13, marginBottom: 16 }}>レースID「{raceId}」のデータはありません。URL をご確認ください。</p>
        <div style={{ display: "flex", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
          <Link href="/races" style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid var(--accent)", color: "var(--accent)", textDecoration: "none", fontSize: 14 }}>レース一覧へ</Link>
          <Link href="/" style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid var(--border)", color: "var(--text-dim)", textDecoration: "none", fontSize: 14 }}>ホームへ</Link>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div role="alert" style={{ padding: 24, textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
        <p style={{ color: "var(--err)", marginBottom: 14 }}>{loadError}</p>
        <div style={{ display: "flex", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ padding: "10px 0", color: "var(--text-dim)", fontSize: 14 }}>自動で取り直しています…</span>
          <Link href="/races" style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid var(--border)", color: "var(--text-dim)", textDecoration: "none", fontSize: 14 }}>レース一覧へ</Link>
        </div>
      </div>
    );
  }

  const rd = raceData;

  /* ── レースヘッダー ── */
  const headChips = rd ? [narrow ? rd.date?.slice(5).replace("-", "/") : rd.date, rd.surface && `${narrow ? "" : "馬場: "}${rd.surface}`, rd.distance && `${rd.distance}m`, rd.direction, rd.weather && `${narrow ? "" : "☁ "}${rd.weather}`, rd.track_condition && `${narrow ? "" : "馬場: "}${rd.track_condition}`, rd.start_time && `${narrow ? "" : "🕐 "}${rd.start_time}`].filter(Boolean) : [];
  const Header = rd ? (narrow ? (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 10px", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 13, padding: "3px 8px", borderRadius: 5, whiteSpace: "nowrap", flex: "none" }}>{rd.venue} {rd.round}R</span>
        {rd.grade && rd.grade !== "未勝利" && rd.grade !== "新馬" && (
          <span style={{ fontWeight: 700, fontSize: 12, padding: "2px 7px", borderRadius: 4, whiteSpace: "nowrap", flex: "none", ...gradeStyle(rd.grade) }}>{rd.grade}</span>
        )}
        <h1 style={{ fontSize: 17, fontWeight: 700, color: "#fff", margin: 0, flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{rd.race_name ?? raceId}</h1>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, overflowX: "auto", whiteSpace: "nowrap", WebkitOverflowScrolling: "touch" }}>
        {predData?.entries_status && <span style={{ flex: "none" }}><EntriesStatusBadge status={predData.entries_status} /></span>}
        {headChips.map((t, i) => (
          <span key={i} style={{ flex: "none", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 7px", fontSize: 12, color: "var(--text-dim)" }}>{t}</span>
        ))}
      </div>
    </div>
  ) : (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "24px 32px", marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <span style={{ background: "var(--accent)", color: "#fff", fontWeight: 800, fontSize: 16, padding: "6px 14px", borderRadius: 6, minWidth: 50, textAlign: "center" }}>
          {rd.venue} {rd.round}R
        </span>
        {rd.grade && rd.grade !== "未勝利" && rd.grade !== "新馬" && (
          <span style={{ fontWeight: 700, fontSize: 13, padding: "4px 12px", borderRadius: 4, ...gradeStyle(rd.grade) }}>{rd.grade}</span>
        )}
      </div>
      <h1 style={{ fontSize: narrow ? 20 : 26, fontWeight: 700, color: "#fff", marginBottom: 8, overflowWrap: "anywhere" }}>{rd.race_name ?? raceId}</h1>
      {predData?.entries_status && <div style={{ margin: "0 0 10px" }}><EntriesStatusBadge status={predData.entries_status} /></div>}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, fontSize: 13, color: "var(--text-dim)" }}>
        {[rd.date, rd.surface && `馬場: ${rd.surface}`, rd.distance && `${rd.distance}m`, rd.direction, rd.weather && `☁ ${rd.weather}`, rd.track_condition && `馬場: ${rd.track_condition}`, rd.start_time && `🕐 ${rd.start_time}`]
          .filter(Boolean)
          .map((t, i) => (
            <span key={i} style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 4, padding: "3px 10px", fontSize: 12 }}>{t}</span>
        ))}
      </div>
    </div>
  )) : null;

  /* ── 出馬表 ── */
  const shutubaEntries = [...(rd?.race_shutuba?.entries ?? [])].sort((a, b) => (Number(a.horse_number) || 99) - (Number(b.horse_number) || 99));

  /* ── レース結果 ── */
  // 着順が数字で取れない馬（取消・除外・中止など）は一番下へ。数字の着順は小さい順、同じ並びのものは馬番順
  const finishKey = (e: { finish_position?: number | string; position?: number | string }) => {
    const n = Number(e.finish_position ?? e.position);
    return Number.isFinite(n) && n > 0 ? n : 10_000;
  };
  // 確定オッズ・人気は確定の結果だけ（匿名・無料はサーバの判定 access.result_confirmed。会員は値がある時）。確定前のオッズは出さない
  const showFinalOdds = (rd?.access ? !!rd.access.result_confirmed : rd?.race_result != null && (rd.race_result as { _display_source?: string })._display_source === "race_result")
    && (rd?.race_result?.entries ?? []).some((e) => positive(e.odds) != null);
  const resultEntries = [...(rd?.race_result?.entries ?? [])].sort((a, b) => finishKey(a) - finishKey(b) || (Number(a.horse_number) || 99) - (Number(b.horse_number) || 99));
  const allTimes = resultEntries.map((e) => parseTimeSec((e.time ?? e.finish_time) as string | undefined)).filter((v): v is number => v != null && v > 0);
  const bestTime = allTimes.length ? Math.min(...allTimes) : null;
  const fieldSize = resultEntries.length;

  function getMeguValues(entry: RaceEntry): MeguHorseData {
    const fromApi = lookupMegu(entry, meguById, meguByNumber);
    if (fromApi) return fromApi;
    // APIデータなし: 実測値のみ走破タイムから推算。想定値は表示しない
    const tSec = parseTimeSec((entry.time ?? entry.finish_time) as string | undefined);
    const calcActual = calcMeguIndex(tSec, bestTime, fieldSize);
    return { megu_final: null, actual_megu: calcActual };
  }

  function MeguCell({ value, title, borderLeft = true }: { value: number | null; title: string; borderLeft?: boolean }) {
    const { color, bg } = meguColor(value);
    return (
      <td
        style={{
          ...TD,
          borderLeft: borderLeft ? "2px solid rgba(167,139,250,0.15)" : undefined,
          fontWeight: value != null ? 700 : 400,
          color,
          background: bg,
        }}
        title={title}
      >
        {value != null ? value.toFixed(1) : "—"}
      </td>
    );
  }

  // 実オッズ（市場の単勝・人気）は画面に出さない。右の列は自前予測（勝率・複勝率・想定単勝）
  const predByNo = new Map<number, PredEntry>();
  (predData?.predictions ?? []).forEach((p) => { if (p.horse_number != null) predByNo.set(Number(p.horse_number), p); });
  const shutubaByNo = new Map<number, RaceEntry>();
  shutubaEntries.forEach((e) => { const n = Number(e.horse_number ?? e.number); if (n > 0) shutubaByNo.set(n, e); });
  const subLine = (e: RaceEntry) => [pick(e, "sex_age") || `${pick(e, "sex")}${pick(e, "age")}`, pick(e, "jockey_name", "jockey"), pick(e, "jockey_weight", "weight", "impost")].filter(Boolean).join(" ");
  const pctCell = (v: number | null | undefined, red?: boolean) => lock ? "🔒" : (red || v == null) ? "—" : `${(v * 100).toFixed(1)}%`;

  const ShutubaTbl = shutubaEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>📋 出馬表データがありません</div>
  ) : (
    <div style={{ overflowX: "auto" }}>
      {predData?.has_prediction === true && predByNo.size > 0 && <MarkNote />}
      <table className="m-sticky" style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 13 }}>
        <thead style={{ background: "var(--surface2)" }}>
          <tr>
            {[["枠", "m-hide"], ["馬番", ""], ["馬名", ""], ["性齢", "m-hide"], ["斤量", "m-hide"], ["騎手", "m-hide"], ["調教師", "m-hide"], ["勝率", ""], ["複勝率", ""], ["想定単勝", ""]].map(([h, c]) => <th key={h} className={c || undefined} style={TH}>{lock && ["勝率", "複勝率", "想定単勝"].includes(h) ? "🔒" : ""}{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {shutubaEntries.map((e, i) => {
            const rawNo = Number(e.horse_number ?? e.number);
            const hasNo = Number.isFinite(rawNo) && rawNo > 0;       // 馬番が未確定（特別登録・出走馬決定の段階）は 0 で来る → 『—』を出す
            const hn = hasNo ? String(rawNo) : "";
            const pr = hasNo ? predByNo.get(rawNo) : undefined;
            const { text: oddsText } = fmtOdds(pr?.win_odds ?? undefined);
            return (
              <tr key={hasNo ? hn : `p${i}-${e.horse_id ?? ""}`} style={{ transition: "background 0.12s" }}>
                <td className="m-hide" style={{ ...TD, textAlign: "center" }}><BracketBadge bn={hasNo ? wakuOf(Number(hn), shutubaEntries.length, Number(e.bracket_number) || null) : null} /></td>
                <td style={{ ...TD, textAlign: "center" }}><NumBadge no={hasNo ? hn : null} bn={hasNo ? wakuOf(Number(hn), shutubaEntries.length, Number(e.bracket_number) || null) : null} /></td>
                <td className="m-name" style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left" }}>
                  {markOf(pr) ? <span title="めぐAI印" style={{ marginRight: 4, color: "#fbbf24" }}>{markOf(pr)}</span> : null}<HorseName id={e.horse_id} name={e.horse_name} />
                  <span className="m-sub">{subLine(e) || "—"}</span>
                </td>
                <td className="m-hide" style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "sex_age") || `${pick(e, "sex")}${pick(e, "age")}` || "—"}</td>
                <td className="m-hide" style={TD}>{pick(e, "jockey_weight", "weight", "impost") || "—"}</td>
                <td className="m-hide" style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "jockey_name", "jockey") || "—"}</td>
                <td className="m-hide" style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "trainer_name", "trainer") || "—"}</td>
                <td style={TD}>{pctCell(pr?.win_prob, pr?.redacted)}</td>
                <td style={TD}>{pctCell(pr?.top3_prob, pr?.redacted)}</td>
                <td style={{ ...TD, color: "var(--text-dim)" }}>{lock || pr?.redacted || pr?.win_odds == null ? (lock ? "🔒" : "—") : oddsText}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  const resultCols: { key: string; label: string; val: (e: RaceEntry) => number | string | null }[] = [
    { key: "pos", label: "着順", val: (e) => finishKey(e) >= 10_000 ? null : finishKey(e) },
    { key: "waku", label: "枠", val: (e) => wakuOf(Number(e.horse_number ?? e.number), resultEntries.length, Number(e.bracket_number) || null) ?? null },
    { key: "no", label: "馬番", val: (e) => Number(e.horse_number ?? e.number) || null },
    { key: "name", label: "馬名", val: (e) => (e.horse_name ? String(e.horse_name) : null) },
    { key: "jockey", label: "騎手", val: (e) => (pick(e, "jockey_name", "jockey") || null) },
    { key: "time", label: "タイム", val: (e) => parseTimeSec((e.time ?? e.finish_time) as string | undefined) },
    { key: "passing", label: "通過順", val: (e) => firstCornerPos(e.passing_order) },
    { key: "l3f", label: "上り3F", val: (e) => { const n = Number(e.last3f ?? e.last_3f); return Number.isFinite(n) && n > 0 ? n : null; } },
    ...(showFinalOdds ? [
      { key: "pop", label: "人気", val: (e: RaceEntry) => { const n = Number(e.popularity); return Number.isFinite(n) && n > 0 ? n : null; } },
      { key: "fodds", label: "単勝", val: (e: RaceEntry) => positive(e.odds) ?? null },
    ] : []),
  ];
  const RES_HIDE = new Set(["waku", "jockey", "passing"]);   // 狭い画面では畳む列（騎手は馬名の下に出す）
  // 並べ替え: 値が取れない行（取消・データなし）は、昇順でも降順でも最後。同じ値は元の順（着順）のまま
  const sortCol = resSort ? resultCols.find((c) => c.key === resSort.key) : null;
  const sortedResult = sortCol && resSort ? [...resultEntries].sort((a, b) => {
    const va = sortCol.val(a), vb = sortCol.val(b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const c = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb), "ja");
    return c * resSort.dir;
  }) : resultEntries;
  const renderResultTbl = () => resultEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>🏆 レース結果データがありません</div>
  ) : (
    <div style={{ overflowX: "auto" }}>
      {!lock && <RaceLevelNote lv={raceLevel} />}
      {!lock && <TrackBiasNote tb={trackBias} />}
      <table data-no-sort style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 13 }}>
        <thead style={{ background: "var(--surface2)" }}>
          <tr>
            {resultCols.map((c) => (
              <th key={c.key} className={RES_HIDE.has(c.key) ? "m-hide" : undefined} style={{ ...TH, cursor: "pointer", userSelect: "none" }} aria-sort={resSort?.key === c.key ? (resSort.dir === 1 ? "ascending" : "descending") : "none"} title="押すと並べ替え（昇順→降順→元の順）"
                onClick={() => setResSort((cur) => (cur?.key !== c.key ? { key: c.key, dir: 1 } : cur.dir === 1 ? { key: c.key, dir: -1 } : null))}>
                {c.label}<span aria-hidden style={{ marginLeft: 3, fontSize: 9, opacity: resSort?.key === c.key ? 1 : 0.3 }}>{resSort?.key === c.key ? (resSort.dir === 1 ? "▲" : "▼") : "↕"}</span>
              </th>
            ))}
            {SHOW_MEGU && <th style={{ ...TH, borderLeft: "2px solid rgba(167,139,250,0.3)", color: "#a78bfa" }}>実測めぐ指数</th>}
          </tr>
        </thead>
        <tbody>
          {sortedResult.map((e, i) => {
            const pos = Number(e.finish_position ?? e.position ?? 99);
            const placeColor = pos === 1 ? "#fbbf24" : pos === 2 ? "#94a3b8" : pos === 3 ? "#d97706" : "var(--text)";
            const tSec = parseTimeSec((e.time ?? e.finish_time) as string | undefined);
            void tSec;
            const megu = getMeguValues(e);
            return (
              <tr key={i} style={{ transition: "background 0.12s" }}>
                <td style={{ ...TD, fontWeight: pos <= 3 ? 800 : 400, color: placeColor, fontSize: pos <= 3 ? 15 : 13 }}>{pos <= 0 || pos >= 99 ? "—" : pos}</td>
                <td className="m-hide" style={{ ...TD, textAlign: "center" }}><BracketBadge bn={wakuOf(Number(e.horse_number ?? e.number), resultEntries.length, Number(e.bracket_number) || null)} /></td>
                <td style={{ ...TD, textAlign: "center" }}><NumBadge no={e.horse_number ?? (e.number as number | string | undefined)} bn={wakuOf(Number(e.horse_number ?? e.number), resultEntries.length, Number(e.bracket_number) || null)} /></td>
                <td className="m-name" style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left" }}>
                  <HorseName id={e.horse_id} name={e.horse_name} />
                  <span className="m-sub">{[pick(e, "sex_age"), pick(e, "jockey_name", "jockey")].filter(Boolean).join(" ") || "—"}</span>
                </td>
                <td className="m-hide" style={{ ...TD, fontSize: 12, color: "var(--text-dim)" }}>{pick(e, "jockey_name", "jockey") || "—"}</td>
                <td style={TD}>{e.time ?? e.finish_time ?? "—"}</td>
                <td className="m-hide" style={{ ...TD, fontSize: 11 }}>{e.passing_order ?? "—"}</td>
                <td style={TD}>{e.last3f ?? e.last_3f ?? "—"}</td>
                {showFinalOdds && <td style={TD}>{positive(Number(e.popularity)) ?? "—"}</td>}
                {showFinalOdds && <td style={TD}>{positive(e.odds) != null ? `${positive(e.odds)!.toFixed(1)}倍` : "—"}</td>}
                {SHOW_MEGU && (
                  <MeguCell
                    value={megu.actual_megu}
                    title={meguSource === "db" ? "実測めぐ指数（レース後算出）" : "実測めぐ指数（β版モック）"}
                  />
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {SHOW_MEGU && (
      <div style={{ padding: "8px 12px", fontSize: 11, color: "var(--text-dim)", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ color: "#a78bfa", fontWeight: 600 }}>めぐ指数</span>
        {meguSource === "db"
          ? <span>レース後の走破タイムから算出（1点≒0.1秒）。</span>
          : <span>β版モック算出（API 未取得時のフォールバック）。</span>
        }
        <span style={{ borderLeft: "1px solid var(--border)", paddingLeft: 10 }}>
          <strong style={{ color: "#22c55e" }}>≥105</strong> 優秀 /
          <strong style={{ color: "#60a5fa", marginLeft: 4 }}>98〜</strong> 良好 /
          <strong style={{ color: "var(--text-dim)", marginLeft: 4 }}>82〜</strong> 標準 /
          <strong style={{ color: "#f87171", marginLeft: 4 }}>＜82</strong> 低調
        </span>
        {meguSource === "db" && <span style={{ color: "#a78bfa", fontSize: 10, background: "rgba(167,139,250,0.15)", borderRadius: 4, padding: "1px 6px" }}>実データ</span>}
        {meguSource === "mock" && <span style={{ color: "#94a3b8", fontSize: 10, background: "rgba(148,163,184,0.15)", borderRadius: 4, padding: "1px 6px" }}>β版</span>}
      </div>
      )}
    </div>
  );

  /* ── AI予測 ── */
  const preds = predData?.predictions ?? [];
  // 匿名・無料会員の応答には確率が無く印だけが来るので、印でも「予測あり」とみなす
  const hasPred = predData?.has_prediction === true && preds.some((p) => p.pred_score != null || p.composite_score != null || p.win_prob != null || !!p.mark_type);
  const sortedPreds = hasPred
    ? [...preds].sort((a, b) => (a.composite_rank ?? a.pred_rank ?? 99) - (b.composite_rank ?? b.pred_rank ?? 99))
    : preds;

  const bracketOf = new Map<number, number>();
  shutubaEntries.forEach((e) => { const n = Number(e.horse_number), b = Number(e.bracket_number); if (n && b) bracketOf.set(n, b); });

  const PRED_HIDE = new Set([0, 4, 7]);   // 枠・連対率・想定複勝は狭い画面で畳む
  const PredPanel = (
    <div>
      {isAdmin && <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {predData?.model_description && (
          <span style={{ fontSize: 11, padding: "4px 10px", borderRadius: 4, background: "rgba(59,130,246,0.12)", color: "#60a5fa" }}>
            {hasPred ? "🧠" : "📊"} {predData.model_description}
          </span>
        )}
        <PredictionSourceBadge source={predData?.prediction_source} note={predData?.prediction_note} />
        {hasPred && predData?.elapsed_sec != null && (
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>⏱ {predData.elapsed_sec}秒 / {predData.total_horses}頭</span>
        )}
      </div>}

      {hasPred && lock && <MemberLockNote />}
      <PublicModeNote access={predData?.access} />
      {hasPred && <ProbCalNote pc={predData?.prob_calibration} preds={predData?.predictions} />}

      {!hasPred && (
        <div style={{ marginBottom: 12, fontSize: 13, color: "var(--text-dim)", padding: "10px 14px", borderRadius: 8, background: "rgba(36,48,73,0.4)" }}>
          {predData?.provisional_entries === true
            ? "出馬表は公示の途中です。AI 予測は、出走馬と枠順が確定した前日以降に表示します。"
            : isAdmin ? "予測結果はまだ算出されていません。" : "予測結果はまだありません。"}
            </div>
          )}

      {sortedPreds.length > 0 ? (
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
          <table className="m-sticky" style={{ width: "100%", borderCollapse: "collapse", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", fontSize: 12 }}>
            <thead style={{ background: "var(--surface2)" }}>
              <tr className="m-hide">
                <th style={{ ...TH, textAlign: "center" }} colSpan={3}>出走馬</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={3}>{predData?.prob_calibration?.applied ? "予測確率（市場補正後）" : "予測確率"}</th>
                <th style={{ ...TH, borderLeft: "1px solid var(--border)" }} colSpan={2}>想定オッズ</th>
              </tr>
              <tr>
                {(["枠","馬番","馬名","勝率","連対率","複勝率","単勝","複勝"] as string[]).map((h, i) => {
                  // 会員限定の列: 予測の列すべて（勝率・複勝率も。匿名・無料会員には印だけを出す。2026-11-25）
                  const isAiCol = i >= 3;
                  return (
                    <th
                      key={h}
                      className={PRED_HIDE.has(i) ? "m-hide" : undefined}
                      style={{
                        ...TH,
                        borderLeft: [3,6].includes(i) ? "1px solid rgba(36,48,73,0.5)" : undefined,
                        color: isAiCol && lock ? "var(--text-dim)" : TH.color,
                      }}
                    >
                      {isAiCol && lock ? <span style={{ marginRight: 3 }}>🔒</span> : null}{h === "単勝" ? <span className="m-only">想定</span> : null}{h}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {sortedPreds.map((p, i) => {
                const rank = p.composite_rank ?? p.pred_rank ?? 0;
                const rowBg = rank === 1 ? "rgba(239,68,68,0.04)" : rank === 2 ? "rgba(59,130,246,0.04)" : rank === 3 ? "rgba(34,197,94,0.04)" : undefined;
                return (
                  <tr key={p.horse_number ?? i} style={{ background: rowBg, transition: "background 0.12s" }}>
                    <td className="m-hide" style={{ ...TD, textAlign: "center" }}><BracketBadge bn={wakuOf(Number(p.horse_number), preds.length, bracketOf.get(Number(p.horse_number)) ?? (Number(p.bracket_number) || null))} /></td>
                    <td style={{ ...TD, textAlign: "center" }}><NumBadge no={p.horse_number} bn={wakuOf(Number(p.horse_number), preds.length, bracketOf.get(Number(p.horse_number)) ?? (Number(p.bracket_number) || null))} /></td>
                    <td className="m-name" style={{ ...TD, fontWeight: 600, color: "#fff", textAlign: "left", whiteSpace: "nowrap" }}>
                      {markOf(p) ? <span title="めぐAI印" style={{ marginRight: 4, color: "#fbbf24" }}>{markOf(p)}</span> : null}<HorseName id={p.horse_id} name={p.horse_name} />
                      <span className="m-sub">{[p.sex_age, p.jockey_name, p.jockey_weight].filter((x) => x != null && x !== "").join(" ") || (shutubaByNo.get(Number(p.horse_number)) ? subLine(shutubaByNo.get(Number(p.horse_number))!) : "") || "—"}</span>
                    </td>
                    {/* AI columns — redact when null or entry.redacted === true */}
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: (p.redacted || p.win_prob == null) ? "var(--text-dim)" : undefined }}>
                      {(p.redacted || p.win_prob == null) ? "●●%" : `${(p.win_prob * 100).toFixed(1)}%`}
                    </td>
                    <td className="m-hide" style={{ ...TD, color: (p.redacted || p.top2_prob == null) ? "var(--text-dim)" : undefined }}>
                      {(p.redacted || p.top2_prob == null) ? "●●%" : `${(p.top2_prob * 100).toFixed(1)}%`}
                    </td>
                    <td style={{ ...TD, color: (p.redacted || p.top3_prob == null) ? "var(--text-dim)" : undefined }}>
                      {(p.redacted || p.top3_prob == null) ? "●●%" : `${(p.top3_prob * 100).toFixed(1)}%`}
                    </td>
                    <td style={{ ...TD, borderLeft: "1px solid rgba(36,48,73,0.3)", color: "var(--text-dim)" }}>
                      {(p.redacted || p.win_odds == null) ? "●●倍" : `${Number(p.win_odds).toFixed(1)}倍`}
                    </td>
                    <td className="m-hide" style={{ ...TD, color: "var(--text-dim)", fontSize: 11 }}>
                      {(p.redacted || p.place_odds_min == null || p.place_odds_max == null)
                        ? "●●〜●●"
                        : `${Number(p.place_odds_min).toFixed(1)}〜${Number(p.place_odds_max).toFixed(1)}`}
                    </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ textAlign: "center", padding: 40, color: "var(--text-dim)" }}>出馬表がないため予測表を表示できません</div>
      )}
      {hasPred && <MarkNote />}

      {hasPred && predData?.feature_highlights?.length ? (
        <div style={{ marginTop: 16, padding: "12px 16px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-dim)", marginBottom: 8 }}>特徴量ハイライト</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {predData.feature_highlights.map((f, i) => (
              <span key={i} style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: f.direction === "positive" ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.12)", color: f.direction === "positive" ? "var(--ok)" : "var(--err)" }}>
                {f.label} {f.magnitude ? `(${f.magnitude})` : ""}
              </span>
            ))}
          </div>
        </div>
      ) : null}

    </div>
  );

  /* ── 出走馬詳細（追走難度との統合） ── */
  const tdEntries = (tdData as Record<string, unknown> & { entries?: unknown[] })?.entries ?? [];
  const HorseDetails = tdEntries.length === 0 ? (
    <div style={{ textAlign: "center", padding: 40 }}>
      <p style={{ color: "var(--text-dim)", marginBottom: 12 }}>追走難度データがありません</p>
      <a href={`/tracking-difficulty`} style={{ color: "var(--accent)", fontSize: 13 }}>
        → 追走難度分析ページで詳細を確認
      </a>
    </div>
  ) : (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))", gap: 12 }}>
      {(tdEntries as TdEntry[]).map((e: TdEntry) => {
        const td = (e.tracking_difficulty as { ease_pct?: number; ease_label?: string; flow_position?: string }) ?? {};
        const pct = td.ease_pct ?? 50;
        const color = pct >= 75 ? "var(--ok)" : pct >= 60 ? "var(--accent)" : pct >= 45 ? "var(--text-dim)" : pct >= 30 ? "var(--warn)" : "var(--err)";
        return (
          <div key={e.horse_id ?? e.horse_number} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontWeight: 700, color: "#fff", fontSize: 13, marginBottom: 8 }}>
              <span style={{ color: "#7dd3fc", marginRight: 6 }}>{e.horse_number}.</span>
              {e.horse_id ? <a href={`/horse/${e.horse_id}`} target="_blank" rel="noreferrer" style={{ color: "#fff", textDecoration: "none" }}>{e.horse_name}</a> : e.horse_name}
            </div>
            <div style={{ fontSize: 11, marginBottom: 6 }}>
              <span style={{ color: "var(--text-dim)" }}>追走容易度: </span>
              <span style={{ fontWeight: 700, color }}>{pct.toFixed(1)}%</span>
              {td.ease_label && <span style={{ marginLeft: 6, padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 600, background: color === "var(--ok)" ? "rgba(34,197,94,0.15)" : "rgba(107,125,149,0.12)", color }}>{td.ease_label}</span>}
            </div>
            <div style={{ height: 6, background: "rgba(36,48,73,0.5)", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
              <div style={{ height: "100%", width: `${Math.min(100, Math.max(2, pct))}%`, background: color, borderRadius: 3 }} />
            </div>
            {td.flow_position && (
              <div style={{ fontSize: 11, color: "var(--text-dim)" }}>想定位置: <span style={{ fontWeight: 600, color: "#7dd3fc" }}>{td.flow_position}</span></div>
            )}
          </div>
        );
      })}
    </div>
  );

  type TdEntry = {
    horse_number?: number;
    horse_id?: string;
    horse_name?: string;
    tracking_difficulty?: { ease_pct?: number; ease_label?: string; flow_position?: string };
  };

  /* ── レンダー ── */
  return (
    <div style={{ minHeight: "100dvh", background: "var(--bg)", color: "var(--text)", padding: narrow ? "0 2px" : 24 }}>
      {navDate && <RaceNav raceId={raceId} date={navDate} />}
      {staleNotice && (
        <div role="alert" style={{ color: "var(--err)", fontSize: 13, padding: "6px 0" }}>
          ⚠️ 最新のレース情報を取得できませんでした。前回の内容を表示しています
          <span style={{ marginLeft: 10, color: "var(--text-dim)" }}>自動で取り直しています…</span>
        </div>
      )}
      {Header}
      {!lock && <PipelineStatus status={predData?.pipeline_status} compact={narrow} />}

      {/* タブバー */}
      <div role="tablist" style={{ display: "flex", gap: 4, marginBottom: narrow ? 8 : 20, borderBottom: "1px solid var(--border)", paddingBottom: 0, flexWrap: "nowrap", overflowX: "auto" }}>
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            role="tab"
            aria-selected={tab === id}
            style={{
              padding: narrow ? "8px 12px" : "10px 20px",
              minHeight: narrow ? 40 : 44,
              whiteSpace: "nowrap",
              flex: "none",
              fontSize: 14,
              fontWeight: 500,
              color: tab === id ? "var(--accent)" : "var(--text-dim)",
              cursor: "pointer",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${tab === id ? "var(--accent)" : "transparent"}`,
              transition: "all 0.15s",
            } as React.CSSProperties}
          >
            {label}
          </button>
        ))}
      </div>

      {/* タブコンテンツ */}
      {tab === "shutuba" && ShutubaTbl}
      {tab === "result" && renderResultTbl()}
      {tab === "predict" && PredPanel}
      {tab === "bet" && (lock ? <MemberLockCard title="サンプル買い目" /> : <BetRecommend raceId={raceId} hasPrediction={hasPred} />)}
      {tab === "horses" && HorseDetails}
    </div>
  );
}
