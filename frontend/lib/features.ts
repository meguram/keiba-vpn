/**
 * UI 機能フラグ（唯一の正）。
 *
 * 現在の UI のゴールは「AI 予測結果の表示 → 期待値計算 → 推奨馬券の提示」の最小構成。
 * それ以外の機能はコードを残したまま enabled:false で隠している（ホームのカード・ナビ・
 * 直リンクが対象。API は変更しない）。
 *
 * 一時的に有効化: NEXT_PUBLIC_ENABLE_FEATURES="all" または "megu-index,growth-curve"（カンマ区切り）
 * 恒久的に有効化: 下の enabled を true にする。
 */

export type FeatureDef = {
  label: string;
  enabled: boolean;
  /** このページ群に属するパス（完全一致またはその配下） */
  paths: string[];
};

export const FEATURES = {
  // ── 最小構成（有効） ──
  predictions: { label: "AI予測（レース一覧・予測表示）", enabled: true, paths: ["/weekly-predictions", "/races"] },
  "race-detail": { label: "レース詳細（予測・期待値・推奨馬券）", enabled: true, paths: ["/race"] },
  betting: { label: "馬券最適化", enabled: true, paths: ["/betting"] },
  auth: { label: "ログイン", enabled: true, paths: ["/login"] },

  // ── 無効化中（コードは残置） ──
  "megu-index": { label: "めぐ指数", enabled: false, paths: ["/megu-index"] },
  "pedigree-race-stats": { label: "血統傾向分析", enabled: false, paths: ["/pedigree-race-stats", "/note-aptitude-race"] },
  "tracking-difficulty": { label: "追走難度", enabled: false, paths: ["/tracking-difficulty"] },
  bloodline: { label: "血統研究・種牡馬メモ", enabled: false, paths: ["/bloodline", "/stallion-notes"] },
  "bloodline-vector": { label: "血統マップ", enabled: false, paths: ["/bloodline-vector"] },
  "data-analysis": { label: "詳細データ分析", enabled: false, paths: ["/data-analysis"] },
  "growth-curve": { label: "成長曲線", enabled: false, paths: ["/growth-curve"] },
  "track-speed": { label: "馬場速度", enabled: false, paths: ["/track-speed"] },
  myostatin: { label: "MSTN 遺伝子", enabled: false, paths: ["/myostatin"] },
  "betting-simulation": { label: "馬券シミュレーション", enabled: false, paths: ["/betting-simulation"] },
  "race-quality": { label: "レース質", enabled: false, paths: ["/race-quality"] },
  "ai-sla": { label: "AI SLA", enabled: false, paths: ["/ai-sla"] },
  "horse-detail": { label: "馬詳細ページ", enabled: false, paths: ["/horse"] },
} as const satisfies Record<string, FeatureDef>;

export type FeatureId = keyof typeof FEATURES;

const OVERRIDE = (process.env.NEXT_PUBLIC_ENABLE_FEATURES ?? "").trim();
const OVERRIDE_ALL = OVERRIDE === "all";
const OVERRIDE_IDS = new Set(OVERRIDE.split(",").map((s) => s.trim()).filter(Boolean));

export function featureEnabled(id: FeatureId): boolean {
  return FEATURES[id].enabled || OVERRIDE_ALL || OVERRIDE_IDS.has(id);
}

function matches(pathname: string, p: string): boolean {
  return pathname === p || pathname.startsWith(p + "/");
}

/** 無効化された機能に属するパスなら false。どの機能にも属さないパス（"/" や /api など）は true。 */
export function pathEnabled(pathname: string): boolean {
  for (const id of Object.keys(FEATURES) as FeatureId[]) {
    if (FEATURES[id].paths.some((p) => matches(pathname, p))) return featureEnabled(id);
  }
  return true;
}
