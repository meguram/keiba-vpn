/** 確率の市場補正（KEIBA_PROB_CALIBRATION=1）の応答の型と文言。補正を適用しないレース（オッズ未確定・予測オッズ・8頭未満）でも理由が返る。 */
export type ProbCalibration = {
  applied: boolean;
  reason?: string | null;
  version?: number | null;
  odds_phase?: string | null;
  method?: Record<string, string> | null;
};

export type ProbModelFields = {
  horse_number?: number;
  horse_name?: string;
  win_prob?: number | null;
  top2_prob?: number | null;
  top3_prob?: number | null;
  win_prob_model?: number | null;
  top2_prob_model?: number | null;
  top3_prob_model?: number | null;
};

export const PROB_CAL_LABEL = "AI予測（市場補正後）";
export const PROB_CAL_EXPLAIN =
  "モデルの確率に、その時点の単勝オッズが示す市場の見方を組み合わせて、確率の外れ（特に人気馬の『来ない』の見積もり）を小さくした値です。期待値・印・推奨は、この補正後の確率で計算しています。元のモデルの確率は下の「モデルの確率を見る」で確認できます。オッズが出る前のレース・予測オッズのレース・8頭未満のレースは市場補正なしの確率です。";

/** 応答に市場補正が適用されたか（prob_calibration が無い応答は『補正の機能が無効』なので表示しない） */
export function calState(pc: ProbCalibration | null | undefined): "applied" | "skipped" | "off" {
  if (!pc) return "off";
  return pc.applied ? "applied" : "skipped";
}

export function oddsPhaseLabel(phase?: string | null): string {
  const m: Record<string, string> = { t60: "発走60分前", t15: "発走15分前", t10: "発走10分前", t05: "発走5分前", t02: "発走2分前", final: "確定", latest: "最新" };
  return phase ? (m[phase] ?? phase) : "";
}

/** モデルの確率と補正後の確率が両方ある馬だけ（詳細の表に出す） */
export function modelRows(preds: ProbModelFields[] | undefined) {
  return (preds ?? []).filter((p) => p.win_prob_model != null && p.win_prob != null).sort((a, b) => (a.horse_number ?? 0) - (b.horse_number ?? 0));
}
