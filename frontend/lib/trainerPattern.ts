/**
 * 調教師 × 最終週の調教パターンの成績（src/scripts/data/trainer_pattern_table.py の凍結表）。
 * めぐ分析（各馬の該当行）とめぐコラム（全件の表）で共有する。通信はしない（計算だけ）。
 *
 * 行: [出走数, 頭数(合算の段は null), 1着, 2着, 3着, 単勝払戻の合計, 複勝払戻の合計(いずれも 100 円=1.0),
 *      区間別の出走数, 区間別の1着数, 区間別の3着以内数, 区間別の単勝払戻, 区間別の複勝払戻]
 */
export type TPRow = [number, number | null, number, number, number, number, number, number[], number[], number[], number[], number[]];

export type TPLevels = { trainer: TPRow | null; place: TPRow | null; pattern: TPRow | null };

export type TPHorse = {
  key: string | null;
  horse_number: number | null;
  horse_name: string | null;
  trainer_id: string | null;
  trainer_name: string | null;
  /** 『場所|パターン』。調教の記録が無い馬は null */
  pat: string | null;
  levels: TPLevels | null;
};

export type TPFigure = { edges: number[]; max_odds: number; y0: number; y1: number; horses: TPHorse[] };

/** 静的 JSON（public/data/trainer_pattern.json） */
export type TPTable = {
  v: number; y0: number; y1: number; max_odds: number; edges: number[];
  trainers: Record<string, { n: string; p: Record<string, TPRow> }>;
};

export type TPStats = {
  n: number; heads: number | null; c1: number; c2: number; c3: number; c4p: number;
  win: number | null; place: number | null; winRoi: number | null; placeRoi: number | null;
};

const div = (a: number, b: number): number | null => (b > 0 ? a / b : null);

export function rowStats(r: TPRow): TPStats {
  const [n, heads, c1, c2, c3, wr, pr] = r;
  return { n, heads, c1, c2, c3, c4p: n - c1 - c2 - c3, win: div(c1, n), place: div(c1 + c2 + c3, n), winRoi: div(wr, n), placeRoi: div(pr, n) };
}

/** 場所と、パターン名（坂路は A1〜X、ウッド・ポリは 4F の時計帯）に分ける */
export function splitPat(pat: string): { place: string; pat: string } {
  const i = pat.indexOf("|");
  return i < 0 ? { place: pat, pat: "" } : { place: pat.slice(0, i), pat: pat.slice(i + 1) };
}

const SLOPE_WORD: Record<string, string> = {
  A3: "終いの1Fが11秒台以下で加速",
  B3: "手前の1Fが11秒台以下で、終いは減速",
  A2: "終い2Fとも12秒台で加速",
  B2: "終い2Fとも12秒台で減速",
  A1: "終いの1Fだけ12秒台で加速（手前は13秒以上）",
  B1: "手前の1Fだけ12秒台で減速（終いは13秒以上）",
  X: "終い2Fとも13秒以上（軽め）",
};

/** パターン名（A1〜X / W52.0-53.9）を言葉にする。画面には記号を出さない */
export function patWord(pat: string | null | undefined): string {
  if (!pat) return "";
  if (SLOPE_WORD[pat]) return SLOPE_WORD[pat];
  const m = /^W(.*)$/.exec(pat);
  if (!m) return pat;
  const b = m[1];
  if (b.startsWith("~")) return `4Fが${b.slice(1)}秒以内`;
  if (b.endsWith("~")) return `4Fが${b.slice(0, -1)}秒以上`;
  return `4Fが${b.replace("-", "〜")}秒`;
}

/** 行を足し合わせる（頭数は足せないので null）。調教師全体・場所の段を作る */
export function sumRows(rows: TPRow[]): TPRow | null {
  if (!rows.length) return null;
  const add = (i: number) => rows.reduce((a, r) => a + (r[i] as number), 0);
  const nb = rows[0][7].length;
  const bins = [7, 8, 9, 10, 11].map((k) => Array.from({ length: nb }, (_, j) => rows.reduce((a, r) => a + (r[k] as number[])[j], 0)));
  const r2 = (v: number) => Math.round(v * 100) / 100;
  return [add(0), null, add(2), add(3), add(4), r2(add(5)), r2(add(6)), bins[0], bins[1], bins[2], bins[3].map(r2), bins[4].map(r2)];
}

/** 区間の見出し（edges=[2,3,5,10,20], max=50 → 2倍未満 / 2-3倍 / … / 20-50倍） */
export function binLabels(edges: number[], maxOdds: number): string[] {
  const hi = [...edges, maxOdds];
  return hi.map((h, i) => (i === 0 ? `${h}倍未満` : `${hi[i - 1]}-${h}倍`));
}

export type BinStat = { label: string; n: number; w: number; t3: number; share: number; win: number | null; place: number | null; winRoi: number | null; placeRoi: number | null };

/** ポップアップの 1 行ずつ（割合・勝率・複勝率・単勝回収率・複勝回収率）と合計 */
export function binStats(r: TPRow, edges: number[], maxOdds: number): { bins: BinStat[]; total: BinStat } {
  const labels = binLabels(edges, maxOdds);
  const [, , , , , , , bn, bw, bt, bwr, bpr] = r;
  const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
  const tot = sum(bn);
  const mk = (label: string, n: number, w: number, t3: number, wr: number, pr: number): BinStat => ({
    label, n, w, t3, share: tot ? n / tot : 0, win: div(w, n), place: div(t3, n), winRoi: div(wr, n), placeRoi: div(pr, n),
  });
  return {
    bins: labels.map((l, i) => mk(l, bn[i], bw[i], bt[i], bwr[i], bpr[i])),
    total: mk("合計", tot, sum(bw), sum(bt), sum(bwr), sum(bpr)),
  };
}

/** 調教師全体 → 場所 → パターンの 3 段（調教師が表に無ければ null） */
export function levelsOf(t: { p: Record<string, TPRow> } | undefined, pat: string | null): TPLevels | null {
  if (!t) return null;
  const all = Object.entries(t.p);
  const trainer = sumRows(all.map(([, r]) => r));
  if (!pat) return { trainer, place: null, pattern: null };
  const place = splitPat(pat).place;
  return { trainer, place: sumRows(all.filter(([k]) => splitPat(k).place === place).map(([, r]) => r)), pattern: t.p[pat] ?? null };
}

export const pct = (v: number | null | undefined, d = 1): string => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
