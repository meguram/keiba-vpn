/**
 * めぐコラム: 「○○の傾向がある」といった通説を、実データで調べた結果の記事。
 *
 * 記事は content/columns/<slug>.json に 1 件 1 ファイルで置き、content/columns/index.ts に登録する。
 * カテゴリは下の COLUMN_CATEGORIES（階層あり）。記事の categories には "bloodline/sire" のようにパスで指定し、
 * 複数指定できる（枠番×血統のような横断的な記事のため）。親カテゴリのページには配下の記事も並ぶ。
 */

export type ColumnCategory = {
  id: string;
  label: string;
  desc?: string;
  children?: ColumnCategory[];
};

export const COLUMN_CATEGORIES: ColumnCategory[] = [
  {
    id: "upset", label: "荒れやすさ", desc: "波乱が起きる条件と、その理由",
    children: [
      { id: "conditions", label: "荒れやすい条件", desc: "どんな条件のレースが荒れるのか" },
      { id: "distance-course", label: "距離・ローカル", desc: "短距離やローカル開催は本当に荒れるのか" },
    ],
  },
  {
    id: "style", label: "脚質・展開", desc: "逃げ・先行、ダート替わりなどの走り方の傾向",
    children: [
      { id: "front-runner", label: "逃げ・先行", desc: "逃げ馬は有利？芝とダート、距離での違い" },
      { id: "surface-switch", label: "初ダート・替わり", desc: "芝からダートに替わったときの脚質・枠順" },
    ],
  },
  {
    id: "horse", label: "馬の属性", desc: "性別・馬具・生まれによる違い",
    children: [
      { id: "sex", label: "性別・去勢", desc: "牝馬の季節性、去勢明け初戦など" },
      { id: "gear", label: "馬具", desc: "ブリンカーなど、装着の効果とベスト条件" },
      { id: "birth", label: "生まれ・母の年齢", desc: "初年度産駒、高齢出産馬" },
    ],
  },
  {
    id: "bloodline", label: "血統", desc: "種牡馬・系統ごとの成長曲線と得意条件",
    children: [
      { id: "maturity", label: "早熟・晩成", desc: "年齢で好走距離や舞台が変わる血統" },
    ],
  },
  {
    id: "travel", label: "輸送・ローテーション", desc: "遠征・滞在・間隔による違い",
    children: [
      { id: "transport", label: "輸送と滞在", desc: "同じレベルの馬で、どれくらい差が出るか" },
    ],
  },
  {
    id: "jockey", label: "騎手", desc: "騎手ごとの得意・不得意",
    children: [
      { id: "distance", label: "距離別の成績", desc: "騎手ごとに、距離で好走率がどう変わるか" },
      { id: "change", label: "乗り替わり", desc: "騎手が替わると成績はどう変わるか" },
    ],
  },
];

/** 調査結果の結論 */
export type ColumnVerdict = "supported" | "partial" | "not_supported" | "inconclusive" | "pending";

export const VERDICTS: Record<ColumnVerdict, { label: string; color: string }> = {
  supported: { label: "傾向あり", color: "#3fb950" },
  partial: { label: "条件付きで傾向あり", color: "#d29922" },
  not_supported: { label: "傾向なし", color: "#f85149" },
  inconclusive: { label: "判断保留", color: "#8b949e" },
  pending: { label: "集計準備中", color: "#58a6ff" },
};

/** 本文の部品。上から順に描画する */
export type ColumnBlock =
  | { type: "heading"; text: string }
  | { type: "text"; text: string }
  | { type: "stats"; items: { label: string; value: string; sub?: string }[] }
  | { type: "table"; columns: string[]; rows: (string | number)[][]; caption?: string }
  | { type: "note"; text: string }
  | { type: "link"; href: string; label: string; desc?: string }
  | ColumnChartBlock
  | JockeyDistanceChartBlock
  | { type: "accordion"; title: string; blocks: ColumnBlock[] };

/** 折れ線グラフ。series が複数あるときは、見たい系列を切り替えて重ねて見られる（baseline は常に点線で出す） */
export type ColumnChartBlock = {
  type: "chart";
  title: string;
  xLabel: string;
  yLabel: string;
  xs: string[];
  series: ChartSeries[];
  /**
   * 「その条件」と「それ以外」の比較。a・b は series の name。2 系列を最初から並べて出し、
   * p（横軸の点ごとの差の p 値）が 0.05 未満の点に印を付ける（src/research/columns/compare_series.py が書く）
   */
  compare?: { a: string; b: string; p: (number | null)[] };
  /** 値の種類。pct（既定）は率で「12.3%」、num は人気などの数で「4.46」と出す */
  format?: ChartFormat;
  /**
   * プルダウンで切り替える条件（例: 距離・芝ダート）。各 options の先頭は「全体」で、全部が先頭のときは series をそのまま出す。
   * それ以外の組み合わせは variants[選んだ値を "|" でつないだキー] の系列に差し替える
   */
  filters?: { label: string; options: string[] }[];
  variants?: Record<string, ChartVariant>;
  /** 右の縦軸に描く系列（横軸が同じで単位の違う値。例: 左に複勝率、右に平均人気）。常に表示し、色を分ける */
  y2?: { label: string; format?: ChartFormat; series: ChartSeries[] };
  /** 値が実データでなく画面確認用のダミー */
  dummy?: boolean;
  caption?: string;
};

/** プルダウンで選んだ条件での系列（compare・caption は省略すると元のものを使う） */
export type ChartVariant = { series: ChartSeries[]; y2Series?: ChartSeries[]; compare?: ColumnChartBlock["compare"]; caption?: string };

/** プルダウンの選択に応じたグラフ（全部が先頭の選択肢、または該当する組み合わせが無ければ元のまま） */
export function chartVariant(block: ColumnChartBlock, picked: string[]): ColumnChartBlock {
  const v = block.variants?.[picked.join("|")];
  if (!v || picked.every((p, i) => p === block.filters?.[i]?.options[0])) return block;
  const y2 = block.y2 && v.y2Series ? { ...block.y2, series: v.y2Series } : block.y2;
  return { ...block, series: v.series, y2, compare: v.compare ?? block.compare, caption: v.caption ?? block.caption };
}

/** 右軸の上限。左軸の補助線（intervals 本の区切り）にそろうよう、区切りの幅を切りのよい数にする */
export function axis2Max(series: { values: (number | null)[]; ci?: ([number, number] | null)[] }[], intervals: number, format: ChartFormat = "num"): number {
  const vs = series.flatMap((s) => [...s.values, ...(s.ci ?? []).map((c) => (c ? c[1] : null))]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const m = vs.length ? Math.max(...vs) : 1;
  const raw = m / Math.max(1, intervals);
  const nice = (format === "pct" ? [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5] : [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]).find((st) => st >= raw - 1e-9) ?? raw;
  return Math.round(nice * intervals * 1000) / 1000;
}

/** n: 点ごとの標本数、ci: 点ごとの 95% 信頼区間 [下限, 上限] */
export type ChartSeries = { name: string; values: (number | null)[]; baseline?: boolean; n?: (number | null)[]; ci?: ([number, number] | null)[] };

export type ChartFormat = "pct" | "num";

/** グラフの値の表示（率は %、数は小数 2 桁） */
export function formatValue(v: number | null | undefined, format: ChartFormat = "pct"): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  return format === "num" ? v.toFixed(2) : `${(v * 100).toFixed(1)}%`;
}

/** 縦軸の目盛りの間隔 */
export function chartStep(max: number, format: ChartFormat = "pct"): number {
  if (format === "num") return max <= 10 ? 1 : 5;
  return max <= 0.2 ? 0.05 : 0.1;
}

/** 有意とみなす p 値の境目 */
export const SIGNIFICANT_P = 0.05;

/** p 値の表示（小さい値は「<0.001」） */
export function formatP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "–";
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}

/** グラフの縦軸の上限（値と信頼区間の最大を切りのよい数に丸める。値が無いときは 0.5） */
export function chartMax(series: { values: (number | null)[]; ci?: ([number, number] | null)[] }[], format: ChartFormat = "pct"): number {
  const vs = series.flatMap((s) => [...s.values, ...(s.ci ?? []).map((c) => (c ? c[1] : null))]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!vs.length) return format === "num" ? 5 : 0.5;
  const m = Math.max(...vs);
  const step = chartStep(m, format);
  return Math.round(Math.max(step, Math.ceil(m / step - 1e-9) * step) * 100) / 100;
}

/** 実データのある系列が 1 つも無い */
export function chartIsEmpty(series: { values: (number | null)[] }[]): boolean {
  return !series.some((s) => s.values.some((v) => typeof v === "number" && Number.isFinite(v)));
}

/** 騎手別・距離帯別勝率グラフ（プルダウン＋ソート切り替え付き） */
export type JockeyDistEntry = {
  name: string;
  rank: number;
  reading: string;
  values: (number | null)[];
};

export type JockeyDistanceChartBlock = {
  type: "jockey_distance_chart";
  title: string;
  xs: string[];
  baseline: (number | null)[];
  jockeys: JockeyDistEntry[];
  caption?: string;
};

export type Column = {
  slug: string;
  title: string;
  /** 検証した通説（「○○の傾向がある」） */
  claim: string;
  verdict: ColumnVerdict;
  /** 一覧に出す要約（1〜2 文） */
  summary: string;
  /** カテゴリのパス（"bloodline/sire" など）。先頭が主カテゴリ */
  categories: string[];
  /** 公開日 YYYY-MM-DD */
  published: string;
  updated?: string;
  /** 集計の対象データ（期間・条件・件数など） */
  dataset: { period: string; conditions?: string; sample?: string };
  body: ColumnBlock[];
  /** true なら開発環境でだけ表示する */
  draft?: boolean;
};

/** パス（"bloodline/sire"）から、ルートからその位置までのカテゴリ列を返す。存在しなければ null */
export function categoryTrail(path: string, tree: ColumnCategory[] = COLUMN_CATEGORIES): ColumnCategory[] | null {
  const trail: ColumnCategory[] = [];
  let level: ColumnCategory[] | undefined = tree;
  for (const id of path.split("/").filter(Boolean)) {
    const hit: ColumnCategory | undefined = level?.find((c) => c.id === id);
    if (!hit) return null;
    trail.push(hit);
    level = hit.children;
  }
  return trail.length ? trail : null;
}

/** 全カテゴリのパス（親も含む）。静的生成用 */
export function allCategoryPaths(tree: ColumnCategory[] = COLUMN_CATEGORIES, prefix = ""): string[] {
  return tree.flatMap((c) => {
    const p = prefix ? `${prefix}/${c.id}` : c.id;
    return [p, ...allCategoryPaths(c.children ?? [], p)];
  });
}

function under(path: string, target: string): boolean {
  return target === path || target.startsWith(path + "/");
}

/** そのカテゴリ（配下を含む）に属する記事。新しい順 */
export function columnsIn(columns: Column[], path: string): Column[] {
  return sortByDate(columns.filter((c) => c.categories.some((t) => under(path, t))));
}

export function countIn(columns: Column[], path: string): number {
  return columns.filter((c) => c.categories.some((t) => under(path, t))).length;
}

export function sortByDate(columns: Column[]): Column[] {
  return [...columns].sort((a, b) => (b.updated ?? b.published).localeCompare(a.updated ?? a.published));
}

/** カテゴリパスを「血統 › 父（種牡馬）」の表示名にする */
export function categoryLabel(path: string): string {
  return categoryTrail(path)?.map((c) => c.label).join(" › ") ?? path;
}
