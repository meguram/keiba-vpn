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
      { id: "lineage", label: "系統", desc: "父系の系統で、産駒の適性はどれくらい決まるか" },
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
      { id: "class-venue", label: "クラス・競馬場・時期", desc: "騎手ごとの得意クラス・競馬場・季節" },
    ],
  },
  {
    id: "weather", label: "天候", desc: "風・気温など、気象の影響",
    children: [
      { id: "wind", label: "風", desc: "直線の向かい風・追い風と、風の強さの影響" },
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
  | ColumnHeatmapBlock
  | ColumnTreemapBlock
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
  /** true なら、プルダウンは前の選択に合わせて variants に存在する選択肢だけを出す（会場 → 距離 のような階層）。先頭の選択肢を「全体」とみなさず、常に variants を引く */
  cascade?: boolean;
  /** true なら縦軸を 0 始まりにせず、値（と信頼区間）の範囲に合わせる（平均ラップ 12 秒前後の差を見るときなど） */
  fitAxis?: boolean;
  /** 最初に重ねて出す系列の name（省略すると先頭の 1 つ） */
  initial?: string[];
  /** 右の縦軸に描く系列（横軸が同じで単位の違う値。例: 左に複勝率、右に平均人気）。常に表示し、色を分ける */
  y2?: { label: string; format?: ChartFormat; series: ChartSeries[] };
  /** 値が実データでなく画面確認用のダミー */
  dummy?: boolean;
  /** 標本数 n がこれ未満の点と、その点につながる線を点線で描く（省略すると全部実線） */
  minN?: number;
  caption?: string;
};

/** プルダウンで選んだ条件での系列（compare・caption は省略すると元のものを使う） */
export type ChartVariant = { series: ChartSeries[]; xs?: string[]; y2Series?: ChartSeries[]; compare?: ColumnChartBlock["compare"]; caption?: string };

/** プルダウンの選択に応じたグラフ（全部が先頭の選択肢、または該当する組み合わせが無ければ元のまま） */
export function chartVariant(block: ColumnChartBlock, picked: string[]): ColumnChartBlock {
  const v = block.variants?.[picked.join("|")];
  if (!v || (!block.cascade && picked.every((p, i) => p === block.filters?.[i]?.options[0]))) return block;
  const y2 = block.y2 && v.y2Series ? { ...block.y2, series: v.y2Series } : block.y2;
  return { ...block, xs: v.xs ?? block.xs, series: v.series, y2, compare: v.compare ?? block.compare, caption: v.caption ?? block.caption };
}

/** cascade のとき、fi 番目のプルダウンに出す選択肢（それより前の選択 picked と組み合わせて variants にあるもの）。cascade でなければ全部 */
export function cascadeOptions(block: ColumnChartBlock, picked: string[], fi: number): string[] {
  const f = block.filters?.[fi];
  if (!f) return [];
  if (!block.cascade || !block.variants) return f.options;
  const head = picked.slice(0, fi);
  const keys = Object.keys(block.variants).map((k) => k.split("|"));
  return f.options.filter((o) => keys.some((k) => head.every((h, i) => k[i] === h) && k[fi] === o));
}

/** fi 番目の選択を value にしたときの選択全体。後ろの選択が成り立たなくなったら、その時点で選べる先頭に寄せる */
export function cascadePick(block: ColumnChartBlock, picked: string[], fi: number, value: string): string[] {
  const next = [...picked];
  next[fi] = value;
  if (!block.cascade) return next;
  for (let i = fi + 1; i < next.length; i++) {
    const opts = cascadeOptions(block, next, i);
    if (!opts.includes(next[i])) next[i] = opts[0] ?? next[i];
  }
  return next;
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

/** pct: 率（12.3%）、num: 数（4.46）、pt: 率の差（+1.2pt。実際 − 見込みなど、負になりうる値） */
export type ChartFormat = "pct" | "num" | "pt";

/** グラフの値の表示（率は %、数は小数 2 桁、率の差は符号付きの pt） */
export function formatValue(v: number | null | undefined, format: ChartFormat = "pct"): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  if (format === "pt") return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}pt`;
  return format === "num" ? v.toFixed(2) : `${(v * 100).toFixed(1)}%`;
}

/** 縦軸の目盛りの文字（率は %、率の差は pt、数はそのまま） */
export function formatTick(t: number, format: ChartFormat = "pct"): string {
  if (format === "num") return `${+t.toFixed(2)}`;
  const v = Math.round(t * 1000) / 10;
  return format === "pt" ? `${v > 0 ? "+" : ""}${v}` : `${v}%`;
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

/**
 * 縦軸の範囲と目盛りの間隔。値（と信頼区間）がすべて 0 以上なら、今までどおり 0 始まり（chartMax・chartStep）。
 * 負の値があれば、下限も値から決める（0 を必ず含み、目盛りは 0 を通る）
 */
export function chartRange(series: { values: (number | null)[]; ci?: ([number, number] | null)[] }[], format: ChartFormat = "pct", fit = false): { min: number; max: number; step: number } {
  const vs = series.flatMap((s) => [...s.values, ...(s.ci ?? []).flatMap((c) => (c ? [c[0], c[1]] : []))]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = vs.length ? Math.min(...vs) : 0;
  if (fit && vs.length) {
    const hi = Math.max(...vs);
    const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
    const step = steps.find((st) => (hi - lo) / st <= 8) ?? steps[steps.length - 1];
    const r = (v: number) => Math.round(v * 1e6) / 1e6;
    const mn = r(Math.floor(lo / step + 1e-9) * step);
    const mx = r(Math.max(mn + step, Math.ceil(hi / step - 1e-9) * step));
    return { min: mn, max: mx, step };
  }
  if (!(lo < 0)) {
    const max = chartMax(series, format);
    return { min: 0, max, step: chartStep(max, format) };
  }
  const hi = Math.max(0, ...vs);
  const span = hi - lo;
  const steps = format === "num" ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] : [0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
  const step = steps.find((st) => span / st <= 6) ?? steps[steps.length - 1];
  const r = (v: number) => Math.round(v * 1e6) / 1e6;
  return { min: r(Math.floor(lo / step + 1e-9) * step), max: r(Math.max(0, Math.ceil(hi / step - 1e-9) * step)), step };
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
  /** 点ごとの騎乗数 */
  n?: (number | null)[];
  /** 点ごとの、騎乗馬の人気（単勝オッズ）から見込まれる勝率 */
  expected?: (number | null)[];
};

export type JockeyDistanceChartBlock = {
  type: "jockey_distance_chart";
  title: string;
  xs: string[];
  baseline: (number | null)[];
  jockeys: JockeyDistEntry[];
  /** 騎乗数がこれ未満の点と、その点につながる線を点線で描く（省略すると全部実線） */
  minN?: number;
  /** 縦軸の名前（既定は「勝率」）、baseline と expected の凡例 */
  yLabel?: string;
  baselineLabel?: string;
  expectedLabel?: string;
  caption?: string;
};

/**
 * ヒートマップ（行 × 列の表に、値の大きさで色を付ける）。色は center（既定 0）より上が青、下が橙の発散型。
 * values[行][列]。n があれば各マスに標本数を出し、minN 未満のマスは色を薄くする
 */
export type ColumnHeatmapBlock = {
  type: "heatmap";
  title: string;
  rowLabel: string;
  colLabel: string;
  rows: string[];
  cols: string[];
  values: (number | null)[][];
  n?: (number | null)[][];
  ci?: ([number, number] | null)[][];
  format?: ChartFormat;
  center?: number;
  /** 色の濃さが最大になる |値 − center|（省略すると値の最大） */
  scale?: number;
  minN?: number;
  /** 凡例の左右（例: ["見込みより下", "見込みより上"]） */
  legend?: [string, string];
  caption?: string;
};

/**
 * ツリーマップの節。value は面積（産駒の出走数など）。metrics[i] は block.metrics[i] の [値, 95%区間の下限, 上限, 標本数]（区間が無ければ null）。
 * kind: line＝名前の付いた系統、branch＝父系の祖先の馬でまとめた群、pool＝少ない種牡馬をまとめた葉（sires 頭）。省略は種牡馬 1 頭の葉
 */
export type TreemapNode = {
  id: string;
  name: string;
  value: number;
  metrics: (number | null)[][];
  kind?: "line" | "branch" | "pool";
  sires?: number;
  children?: TreemapNode[];
};

/** pct: 割合（51.5%）、m: 距離の差（+45m）、pos: 位置（0.47）、pt: 率の差（+1.2pt）、num: 数（+0.53） */
export type TreemapFormat = "pct" | "m" | "pos" | "pt" | "num";

/** 色を付ける指標。center から ±scale で色が最も濃くなる（発散型）。low・high は両端の意味 */
export type TreemapMetric = {
  key: string;
  label: string;
  kind?: string;
  low: string;
  high: string;
  format: TreemapFormat;
  center: number;
  scale: number;
  desc?: string;
};

/**
 * 順序を保つツリーマップ。木（root）をブラウザでレイアウトし、節をタップするとその節へ潜る（パンくずで戻る）。
 * 色はプルダウンで選んだ指標。start は最初に表示する節の id、maxDepth は枠と見出しを描く深さ
 */
export type ColumnTreemapBlock = {
  type: "treemap";
  title: string;
  metrics: TreemapMetric[];
  defaultMetric?: number;
  start?: string;
  maxDepth?: number;
  root: TreemapNode;
  caption?: string;
};

export function formatTreemapValue(v: number | null | undefined, format: TreemapFormat): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  const sign = (x: number, s: string) => (Number(s) === 0 ? s.replace(/^-/, "") : `${x > 0 ? "+" : ""}${s}`);
  switch (format) {
    case "pct": return `${(v * 100).toFixed(1)}%`;
    case "m": return `${sign(v, v.toFixed(0))}m`;
    case "pos": return v.toFixed(2);
    case "pt": return `${sign(v, (v * 100).toFixed(1))}pt`;
    default: return sign(v, v.toFixed(2));
  }
}

export type Rect = { x: number; y: number; w: number; h: number };

/** 値の列を、0 以下・非有限を 0 にして返す */
function cleanValues(values: number[]): number[] {
  return values.map((v) => (v > 0 && Number.isFinite(v) ? v : 0));
}

/** 矩形の縦横比（長い辺 / 短い辺、1 以上）。面積 0 は 1 とする */
export function aspectRatio(r: Rect): number {
  const a = Math.min(r.w, r.h), b = Math.max(r.w, r.h);
  return a > 0 ? b / a : 1;
}

/**
 * 順序を保つツリーマップのレイアウト。値の列を 2 つに分けて縦か横に切ることを繰り返す（先の要素が左か上に来るので、
 * 並び順が左→右・上→下に保たれ、似た適性が隣り合う）。分ける位置と向きは、子の矩形の (log 縦横比)² の和が最小になるものを探す
 * （2 乗で、細い短冊を強く嫌う）。この悪さは矩形の縦横比だけで決まり大きさに依らないので、(lo, hi, 縦横比を丸めた値) で覚える。
 * 値が 0 以下の要素は面積 0 の矩形
 */
export function orderedLayout(values: number[], rect: Rect): Rect[] {
  const vs = cleanValues(values);
  const n = vs.length;
  const pre = [0];
  for (const v of vs) pre.push(pre[pre.length - 1] + v);
  const sum = (lo: number, hi: number) => pre[hi] - pre[lo];
  const pos = vs.map((v) => (v > 0 ? 1 : 0));
  const memo = new Map<string, { cost: number; k: number; vertical: boolean }>();
  const STEP = 0.1;                                       // log(幅/高さ) を丸める刻み
  const best = (lo: number, hi: number, logAR: number): { cost: number; k: number; vertical: boolean } => {
    let cnt = 0;
    for (let i = lo; i < hi; i++) cnt += pos[i];
    if (cnt <= 1) return { cost: cnt ? logAR * logAR : 0, k: -1, vertical: true };
    const q = Math.round(logAR / STEP);
    const key = `${lo},${hi},${q}`;
    const hit = memo.get(key);
    if (hit) return hit;
    const ar = Math.exp(q * STEP);                          // 幅 / 高さ（面積 1 とみなす）
    const total = sum(lo, hi);
    let out = { cost: Infinity, k: lo + 1, vertical: true };
    for (let k = lo + 1; k < hi; k++) {
      const f = sum(lo, k) / total;
      if (f <= 0 || f >= 1) continue;
      // 縦に切る（左右に並べる）: 幅が f と 1-f に分かれる
      const cv = best(lo, k, Math.log(ar * f)).cost + best(k, hi, Math.log(ar * (1 - f))).cost;
      if (cv < out.cost) out = { cost: cv, k, vertical: true };
      // 横に切る（上下に並べる）: 高さが f と 1-f に分かれる
      const ch = best(lo, k, Math.log(ar / f)).cost + best(k, hi, Math.log(ar / (1 - f))).cost;
      if (ch < out.cost) out = { cost: ch, k, vertical: false };
    }
    memo.set(key, out);
    return out;
  };
  const out: Rect[] = vs.map(() => ({ x: rect.x, y: rect.y, w: 0, h: 0 }));
  const place = (lo: number, hi: number, r: Rect) => {
    let cnt = 0, only = -1;
    for (let i = lo; i < hi; i++) if (pos[i]) { cnt++; only = i; }
    if (cnt === 0) return;
    if (cnt === 1) { out[only] = r; return; }
    if (r.w <= 0 || r.h <= 0) { for (let i = lo; i < hi; i++) out[i] = { x: r.x, y: r.y, w: 0, h: 0 }; return; }
    const { k, vertical } = best(lo, hi, Math.log(r.w / r.h));
    const f = sum(lo, k) / sum(lo, hi);
    if (vertical) {
      place(lo, k, { x: r.x, y: r.y, w: r.w * f, h: r.h });
      place(k, hi, { x: r.x + r.w * f, y: r.y, w: r.w * (1 - f), h: r.h });
    } else {
      place(lo, k, { x: r.x, y: r.y, w: r.w, h: r.h * f });
      place(k, hi, { x: r.x, y: r.y + r.h * f, w: r.w, h: r.h * (1 - f) });
    }
  };
  if (n) place(0, n, rect);
  return out;
}

/** 入れ子のツリーマップで、各段の枠の内側の余白と見出しの帯 */
export const TREEMAP_HEADER = 16;
/** 内側の短い辺がこれ未満の節は、子に分けず 1 つの塊として描く（細い帯の中に、さらに細い短冊を作らない） */
export const TREEMAP_MIN_SPLIT = 10;
/** collapsed: 子があるが、小さい・細いので分けずに塊として描く節 */
export type TreemapItem = { node: TreemapNode; rect: Rect; depth: number; head: boolean; collapsed: boolean };

/** 見出しの帯を付けるか（枠を描く深さで、帯を除いても子に十分な高さが残るとき） */
export function treemapHasHeader(rect: Rect, depth: number, maxDepth: number): boolean {
  return depth > 0 && depth <= maxDepth && rect.h >= TREEMAP_HEADER * 2.2 && rect.w >= 44;
}

/**
 * 入れ子のツリーマップ全体のレイアウト（純粋な関数）。view を W×H に置き、子を orderedLayout で再帰的に詰める。
 * 枠を描く深さ（maxDepth まで）は内側に余白（小さい矩形では 1px）と見出しの帯を取り、それより深い段は余白なしで詰める
 * （段ごとに余白を取ると、深い入れ子ほど細く・小さくなるため）。内側の短い辺が TREEMAP_MIN_SPLIT 未満の節は子に分けない。
 */
export function treemapLayout(view: TreemapNode, W: number, H: number, maxDepth: number): TreemapItem[] {
  const items: TreemapItem[] = [];
  const rec = (node: TreemapNode, rect: Rect, depth: number) => {
    const head = treemapHasHeader(rect, depth, maxDepth);
    const kids = node.children ?? [];
    let inner = rect;
    if (depth > 0 && depth <= maxDepth) {
      const pad = Math.min(rect.w, rect.h) >= 24 ? 2 : 1;
      const top = head ? TREEMAP_HEADER : 0;
      inner = { x: rect.x + pad, y: rect.y + top + pad, w: Math.max(0, rect.w - pad * 2), h: Math.max(0, rect.h - top - pad * 2) };
    }
    const collapsed = kids.length > 0 && depth > 0 && Math.min(inner.w, inner.h) < TREEMAP_MIN_SPLIT;
    items.push({ node, rect, depth, head: head && !collapsed, collapsed });
    if (!kids.length || collapsed) return;
    const rs = orderedLayout(kids.map((k) => k.value), inner);
    kids.forEach((k, i) => rec(k, rs[i], depth + 1));
  };
  rec(view, { x: 0, y: 0, w: W, h: H }, 0);
  return items;
}

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
