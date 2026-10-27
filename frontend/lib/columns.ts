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
    id: "bloodline", label: "血統", desc: "種牡馬・母父・系統ごとの得意条件",
    children: [
      { id: "sire", label: "父（種牡馬）" },
      { id: "broodmare-sire", label: "母父" },
      { id: "line", label: "系統" },
    ],
  },
  {
    id: "gate", label: "枠番・馬番", desc: "内枠有利・外枠不利などの通説",
    children: [
      { id: "inner-outer", label: "内枠・外枠" },
      { id: "by-course", label: "コース別" },
    ],
  },
  {
    id: "course", label: "コース・馬場", desc: "競馬場・距離・馬場状態による傾向",
    children: [
      { id: "track-condition", label: "馬場状態" },
      { id: "distance", label: "距離" },
      { id: "venue", label: "競馬場" },
    ],
  },
  { id: "jockey-trainer", label: "騎手・調教師", desc: "乗り替わり・コース巧者など" },
  { id: "rotation", label: "ローテーション", desc: "休み明け・間隔・臨戦過程" },
  {
    id: "horse", label: "馬の属性", desc: "馬体重・年齢・性別など",
    children: [
      { id: "body-weight", label: "馬体重" },
      { id: "age-sex", label: "年齢・性別" },
    ],
  },
];

/** 調査結果の結論 */
export type ColumnVerdict = "supported" | "partial" | "not_supported" | "inconclusive";

export const VERDICTS: Record<ColumnVerdict, { label: string; color: string }> = {
  supported: { label: "傾向あり", color: "#3fb950" },
  partial: { label: "条件付きで傾向あり", color: "#d29922" },
  not_supported: { label: "傾向なし", color: "#f85149" },
  inconclusive: { label: "判断保留", color: "#8b949e" },
};

/** 本文の部品。上から順に描画する */
export type ColumnBlock =
  | { type: "heading"; text: string }
  | { type: "text"; text: string }
  | { type: "stats"; items: { label: string; value: string; sub?: string }[] }
  | { type: "table"; columns: string[]; rows: (string | number)[][]; caption?: string }
  | { type: "note"; text: string };

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
