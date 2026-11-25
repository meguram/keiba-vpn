/**
 * 匿名でも読める文章のページ（/learn）。本文はサーバで描き、HTML に入る（AdSense の審査・広告の巡回が読める）。
 * 計画: docs/operations/adsense_plan_20261125.md §8.6
 *
 * 出してよいコラムは許可リストだけ（統括役の承認 2026-11-25）。条件:
 * - 確定の成績（着順・通過順など公式の結果）と確定オッズを集計した率だけで、netkeiba 独自の情報（追い切り・厩舎コメント・指数など）と確定前のオッズを使わない
 *   （ユーザーの判断 2026-11-25: 確定の結果・確定オッズは公式の事実で、匿名にも出してよい）
 * - 区間ラップ・馬体重を使わない（取得元の規約を確かめていない）
 * - 記事のタイトル・本文に「買い」・回収率を出さない（用語集の語の説明は除く）
 * 足すときは tests/frontend/test_learn_columns.py の検査を通す。
 */
import type { Column } from "@/lib/columns";

export const PUBLIC_COLUMN_SLUGS = [
  "even-gate",
  "full-gate-outside",
  "opening-week-inside",
  "gate-more",
  "direction-switch",
  "distance-extension",
  "old-horse",
  "older-mare",
  "layoff-first-start",
  "second-start-after-layoff",
  "class-up-first",
  "first-time",
  "local-to-central",
  "style-by-surface",
  "trip-region",
] as const;

const PUBLIC = new Set<string>(PUBLIC_COLUMN_SLUGS);

export function isPublicColumn(slug: string): boolean {
  return PUBLIC.has(slug);
}

/** 本文の「関連するコラム」のリンクを、公開の記事は /learn/column/ へ付け替え、公開でない記事へのリンクは外す */
function relink(node: unknown): unknown {
  if (Array.isArray(node)) {
    const out: unknown[] = [];
    for (const x of node) {
      if (x && typeof x === "object" && (x as { type?: string }).type === "link") {
        const href = String((x as { href?: string }).href ?? "");
        const m = /^\/column\/([^/?#]+)$/.exec(href);
        if (!m || !PUBLIC.has(m[1])) continue;
        out.push({ ...(x as object), href: `/learn/column/${m[1]}` });
        continue;
      }
      out.push(relink(x));
    }
    return out;
  }
  if (node && typeof node === "object") {
    const o: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) o[k] = relink(v);
    return o;
  }
  return node;
}

export function publicColumn(col: Column): Column {
  return { ...col, body: relink(col.body) as Column["body"] };
}

/** 用語集（/learn/glossary）に、コラムの用語集（guides/glossary.json）と並べて出すサイトの言葉。言い方は docs/operations/wording_glossary.md に合わせる */
export const SITE_TERMS: { term: string; desc: string }[] = [
  { term: "AI の予想", desc: "過去のレースの成績から作ったモデルが計算した、各馬が 1 着・3 着以内に入る見込みです。参考情報で、的中を保証しません。" },
  { term: "印（◎○▲△）", desc: "AI の予想をもとに付けた、馬ごとの目安です。◎ が最も高い評価です。" },
  { term: "サンプル買い目", desc: "AI の予想とオッズから、仮に組んだ買い目の例です。買うことを勧めるものではありません。" },
  { term: "参考度", desc: "レースごとに、AI の予想がどれだけ参考になりそうかを 0〜4 の段階で表したものです（見送り・標準・注目・高注目・最注目）。" },
  { term: "確定オッズ", desc: "締め切った後に決まった、最終のオッズです。めぐコラムの「オッズから見込まれる率」は、確定オッズから計算しています。" },
  { term: "結果の確定", desc: "着順と払戻が公式に決まった状態です。" },
];
