/**
 * めぐコラムの記事一覧。記事を足すときは、この階層に <slug>.json を置いて下の ALL に 1 行足す。
 * 型と書式は lib/columns.ts の Column を参照。draft: true の記事は本番ビルドでは出ない。
 */
import type { Column } from "@/lib/columns";
import sampleGateInner from "./sample-gate-inner.json";

const ALL = [
  sampleGateInner,
] as Column[];

const SHOW_DRAFTS = process.env.NODE_ENV !== "production";

export const COLUMNS: Column[] = ALL.filter((c) => SHOW_DRAFTS || !c.draft);

export function findColumn(slug: string): Column | undefined {
  return COLUMNS.find((c) => c.slug === slug);
}
