/**
 * 全ページ共通のバー（components/Nav.tsx）に並べるページ導線の定義。バーはルートレイアウトに 1 つだけあり、どのページでも表示される。
 *
 * 項目を足すとき: この配列に 1 件足す（Nav.tsx は変えない）。
 *  - 新しい画面なら lib/features.ts にも登録する（feature に FeatureId を指定すると、その機能が無効のときは項目も隠れる）。
 *  - 機能フラグの無い項目（コラムなど、まだ画面が無いもの）は enabled:false で置いておき、画面ができたら true にする。
 *  - paths は、その項目が「いま開いているページ」として強調される条件（完全一致またはその配下）。
 */
import { featureEnabled, type FeatureId } from "@/lib/features";

export type SiteNavItem = {
  id: string;
  label: string;
  href: string;
  /** 強調表示（現在地）にするパス。完全一致またはその配下 */
  paths: string[];
  /** 紐づく機能フラグ。無効なら項目も隠す */
  feature?: FeatureId;
  /** 機能フラグを持たない項目の表示/非表示（既定 true） */
  enabled?: boolean;
};

export const SITE_NAV: SiteNavItem[] = [
  {
    id: "megu-ai",
    label: "めぐAI",
    href: "/weekly-predictions",
    paths: ["/weekly-predictions", "/races", "/race", "/prediction-results"],
    feature: "predictions",
  },
  // 今後の例（画面ができたら enabled を外す）:
  { id: "megu-column", label: "めぐコラム", href: "/column", paths: ["/column"], enabled: false },
];

function matches(pathname: string, p: string): boolean {
  return pathname === p || pathname.startsWith(p + "/");
}

export function visibleNavItems(): SiteNavItem[] {
  return SITE_NAV.filter((it) => (it.feature ? featureEnabled(it.feature) : it.enabled !== false));
}

export function isNavActive(item: SiteNavItem, pathname: string | null): boolean {
  return !!pathname && item.paths.some((p) => matches(pathname, p));
}
