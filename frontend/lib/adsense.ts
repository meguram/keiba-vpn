// Google AdSense（無料ゾーンの広告）の設定と、読み込むかの判定。
// パブリッシャー ID は秘密ではない（ページの HTML・ads.txt に出る値）。
// 計画: docs/operations/adsense_plan_20261125.md

export const ADSENSE_CLIENT = "ca-pub-5321507754924902";
export const ADSENSE_SRC = `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT}`;

/** 広告のスクリプトを読み込むホスト。手元の開発・別名のホストでは読まない（自分の表示を無効なトラフィックにしない）。 */
export const ADSENSE_HOSTS = ["app.megu-keiba.com", "stg.megu-keiba.com"];

/** 読み込みの部品（components/ads/AdSenseLoader.tsx）の script の id */
export const ADSENSE_SCRIPT_ID = "adsbygoogle-js";

/** このタブで広告のスクリプトを読み込んだか（ログイン後に全体を読み直すかの判定に使う）。 */
export function adsenseLoaded(): boolean {
  if (typeof window === "undefined") return false;
  return !!document.getElementById(ADSENSE_SCRIPT_ID) || "adsbygoogle" in window;
}

export type AdAuth = { loading: boolean; failed?: boolean; isMember: boolean; isAdmin: boolean };

/**
 * 読み込むのは、区分が「匿名」か「無料会員」と確かに分かったときだけ。
 * - 確認中（loading）・確認の失敗（failed）は読まない（会員を匿名と取り違えない）。
 * - 有料会員と開発者（is_member は paid＋dev。isAdmin は dev）は読まない。
 * - 止める印（NEXT_PUBLIC_ADSENSE=off、ビルド時）があれば誰にも読まない。
 */
export function shouldLoadAdsense(auth: AdAuth, hostname: string, flag?: string): boolean {
  if ((flag ?? "").trim().toLowerCase() === "off") return false;
  if (!ADSENSE_HOSTS.includes(hostname.toLowerCase())) return false;
  if (auth.loading || auth.failed) return false;
  return !auth.isMember && !auth.isAdmin;
}
