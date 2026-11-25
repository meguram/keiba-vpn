"use client";
import Script from "next/script";
import { useEffect, useState } from "react";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { ADSENSE_SCRIPT_ID, ADSENSE_SRC, shouldLoadAdsense } from "@/lib/adsense";

// 匿名・無料会員の画面にだけ、AdSense のスクリプトを読み込む（有料会員・開発者には読まない）。
// HTML は全区分で同じ（共有キャッシュの中身を区分で変えない）。判定は画面の側で、区分が分かった後に行う。
// 広告の枠（ユニット）はまだ置かない。枠を置くときは高さを先に取って表示のずれを防ぐ。
// ログインの後は、ログインの画面が全体を読み直す（app/login/page.tsx）。読んだスクリプトを捨て、区分も取り直す。
export function AdSenseLoader() {
  const auth = useAuthStatus();
  const [host, setHost] = useState("");
  useEffect(() => setHost(window.location.hostname), []);
  if (!host || !shouldLoadAdsense(auth, host, process.env.NEXT_PUBLIC_ADSENSE)) return null;
  return <Script id={ADSENSE_SCRIPT_ID}src={ADSENSE_SRC} strategy="afterInteractive" crossOrigin="anonymous" />;
}
