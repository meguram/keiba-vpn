"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback } from "react";
import { PrefetchKind } from "next/dist/client/components/router-reducer/router-reducer-types";

type LinkProps = React.ComponentProps<typeof Link>;

/** 画面に入っただけでは先読みせず、マウスが乗った・触れた・フォーカスしたときだけ先読みする（router.prefetch）。
 *  Next 15 の Link は既定で画面に入ったリンクを全部先読みし、一覧の多い画面では 1 画面で十数件の区画の取得になっていた。
 *  prefetch={false} はマウスが乗ったときの先読みも止めるので、ここで呼ぶ。
 *  種類は Link の既定と同じ AUTO（動的な経路は loading の区切りまで）。router.prefetch の既定の FULL はページを丸ごと描画させる。
 *  同じ href の重複は Next のルータのキャッシュが使い回すので、ここでは覚えない（古くなった後の hover で取り直せるように）。 */
export function useIntentPrefetch(href: LinkProps["href"]): () => void {
  const router = useRouter();
  return useCallback(() => {
    if (typeof href !== "string" || !href.startsWith("/")) return;
    try { router.prefetch(href, { kind: PrefetchKind.AUTO }); } catch { /* 先読みの失敗は無視 */ }
  }, [router, href]);
}

export function IntentLink({ onMouseEnter, onTouchStart, onFocus, ...rest }: LinkProps) {
  const warm = useIntentPrefetch(rest.href);
  return (
    <Link
      {...rest}
      prefetch={false}
      onMouseEnter={(e) => { warm(); onMouseEnter?.(e); }}
      onTouchStart={(e) => { warm(); onTouchStart?.(e); }}
      onFocus={(e) => { warm(); onFocus?.(e); }}
    />
  );
}
