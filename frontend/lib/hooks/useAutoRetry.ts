"use client";
import { useEffect, useRef } from "react";

/** active の間、firstMs 後から再試行を繰り返す（間隔は 1.5 倍ずつ延ばし maxMs まで）。利用者に再取得の操作をさせないための自動再試行。 */
export function useAutoRetry(active: boolean, retry: (() => void) | undefined, firstMs = 3000, maxMs = 30000): void {
  const fn = useRef(retry);
  fn.current = retry;
  const enabled = active && !!retry;
  useEffect(() => {
    if (!enabled) return;
    let delay = firstMs;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      timer = setTimeout(() => {
        if (typeof document !== "undefined" && document.hidden) { tick(); return; }     // 見ていないタブでは取りに行かない
        fn.current?.();
        delay = Math.min(maxMs, Math.round(delay * 1.5));
        tick();
      }, delay);
    };
    tick();
    return () => clearTimeout(timer);
  }, [enabled, firstMs, maxMs]);
}
