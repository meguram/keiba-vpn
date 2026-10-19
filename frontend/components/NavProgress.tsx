"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

/** ページ遷移中に画面上端へ細い進捗バーを出す。リンクのクリックで始まり、パスが変わったら完了する。 */
export function NavProgress() {
  const path = usePathname();
  const [state, setState] = useState<"idle" | "run" | "done">("idle");
  const guard = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const start = () => {
      setState("run");
      if (guard.current) clearTimeout(guard.current);
      guard.current = setTimeout(() => setState("idle"), 20_000);
    };
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (!a || a.target === "_blank" || a.hasAttribute("download")) return;
      const u = new URL(a.href, window.location.href);
      if (u.origin !== window.location.origin) return;
      if (u.pathname === window.location.pathname) return;
      start();
    };
    window.addEventListener("nav:start", start);
    document.addEventListener("click", onClick);
    return () => {
      window.removeEventListener("nav:start", start);
      document.removeEventListener("click", onClick);
      if (guard.current) clearTimeout(guard.current);
    };
  }, []);

  useEffect(() => {
    setState((s) => {
      if (s !== "run") return s;
      return "done";
    });
    const t = setTimeout(() => setState("idle"), 400);
    return () => clearTimeout(t);
  }, [path]);

  if (state === "idle") return null;
  return <div className={`nav-progress nav-progress-${state}`} role="progressbar" aria-label="ページを読み込み中" />;
}
