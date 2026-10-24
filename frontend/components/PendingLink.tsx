"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { prefetchJson } from "@/lib/httpCache";

type Props = Omit<React.ComponentProps<typeof Link>, "onClick"> & {
  /** マウスが乗った／触れ始めたときに温める GET の URL（安全な GET のみ） */
  warm?: string[];
  onClick?: React.MouseEventHandler<HTMLAnchorElement>;
};

/** クリック直後に『読み込み中』を示し、その間の再クリック（連打）を無視するリンク。 */
export function PendingLink({ warm, onClick, style, children, ...rest }: Props) {
  const [pending, setPending] = useState(false);
  const path = usePathname();
  const warmed = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setPending(false);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [path]);

  function warmUp() {
    if (warmed.current || !warm?.length) return;
    warmed.current = true;
    warm.forEach((u) => prefetchJson(u, { ttlMs: 30_000 }));
  }

  return (
    <Link
      {...rest}
      aria-busy={pending || undefined}
      onMouseEnter={warmUp}
      onFocus={warmUp}
      onTouchStart={warmUp}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;   // 新しいタブなどは通常どおり
        if (pending) { e.preventDefault(); return; }                                     // 連打は 1 回にまとめる
        onClick?.(e);
        const to = typeof rest.href === "string" ? rest.href.split("?")[0].split("#")[0] : "";
        if (to && to === path) return;                                                   // 今いるページへのリンクは待たない
        setPending(true);
        window.dispatchEvent(new Event("nav:start"));
        timer.current = setTimeout(() => setPending(false), 15_000);                    // 失敗時に固まらない
      }}
      style={{ position: "relative", ...style, ...(pending ? { opacity: 0.6, cursor: "progress" } : null) }}
    >
      {children}
      {pending && <span className="pl-spin" aria-hidden="true" />}
    </Link>
  );
}
