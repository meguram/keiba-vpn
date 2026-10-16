"use client";
import { useEffect, useState } from "react";

export type ViewportMode = "mobile" | "desktop";

/** スマホ: 幅 < 640px、または 指で操作（pointer: coarse）で幅 < 900px。それ以外は PC。SSR/初回描画は desktop。 */
const QUERY = "(max-width: 639px), ((pointer: coarse) and (max-width: 899px))";

export function useViewportMode(): ViewportMode {
  const [mode, setMode] = useState<ViewportMode>("desktop");
  useEffect(() => {
    const mq = window.matchMedia(QUERY);
    const update = () => setMode(mq.matches ? "mobile" : "desktop");
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return mode;
}
