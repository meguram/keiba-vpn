"use client";
import { useEffect, useState } from "react";

/** 待ち時間の段階。0=すぐ(表示しない) / 1=1秒超 / 2=4秒超 / 3=10秒超。active が false になるとリセットする。 */
export type SlowStage = 0 | 1 | 2 | 3;
export const SLOW_STAGES_MS = [1000, 4000, 10000] as const;

export function useSlowStage(active: boolean): SlowStage {
  const [stage, setStage] = useState<SlowStage>(0);
  useEffect(() => {
    setStage(0);
    if (!active) return;
    const timers = SLOW_STAGES_MS.map((ms, i) => setTimeout(() => setStage((i + 1) as SlowStage), ms));
    return () => timers.forEach(clearTimeout);
  }, [active]);
  return stage;
}
