"use client";

import { BetRecommend } from "@/components/BetRecommend";
import { MemberLockCard } from "@/components/upgrade/MemberLock";

/** 週間予測の『サンプル買い目』タブ */
export function WeeklyBetTab({ raceId, hasPred, lock, authLoading, resultKey }: { raceId: string; hasPred: boolean; lock: boolean; authLoading: boolean; resultKey?: string }) {
  if (authLoading) {
    return <div role="status" style={{ padding: "16px 0", color: "var(--text-dim)", fontSize: 13 }}>ログイン状態を確認中…</div>;
  }
  if (lock) return <MemberLockCard title="サンプル買い目" />;

  return (
    <div style={{ padding: "12px 0" }}>
      <BetRecommend raceId={raceId} hasPrediction={hasPred} resultKey={resultKey} />
    </div>
  );
}
