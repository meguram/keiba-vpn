"use client";
import { useSlowStage } from "@/lib/hooks/useSlowStage";

/** 読み込み中の表示。すぐ終わるときは回転だけ(文言でちらつかせない)、1秒を超えたら何を待っているか、4秒を超えたら時間がかかっている旨、10秒を超えたら再試行も出す。 */
export function LoadingNotice({ what, onRetry, size = 24, padding = "20px 0" }: { what: string; onRetry?: () => void; size?: number; padding?: string }) {
  const stage = useSlowStage(true);
  return (
    <div role="status" aria-live="polite" style={{ padding, textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
      <div style={{ width: size, height: size, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "ln-spin 1s linear infinite", margin: "0 auto 8px" }} />
      <style>{`@keyframes ln-spin { to { transform: rotate(360deg); } }`}</style>
      {stage >= 1 && <div>{what}を取得中…</div>}
      {stage >= 2 && <div style={{ marginTop: 4, fontSize: 12 }}>通常より時間がかかっています（混雑や再計算中の可能性があります）</div>}
      {stage >= 3 && onRetry && (
        <button type="button" onClick={onRetry} style={{ marginTop: 8, minHeight: 44, padding: "0 16px", borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.15)", color: "var(--accent)", cursor: "pointer", fontSize: 13 }}>
          もう一度取得する
        </button>
      )}
    </div>
  );
}
