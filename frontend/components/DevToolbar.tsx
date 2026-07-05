"use client";

import { useState } from "react";

type PullResult = {
  status?: string;
  message?: string;
  before?: string;
  after?: string;
};

type Props = {
  loggedIn: boolean;
};

export function DevToolbar({ loggedIn }: Props) {
  const [pullBusy, setPullBusy] = useState(false);
  const [pullMsg, setPullMsg] = useState("");

  if (!loggedIn) {
    return null;
  }

  async function runGitPull() {
    if (pullBusy) return;
    setPullBusy(true);
    setPullMsg("");
    try {
      const res = await fetch("/api/v1/admin/git-pull", {
        method: "POST",
        credentials: "include",
      });
      const data: PullResult = await res.json().catch(() => ({}));
      const text = data.message || (res.ok ? "完了" : "失敗");
      setPullMsg(data.before && data.after ? `${text} (${data.before}→${data.after})` : text);
    } catch {
      setPullMsg("通信エラー");
    } finally {
      setPullBusy(false);
    }
  }

  return (
    <div className="ml-auto flex items-center gap-2">
      <button
        type="button"
        className="rounded border px-2 py-1 text-xs"
        style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}
        disabled={pullBusy}
        onClick={runGitPull}
        title="origin から fast-forward pull（未コミット変更がある場合はスキップ）"
      >
        {pullBusy ? "pull中…" : "git pull"}
      </button>
      {pullMsg && (
        <span className="max-w-[14rem] truncate text-xs" style={{ color: "var(--text-dim)" }} title={pullMsg}>
          {pullMsg}
        </span>
      )}
    </div>
  );
}
