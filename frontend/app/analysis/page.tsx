"use client";

import { useState } from "react";
import { PageShell } from "@/components/PageShell";

// 画面は準備中。タブの枠だけ置き、中身は各タブの設計が決まってから実装する。
const TABS = [
  {
    id: "entries",
    label: "出馬表データ",
    plan: [
      "出馬表に、騎手のコース別過去成績を並べて表示",
      "枠番ごとの成績統計（勝率・複勝率など）を表示",
    ],
  },
  {
    id: "growth",
    label: "成長曲線",
    plan: ["各出走馬の成長曲線を閲覧"],
  },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function AnalysisPage() {
  const [tab, setTab] = useState<TabId>("entries");
  const current = TABS.find((t) => t.id === tab)!;

  return (
    <PageShell title="めぐ分析" description="予想に役立つデータを、出馬表に合わせて提供します。" note={false}>
      <div role="tablist" style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--border)", flexWrap: "nowrap", overflowX: "auto" }}>
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            role="tab"
            aria-selected={tab === id}
            style={{
              padding: "10px 20px",
              minHeight: 44,
              whiteSpace: "nowrap",
              flex: "none",
              fontSize: 14,
              fontWeight: 500,
              color: tab === id ? "var(--accent)" : "var(--text-dim)",
              cursor: "pointer",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${tab === id ? "var(--accent)" : "transparent"}`,
              transition: "all 0.15s",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <div role="tabpanel" className="card space-y-2">
        <p className="font-bold">準備中</p>
        <ul className="list-disc pl-5 text-sm" style={{ color: "var(--text-dim)" }}>
          {current.plan.map((p) => <li key={p}>{p}</li>)}
        </ul>
      </div>
    </PageShell>
  );
}
