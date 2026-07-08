"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { RaceSummary } from "@/lib/api";
import { PageShell } from "@/components/PageShell";
import { USE_MOCK, MOCK_RACES } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";
import { MemberUpgradeModal } from "@/components/upgrade/MemberUpgradeModal";

export default function RacesPage() {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [showModal, setShowModal] = useState(false);
  const { isMember, loading: authLoading } = useAuthStatus();

  useEffect(() => {
    if (USE_MOCK) {
      setRaces(MOCK_RACES);
      return;
    }
    fetch("/api/v1/races")
      .then((r) => (r.ok ? r.json() : { races: [] }))
      .then((d) => setRaces(d.races ?? []))
      .catch(() => setRaces([]));
  }, []);

  return (
    <>
      {showModal && <MemberUpgradeModal onClose={() => setShowModal(false)} />}
      <PageShell title="レース一覧">
        {/* Non-member banner */}
        {!authLoading && !isMember && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 16px",
              borderRadius: 8,
              background: "rgba(59,130,246,0.08)",
              border: "1px solid rgba(59,130,246,0.25)",
              marginBottom: 16,
              fontSize: 13,
              color: "var(--text)",
              flexWrap: "wrap",
            }}
          >
            <span>🔒 AI予測データは会員限定です。</span>
            <button
              onClick={() => setShowModal(true)}
              style={{
                background: "none",
                border: "none",
                color: "var(--accent)",
                cursor: "pointer",
                fontSize: 13,
                padding: 0,
                textDecoration: "underline",
              }}
            >
              会員登録はこちら →
            </button>
          </div>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {races.map((r) => (
            <Link
              key={r.race_id}
              href={`/race/${r.race_id}`}
              className="card block hover:border-accent"
            >
              <div className="text-sm" style={{ color: "var(--text-dim)" }}>
                {r.race_date} {r.start_time}
              </div>
              <div className="font-semibold">
                {r.venue} {r.race_name}
              </div>
              <div className="text-sm">
                {r.surface} {r.distance}m · {r.field_size}頭
              </div>
            </Link>
          ))}
        </div>
      </PageShell>
    </>
  );
}
