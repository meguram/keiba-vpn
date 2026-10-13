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
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [loading, setLoading] = useState(true);
  const { isMember, loading: authLoading } = useAuthStatus();

  // 一覧の取得元: まず DB 経由の /api/v1/races。空（DB 無しの dev など）ならストレージの開催日別一覧にフォールバックする
  useEffect(() => {
    if (USE_MOCK) {
      setRaces(MOCK_RACES);
      setLoading(false);
      return;
    }
    (async () => {
      try {
        const r = await fetch("/api/v1/races");
        const d = r.ok ? await r.json() : { races: [] };
        if ((d.races ?? []).length) {
          setRaces(d.races);
          return;
        }
        const days = Number(process.env.NEXT_PUBLIC_PICKER_PAST_DAYS) || 30;
        const dr = await fetch(`/api/scrape-dates?picker_past_days=${days}`, { cache: "no-store" });
        const list: string[] = dr.ok ? ((await dr.json()).dates ?? []) : [];
        setDates(list);
        const preferred = process.env.NEXT_PUBLIC_DEFAULT_RACE_DATE;
        setDate(preferred && list.includes(preferred) ? preferred : (list[0] ?? ""));
      } catch {
        setRaces([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!date) return;
    (async () => {
      try {
        const r = await fetch(`/api/race-list/${date}`);
        const d = r.ok ? await r.json() : { races: [] };
        const iso = `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}`;
        setRaces(
          (d.races ?? []).map((x: Record<string, unknown>) => ({
            race_id: String(x.race_id),
            race_name: `${x.round ?? ""}R ${x.race_name ?? ""}`.trim(),
            venue: x.venue as string | undefined,
            race_date: iso,
            start_time: (x.start_time as string | null) ?? undefined,
            surface: (x.surface as string | null) ?? undefined,
            distance: (x.distance as number | null) ?? undefined,
            field_size: (x.entries_count as number | null) ?? undefined,
          })),
        );
      } catch {
        setRaces([]);
      }
    })();
  }, [date]);

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

        {dates.length > 0 && (
          <div style={{ marginBottom: 12, fontSize: 13 }}>
            <label style={{ color: "var(--text-dim)" }}>
              開催日{" "}
              <select
                value={date}
                onChange={(e) => setDate(e.target.value)}
                style={{ marginLeft: 6, padding: "5px 8px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)" }}
              >
                {dates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </label>
          </div>
        )}
        {!loading && races.length === 0 && (
          <div style={{ color: "var(--text-dim)", fontSize: 13, padding: "24px 0" }}>表示できるレースがありません。</div>
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
                {[r.surface, r.distance ? `${r.distance}m` : "", r.field_size ? `${r.field_size}頭` : ""].filter(Boolean).join(" · ")}
              </div>
            </Link>
          ))}
        </div>
      </PageShell>
    </>
  );
}
