import Link from "next/link";
import { fetchApi, RaceSummary } from "@/lib/api";
import { USE_MOCK, MOCK_RACES } from "@/lib/mock";

async function getHealth() {
  if (USE_MOCK) return "OK (mock)";
  try {
    await fetchApi<{ status: string }>("/api/v1/health");
    return "OK";
  } catch {
    return "ERROR";
  }
}

async function getRaces(): Promise<RaceSummary[]> {
  if (USE_MOCK) return MOCK_RACES.slice(0, 8);
  try {
    const data = await fetchApi<{ races: RaceSummary[] }>("/api/v1/races");
    return data.races.slice(0, 8);
  } catch {
    return [];
  }
}

export default async function DashboardPage() {
  const [health, races] = await Promise.all([getHealth(), getRaces()]);
  const cards = [
    { title: "AI予測", href: "/tracking-difficulty", color: "#58a6ff" },
    { title: "血統分析", href: "/bloodline-cluster", color: "#bc8cff" },
    { title: "データ分析", href: "/track-speed", color: "#39d2c0" },
    { title: "馬券最適化", href: "/betting", color: "#3fb950" },
  ];

  return (
    <div className="space-y-6">
      <header className="flex items-center gap-3">
        <h1 className="text-2xl font-bold">ダッシュボード</h1>
        <span className="rounded-full px-2 py-0.5 text-xs" style={{ background: health === "OK" ? "var(--ok)" : "var(--err)" }}>
          API {health}
        </span>
      </header>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {cards.map((c) => (
          <Link key={c.href} href={c.href} className="card hover:opacity-90" style={{ borderColor: c.color }}>
            <h2 style={{ color: c.color }}>{c.title}</h2>
          </Link>
        ))}
      </div>
      <section className="card">
        <h2 className="mb-3 font-semibold">直近レース</h2>
        <ul className="space-y-2">
          {races.map((r) => (
            <li key={r.race_id}>
              <Link href={`/race/${r.race_id}`} className="text-accent hover:underline">
                {r.race_date} {r.venue} {r.race_name || r.race_id}
              </Link>
            </li>
          ))}
          {races.length === 0 && <li style={{ color: "var(--text-dim)" }}>レースデータなし（DB または API 接続を確認）</li>}
        </ul>
      </section>
    </div>
  );
}
