import Link from "next/link";
import { fetchApi, RaceSummary } from "@/lib/api";
import { PageShell } from "@/components/PageShell";
import { USE_MOCK, MOCK_RACES } from "@/lib/mock";

export default async function RacesPage() {
  let races: RaceSummary[] = [];
  if (USE_MOCK) {
    races = MOCK_RACES;
  } else {
    try {
      const data = await fetchApi<{ races: RaceSummary[] }>("/api/v1/races");
      races = data.races;
    } catch {
      races = [];
    }
  }

  return (
    <PageShell title="レース一覧">
      <div className="grid gap-3 md:grid-cols-2">
        {races.map((r) => (
          <Link key={r.race_id} href={`/race/${r.race_id}`} className="card block hover:border-accent">
            <div className="text-sm" style={{ color: "var(--text-dim)" }}>{r.race_date} {r.start_time}</div>
            <div className="font-semibold">{r.venue} {r.race_name}</div>
            <div className="text-sm">{r.surface} {r.distance}m · {r.field_size}頭</div>
          </Link>
        ))}
      </div>
    </PageShell>
  );
}
