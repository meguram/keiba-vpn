import { PageShell } from "@/components/PageShell";

const APTITUDE_DATA = [
  { horse_name: "アイアンウィル", turf_1200: 62, turf_1600: 88, turf_2000: 91, turf_2400: 79, dirt_1400: 55, dirt_1800: 48, soft_going: 83, heavy_going: 71 },
  { horse_name: "ゴールドストーム", turf_1200: 75, turf_1600: 92, turf_2000: 88, turf_2400: 82, dirt_1400: 61, dirt_1800: 54, soft_going: 78, heavy_going: 65 },
  { horse_name: "サンライズキング", turf_1200: 68, turf_1600: 84, turf_2000: 89, turf_2400: 85, dirt_1400: 58, dirt_1800: 52, soft_going: 81, heavy_going: 74 },
  { horse_name: "シルバーミスト", turf_1200: 71, turf_1600: 78, turf_2000: 82, turf_2400: 76, dirt_1400: 70, dirt_1800: 74, soft_going: 72, heavy_going: 69 },
  { horse_name: "スターラッシュ", turf_1200: 81, turf_1600: 86, turf_2000: 79, turf_2400: 71, dirt_1400: 63, dirt_1800: 57, soft_going: 75, heavy_going: 62 },
];

const COLS = [
  { key: "turf_1200", label: "芝1200" },
  { key: "turf_1600", label: "芝1600" },
  { key: "turf_2000", label: "芝2000" },
  { key: "turf_2400", label: "芝2400" },
  { key: "dirt_1400", label: "ダ1400" },
  { key: "dirt_1800", label: "ダ1800" },
  { key: "soft_going", label: "稍重" },
  { key: "heavy_going", label: "重/不良" },
] as const;

function scoreColor(v: number) {
  if (v >= 85) return "var(--ok)";
  if (v >= 70) return "var(--warn)";
  return "var(--err)";
}

export default function NoteAptitudeRacePage() {
  return (
    <PageShell title="血統適性マップ" description="コース・距離・馬場別適性スコア（AN-11）">
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">馬名</th>
              {COLS.map((c) => (
                <th key={c.key} className="py-2 px-3 text-right">{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {APTITUDE_DATA.map((row) => (
              <tr key={row.horse_name} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-medium">{row.horse_name}</td>
                {COLS.map((c) => {
                  const v = row[c.key];
                  return (
                    <td key={c.key} className="py-2 px-3 text-right font-mono text-xs" style={{ color: scoreColor(v) }}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
