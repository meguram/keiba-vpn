import { PageShell } from "@/components/PageShell";

const MODELS = [
  { id: "keiba_lgbm", label: "keiba_lgbm v2.4", status: "healthy", latency_ms: 142, last_trained: "2026-07-01", accuracy: 0.712, drift_score: 0.031 },
  { id: "tracking_difficulty", label: "tracking_difficulty v1.8", status: "healthy", latency_ms: 98, last_trained: "2026-06-28", accuracy: 0.783, drift_score: 0.018 },
  { id: "final_odds", label: "final_odds v3.1", status: "degraded", latency_ms: 287, last_trained: "2026-06-15", accuracy: 0.641, drift_score: 0.074 },
  { id: "pace_predictor", label: "pace_predictor v2.0", status: "healthy", latency_ms: 115, last_trained: "2026-06-30", accuracy: 0.698, drift_score: 0.042 },
];

const CRON_SCHEDULE = [
  { job: "scraper:jra-race-entry", schedule: "毎日 07:00", last_run: "2026-07-04 07:00", duration_sec: 34, status: "ok" },
  { job: "scraper:jra-result", schedule: "レース翌日 06:00", last_run: "2026-07-04 06:00", duration_sec: 28, status: "ok" },
  { job: "inference:keiba_lgbm", schedule: "毎日 09:30", last_run: "2026-07-04 09:30", duration_sec: 8, status: "ok" },
  { job: "inference:pace_predictor", schedule: "毎日 09:45", last_run: "2026-07-04 09:45", duration_sec: 5, status: "ok" },
  { job: "inference:final_odds", schedule: "出走 30 分前", last_run: "2026-07-04 14:10", duration_sec: 11, status: "warn" },
  { job: "model:retrain-weekly", schedule: "毎週月曜 03:00", last_run: "2026-06-30 03:00", duration_sec: 1820, status: "ok" },
];

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { bg: string; color: string; label: string }> = {
    healthy: { bg: "rgba(34,197,94,0.18)", color: "var(--ok)", label: "正常" },
    degraded: { bg: "rgba(245,158,11,0.18)", color: "var(--warn)", label: "低下" },
    down: { bg: "rgba(239,68,68,0.18)", color: "var(--err)", label: "停止" },
    ok: { bg: "rgba(34,197,94,0.18)", color: "var(--ok)", label: "OK" },
    warn: { bg: "rgba(245,158,11,0.18)", color: "var(--warn)", label: "警告" },
    error: { bg: "rgba(239,68,68,0.18)", color: "var(--err)", label: "エラー" },
  };
  const s = map[status] ?? map.ok;
  return (
    <span className="rounded px-2 py-0.5 text-xs font-semibold" style={{ background: s.bg, color: s.color }}>
      {s.label}
    </span>
  );
}

export default function AiSlaPage() {
  return (
    <PageShell title="AI SLA 監視" description="MLflow モデルステータス・Cron ジョブ SLA ダッシュボード">
      <section className="space-y-3">
        <h2 className="font-semibold">モデルステータス</h2>
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ color: "var(--text-dim)" }}>
                <th className="py-2 px-3 text-left">モデル</th>
                <th className="py-2 px-3 text-center">状態</th>
                <th className="py-2 px-3 text-right">レイテンシ</th>
                <th className="py-2 px-3 text-right">精度</th>
                <th className="py-2 px-3 text-right">ドリフトスコア</th>
                <th className="py-2 px-3 text-right">最終学習</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.id} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="py-2 px-3 font-mono text-xs">{m.label}</td>
                  <td className="py-2 px-3 text-center"><StatusBadge status={m.status} /></td>
                  <td className="py-2 px-3 text-right font-mono">{m.latency_ms} ms</td>
                  <td className="py-2 px-3 text-right font-mono">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right font-mono" style={{ color: m.drift_score > 0.06 ? "var(--warn)" : "var(--ok)" }}>
                    {m.drift_score.toFixed(3)}
                  </td>
                  <td className="py-2 px-3 text-right text-xs" style={{ color: "var(--text-dim)" }}>{m.last_trained}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="space-y-3">
        <h2 className="font-semibold">Cron ジョブ SLA</h2>
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ color: "var(--text-dim)" }}>
                <th className="py-2 px-3 text-left">ジョブ</th>
                <th className="py-2 px-3 text-center">スケジュール</th>
                <th className="py-2 px-3 text-right">最終実行</th>
                <th className="py-2 px-3 text-right">所要時間</th>
                <th className="py-2 px-3 text-center">状態</th>
              </tr>
            </thead>
            <tbody>
              {CRON_SCHEDULE.map((j) => (
                <tr key={j.job} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="py-2 px-3 font-mono text-xs">{j.job}</td>
                  <td className="py-2 px-3 text-center text-xs" style={{ color: "var(--text-dim)" }}>{j.schedule}</td>
                  <td className="py-2 px-3 text-right font-mono text-xs">{j.last_run}</td>
                  <td className="py-2 px-3 text-right font-mono text-xs">{j.duration_sec} s</td>
                  <td className="py-2 px-3 text-center"><StatusBadge status={j.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </PageShell>
  );
}
