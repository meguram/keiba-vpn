/** レース詳細へ移動している間、すぐに骨組みを出す（クリックしてから何も変わらない時間をなくす） */
export default function Loading() {
  const bar = (w: string, h = 14) => <div className="sk" style={{ width: w, height: h, borderRadius: 6 }} />;
  return (
    <div style={{ padding: 24 }} aria-busy="true" aria-label="レースデータを読み込み中">
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>{bar("72px", 36)}{bar("72px", 36)}{bar("72px", 36)}</div>
      <div className="card" style={{ marginBottom: 16, display: "grid", gap: 10 }}>
        {bar("38%", 22)}
        {bar("60%")}
        {bar("45%")}
      </div>
      <div className="card" style={{ display: "grid", gap: 10 }}>
        {Array.from({ length: 8 }).map((_, i) => <div key={i}>{bar(i % 2 ? "92%" : "100%", 28)}</div>)}
      </div>
      <style>{`.sk{background:linear-gradient(90deg,var(--surface2,#15213c),var(--border,#243358),var(--surface2,#15213c));background-size:200% 100%;animation:sk 1.2s ease-in-out infinite}@keyframes sk{from{background-position:200% 0}to{background-position:-200% 0}}`}</style>
    </div>
  );
}
