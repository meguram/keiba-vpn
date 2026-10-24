import Link from "next/link";

export default function NotFound() {
  return (
    <div style={{ textAlign: "center", padding: "64px 16px", color: "var(--text)" }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>🐎</div>
      <h1 style={{ fontSize: 20, fontWeight: 700, color: "#f0f6fc", marginBottom: 8 }}>ページが見つかりません</h1>
      <p style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 20 }}>URL が間違っているか、ページが移動または非公開になっています。</p>
      <div style={{ display: "flex", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
        <Link href="/races" style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid var(--accent)", color: "var(--accent)", textDecoration: "none", fontSize: 14 }}>レース一覧へ</Link>
        <Link href="/" style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid var(--border)", color: "var(--text-dim)", textDecoration: "none", fontSize: 14 }}>ホームへ</Link>
      </div>
    </div>
  );
}
