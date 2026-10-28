import Link from "next/link";
import { ColumnChart } from "@/components/column/ColumnChart";
import {
  COLUMN_CATEGORIES, VERDICTS, categoryLabel, columnsIn, countIn,
  type Column, type ColumnBlock, type ColumnCategory,
} from "@/lib/columns";

const dim: React.CSSProperties = { color: "var(--text-dim)", fontSize: 12 };

export function VerdictBadge({ verdict }: { verdict: Column["verdict"] }) {
  const v = VERDICTS[verdict];
  return (
    <span style={{ fontSize: 12, fontWeight: 700, padding: "2px 10px", borderRadius: 999, whiteSpace: "nowrap",
      color: v.color, border: `1px solid ${v.color}80` }}>
      {v.label}
    </span>
  );
}

export function CategoryChip({ path }: { path: string }) {
  return (
    <Link href={`/column/category/${path}`} style={{ fontSize: 11, color: "var(--text-dim)", textDecoration: "none",
      padding: "1px 8px", borderRadius: 6, border: "1px solid var(--border)", whiteSpace: "nowrap" }}>
      {categoryLabel(path)}
    </Link>
  );
}

export function ColumnCard({ column }: { column: Column }) {
  return (
    <Link href={`/column/${column.slug}`} className="card block" style={{ textDecoration: "none", color: "inherit" }}>
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1" style={{ minWidth: 0 }}>
          <p style={dim}>通説：「{column.claim}」</p>
          <h2 className="font-bold" style={{ fontSize: 16 }}>
            {column.draft && <span style={{ color: "#d29922" }}>[下書き] </span>}
            {column.title}
          </h2>
          <p className="text-sm" style={{ color: "var(--text-dim)" }}>{column.summary}</p>
        </div>
        <VerdictBadge verdict={column.verdict} />
      </div>
      <div className="flex flex-wrap items-center gap-2" style={{ marginTop: 8 }}>
        {column.categories.map((p) => <span key={p} style={dim}>#{categoryLabel(p)}</span>)}
        <span style={{ ...dim, marginLeft: "auto" }}>{column.updated ?? column.published}</span>
      </div>
    </Link>
  );
}

export function ColumnList({ columns, empty }: { columns: Column[]; empty: string }) {
  if (!columns.length) return <div className="card text-sm" style={{ color: "var(--text-dim)" }}>{empty}</div>;
  return <div className="space-y-3">{columns.map((c) => <ColumnCard key={c.slug} column={c} />)}</div>;
}

/** カテゴリの木。active は現在地のパス */
export function CategoryTree({ columns, active }: { columns: Column[]; active?: string }) {
  const render = (nodes: ColumnCategory[], prefix: string, depth: number) => (
    <ul style={{ paddingLeft: depth ? 14 : 0 }}>
      {nodes.map((c) => {
        const path = prefix ? `${prefix}/${c.id}` : c.id;
        const on = active === path;
        const n = countIn(columns, path);
        return (
          <li key={path}>
            <Link href={`/column/category/${path}`} aria-current={on ? "page" : undefined}
              className="flex items-center justify-between gap-2"
              style={{ padding: "5px 8px", borderRadius: 6, fontSize: depth ? 13 : 14, fontWeight: depth ? 400 : 600,
                textDecoration: "none", color: on ? "var(--accent)" : n ? "var(--text)" : "var(--text-dim)",
                background: on ? "rgba(88,166,255,0.12)" : "transparent" }}>
              <span>{c.label}</span>
              <span style={dim}>{n}</span>
            </Link>
            {c.children && render(c.children, path, depth + 1)}
          </li>
        );
      })}
    </ul>
  );
  return (
    <nav aria-label="コラムのカテゴリ" className="card" style={{ padding: 8 }}>
      <Link href="/column" style={{ display: "block", padding: "5px 8px", fontSize: 13, textDecoration: "none",
        color: active ? "var(--text-dim)" : "var(--accent)" }}>
        すべてのコラム（{columns.length}）
      </Link>
      {render(COLUMN_CATEGORIES, "", 0)}
    </nav>
  );
}

/** 左にカテゴリの木、右に本文。狭い画面では縦に並ぶ */
export function ColumnLayout({ columns, active, children }: { columns: Column[]; active?: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-4 md:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="hidden md:block"><div style={{ position: "sticky", top: 56 }}><CategoryTree columns={columns} active={active} /></div></aside>
      <div className="space-y-4" style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}

export function ColumnBody({ blocks }: { blocks: ColumnBlock[] }) {
  return (
    <div className="space-y-3">
      {blocks.map((b, i) => {
        switch (b.type) {
          case "heading":
            return <h2 key={i} className="font-bold" style={{ fontSize: 17, marginTop: 12 }}>{b.text}</h2>;
          case "text":
            return <p key={i} style={{ lineHeight: 1.8, whiteSpace: "pre-wrap" }}>{b.text}</p>;
          case "note":
            return (
              <p key={i} className="text-sm" style={{ padding: "8px 12px", borderLeft: "3px solid #d29922",
                background: "rgba(210,153,34,0.08)", borderRadius: 4 }}>{b.text}</p>
            );
          case "stats":
            return (
              <div key={i} className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))" }}>
                {b.items.map((s) => (
                  <div key={s.label} className="card" style={{ padding: "10px 12px" }}>
                    <p style={dim}>{s.label}</p>
                    <p className="font-bold font-mono" style={{ fontSize: 20 }}>{s.value}</p>
                    {s.sub && <p style={dim}>{s.sub}</p>}
                  </div>
                ))}
              </div>
            );
          case "chart":
            return <ColumnChart key={i} block={b} />;
          case "link":
            return (
              <Link key={i} href={b.href} className="card block" style={{ textDecoration: "none", color: "inherit", borderColor: "var(--accent)" }}>
                <p className="font-bold" style={{ color: "var(--accent)" }}>{b.label} →</p>
                {b.desc && <p className="text-sm" style={{ color: "var(--text-dim)" }}>{b.desc}</p>}
              </Link>
            );
          case "table":
            return (
              <div key={i} style={{ overflowX: "auto" }}>
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ color: "var(--text-dim)" }}>
                      {b.columns.map((h) => <th key={h} className="py-2 px-3 text-left">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {b.rows.map((r, j) => (
                      <tr key={j} style={{ borderTop: "1px solid var(--border)" }}>
                        {r.map((v, k) => (
                          <td key={k} className={`py-2 px-3 ${typeof v === "number" ? "font-mono" : ""}`}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {b.caption && <p style={{ ...dim, marginTop: 4 }}>{b.caption}</p>}
              </div>
            );
        }
      })}
    </div>
  );
}

/** トップ: カテゴリをカードで並べ、配下の小カテゴリへ 1 タップで進めるようにする */
export function CategoryGrid({ columns, nodes = COLUMN_CATEGORIES, prefix = "" }: { columns: Column[]; nodes?: ColumnCategory[]; prefix?: string }) {
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))" }}>
      {nodes.map((c) => {
        const path = prefix ? `${prefix}/${c.id}` : c.id;
        const n = countIn(columns, path);
        return (
          <div key={path} className="card" style={{ padding: 0, overflow: "hidden" }}>
            <Link href={`/column/category/${path}`} style={{ display: "block", padding: "12px 14px", textDecoration: "none", color: "inherit" }}>
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="font-bold" style={{ fontSize: 16 }}>{c.label}</h3>
                <span style={dim}>{n}本</span>
              </div>
              {c.desc && <p className="text-sm" style={{ color: "var(--text-dim)", marginTop: 2 }}>{c.desc}</p>}
            </Link>
            {c.children && (
              <ul style={{ borderTop: "1px solid var(--border)" }}>
                {c.children.map((ch) => {
                  const cp = `${path}/${ch.id}`;
                  return (
                    <li key={cp}>
                      <Link href={`/column/category/${cp}`} className="flex items-center justify-between gap-2"
                        style={{ padding: "8px 14px", fontSize: 13, textDecoration: "none", color: "var(--text)", borderTop: "1px solid rgba(36,48,73,.4)" }}>
                        <span>{ch.label}</span>
                        <span style={dim}>{countIn(columns, cp)} ›</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** カテゴリページの上に出す小カテゴリのカード（子があるときだけ） */
export function SubCategories({ columns, path, node }: { columns: Column[]; path: string; node: ColumnCategory }) {
  if (!node.children?.length) return null;
  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
      {node.children.map((ch) => {
        const cp = `${path}/${ch.id}`;
        const items = columnsIn(columns, cp);
        return (
          <Link key={cp} href={`/column/category/${cp}`} className="card block" style={{ textDecoration: "none", color: "inherit", padding: "10px 12px" }}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-bold" style={{ fontSize: 14 }}>{ch.label}</span>
              <span style={dim}>{items.length}本</span>
            </div>
            {ch.desc && <p style={{ ...dim, marginTop: 2 }}>{ch.desc}</p>}
          </Link>
        );
      })}
    </div>
  );
}
