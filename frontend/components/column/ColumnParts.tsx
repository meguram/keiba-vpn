import Link from "next/link";
import { ColumnChart, ColumnHeatmap, JockeyDistanceChart } from "@/components/column/ColumnChart";
import { ColumnTreemap } from "@/components/column/ColumnTreemap";
import { ColumnWordcloud } from "@/components/column/ColumnWordcloud";
import { ColumnTrainerPattern } from "@/components/column/ColumnTrainerPattern";
import { VerdictMark } from "@/components/column/ColumnCard";
import { ChipScroll } from "@/components/column/ChipScroll";
import {
  COLUMN_CATEGORIES, VERDICTS, articleParts, categoryLabel, frontPlan, categoryTrail, countIn,
  type Column, type ColumnBlock, type ColumnGuide,
} from "@/lib/columns";

/**
 * カテゴリの案内。PC は左の縦の一覧、スマホは上の横に流れるチップ（大カテゴリ → 選ぶと小カテゴリ）。
 * active は今のカテゴリのパス（一覧のトップでは undefined）。
 */
function CategoryNav({ columns, active }: { columns: Column[]; active?: string }) {
  const top = active?.split("/")[0];
  const topNode = COLUMN_CATEGORIES.find((c) => c.id === top);
  const cur = (p?: string) => (active === p ? "page" : undefined);
  return (
    <>
      <nav aria-label="コラムのカテゴリ" className="col-side col-panel">
        <Link href="/column" aria-current={cur(undefined)}><b style={{ fontWeight: 400 }}>すべて</b><span>{columns.length}</span></Link>
        {COLUMN_CATEGORIES.map((c) => (
          <div key={c.id}>
            <Link href={`/column/category/${c.id}`} aria-current={c.id === active ? "page" : c.id === top ? "location" : undefined}><b style={{ fontWeight: 400 }}>{c.label}</b><span>{countIn(columns, c.id)}</span></Link>
            {c.id === top && c.children && (
              <div className="col-side-sub">
                {c.children.map((ch) => {
                  const p = `${c.id}/${ch.id}`;
                  return <Link key={p} href={`/column/category/${p}`} aria-current={cur(p)}><b style={{ fontWeight: 400 }}>{ch.label}</b><span>{countIn(columns, p)}</span></Link>;
                })}
              </div>
            )}
          </div>
        ))}
      </nav>
      <div className="col-chips-top" style={{ display: "grid", gap: 8 }}>
        <nav aria-label="コラムのカテゴリ（大）" className="col-chips">
          <Link href="/column" className="col-chip" aria-current={cur(undefined)}>すべて</Link>
          {COLUMN_CATEGORIES.map((c) => (
            <Link key={c.id} href={`/column/category/${c.id}`} className="col-chip" aria-current={c.id === active ? "page" : c.id === top ? "location" : undefined}>{c.label}</Link>
          ))}
        </nav>
        {topNode?.children && (
          <nav aria-label="コラムのカテゴリ（小）" className="col-chips col-chips-sub">
            <Link href={`/column/category/${topNode.id}`} className="col-chip" aria-current={cur(topNode.id)}>{topNode.label}のすべて</Link>
            {topNode.children.map((ch) => {
              const p = `${topNode.id}/${ch.id}`;
              return <Link key={p} href={`/column/category/${p}`} className="col-chip" aria-current={cur(p)}>{ch.label}</Link>;
            })}
          </nav>
        )}
        <ChipScroll />
      </div>
    </>
  );
}

/** 一覧・カテゴリのページの枠: 左（スマホは上）にカテゴリ、右に一覧 */
export function ColumnBrowse({ columns, active, children }: { columns: Column[]; active?: string; children: React.ReactNode }) {
  return (
    <div className="col col-browse">
      <CategoryNav columns={columns} active={active} />
      <div className="col-main">{children}</div>
    </div>
  );
}

/** パンくず（めぐコラム › 大 › 小）。last=false なら末端もリンクにする */
export function ColumnTrail({ path, linkLast = true }: { path: string; linkLast?: boolean }) {
  const trail = categoryTrail(path) ?? [];
  const segs = path.split("/");
  return (
    <p className="col-small">
      <Link href="/column" className="col-link">めぐコラム</Link>
      {trail.map((c, i) => {
        const p = segs.slice(0, i + 1).join("/");
        return (
          <span key={p}>
            {" › "}
            {linkLast || i < trail.length - 1 ? <Link href={`/column/category/${p}`} className="col-link">{c.label}</Link> : c.label}
          </span>
        );
      })}
    </p>
  );
}

/** 記事（新聞の小コラムの量）: 結論・使い方 → 本文の要約 → 数字の囲み → 図か表 1 つ → 詳しい調べ方（折りたたみ）→ 関連するコラム。何を外に出すかは frontPlan が決める */
export function ColumnArticle({ col, guide, glossary }: { col: Column; guide?: ColumnGuide; glossary: Record<string, string> }) {
  const p = articleParts(col.body, guide, glossary);
  const f = frontPlan(p, col.verdict);
  const ds = col.dataset;
  const visuals = f.main === "visual" ? p.visuals.slice(1) : p.visuals;     // 2 枚目からの図は折りたたみへ
  return (
    <article className="col col-article col-stack">
      <div className="col-lead">
        <p className="col-lead-k">結論: <VerdictMark verdict={col.verdict} /></p>
        <p className="col-lead-t">{p.answer}</p>
        {p.use && <p className="col-lead-u">{p.use.title}: {p.use.text}</p>}
      </div>
      {f.showSummary && <p className="col-pre">{p.summary}</p>}
      {f.showStats && <Stats items={p.stats} />}
      {f.main === "visual" && <Block b={p.visuals[0].block} />}
      {f.main === "keyTable" && p.keyTable && <Table columns={p.keyTable.columns} rows={p.keyTable.rows} caption={p.keyTable.caption} />}
      <details className="col-details">
        <summary>詳しい調べ方（数字・図・集計の条件・言葉の説明）</summary>
        <div className="col-more">
          {!f.showSummary && p.summary && <p className="col-pre">{p.summary}</p>}
          {!f.showStats && p.stats.length > 0 && <Stats items={p.stats} />}
          {p.statsRest.length > 0 && <Stats items={p.statsRest} />}
          {p.statsGuide && <div><p className="col-h3">数字の読み方</p><p className="col-pre">{p.statsGuide}</p></div>}
          {p.visuals[0]?.guide && f.main === "visual" && <div><p className="col-h3">図の読み方</p><p className="col-pre">{p.visuals[0].guide}</p></div>}
          {visuals.map((v, i) => (
            <div key={i}>
              <Block b={v.block} />
              {v.guide && <p className="col-cap">読み方: {v.guide}</p>}
            </div>
          ))}
          {p.keyTable?.guide && <div><p className="col-h3">表の読み方</p><p className="col-pre">{p.keyTable.guide}</p></div>}
          {p.notes.map((n, i) => (
            <div key={i}>
              <p className="col-h3">{n.title}</p>
              <p className="col-pre">{n.text}</p>
              {n.example && <p className="col-pre col-small">例）{n.example}</p>}
            </div>
          ))}
          <ColumnBody blocks={p.details} />
          <div>
            <p className="col-h3">集計の条件</p>
            <dl className="col-kv">
              <dt>通説</dt><dd>「{col.claim}」</dd>
              {p.question && <><dt>調べたこと</dt><dd>{p.question}</dd></>}
              <dt>対象期間</dt><dd>{ds.period}</dd>
              {ds.conditions && <><dt>対象条件</dt><dd>{ds.conditions}</dd></>}
              {ds.sample && <><dt>件数</dt><dd>{ds.sample}</dd></>}
              <dt>公開日</dt><dd>{col.published}{col.updated && `（更新 ${col.updated}）`}</dd>
              <dt>結論の種別</dt><dd>{VERDICTS[col.verdict].label}</dd>
              <dt>カテゴリ</dt>
              <dd>
                {col.categories.map((c, i) => (
                  <span key={c}>{i > 0 && "・"}<Link href={`/column/category/${c}`} className="col-link">{categoryLabel(c)}</Link></span>
                ))}
              </dd>
            </dl>
          </div>
          {p.terms.length > 0 && (
            <div>
              <p className="col-h3">言葉の説明</p>
              <dl className="col-kv">
                {p.terms.map((t) => <div key={t.term} style={{ display: "contents" }}><dt>{t.term}</dt><dd>{t.desc}</dd></div>)}
              </dl>
            </div>
          )}
        </div>
      </details>
      {p.links.length > 0 && (
        <section>
          <h2 className="col-h2">関連するコラム</h2>
          <ul className="col-rel">
            {p.links.map((l) => <li key={l.href}><Link href={l.href}>{l.desc ?? l.label} →</Link></li>)}
          </ul>
        </section>
      )}
    </article>
  );
}

function Stats({ items }: { items: { label: string; value: string; sub?: string }[] }) {
  return (
    <div className="col-stats" style={{ "--n": Math.min(items.length, 3) } as React.CSSProperties}>
      {items.map((s) => (
        <div key={s.label} className="col-stat col-panel">
          <p className="col-stat-lab">{s.label}</p>
          <p className="col-stat-val">{s.value}</p>
          {s.sub && <p className="col-stat-sub">{s.sub}</p>}
        </div>
      ))}
    </div>
  );
}

function Table({ columns, rows, caption }: { columns: string[]; rows: (string | number)[][]; caption?: string }) {
  return (
    <div className="col-table">
      <table>
        <thead><tr>{columns.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, j) => (
            <tr key={j}>{r.map((v, k) => <td key={k} className={typeof v === "number" ? "font-mono" : undefined}>{v}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {caption && <p className="col-cap">{caption}</p>}
    </div>
  );
}

/** 1 つの部品を描く（図・表・文）。囲みは地の囲みだけ */
function Block({ b }: { b: ColumnBlock }) {
  switch (b.type) {
    case "heading": return <h3 className="col-h3">{b.text}</h3>;
    case "text": return <p className="col-pre">{b.text}</p>;
    case "note": return <p className="col-pre col-small">注: {b.text}</p>;
    case "guide": return <p className="col-pre col-small">{b.title}: {b.text}</p>;
    case "glossary": return (
      <dl className="col-kv">{b.items.map((t) => <div key={t.term} style={{ display: "contents" }}><dt>{t.term}</dt><dd>{t.desc}</dd></div>)}</dl>
    );
    case "stats": return <Stats items={b.items} />;
    case "chart": return <ColumnChart block={b} />;
    case "jockey_distance_chart": return <JockeyDistanceChart block={b} />;
    case "heatmap": return <ColumnHeatmap block={b} />;
    case "treemap": return <ColumnTreemap block={b} />;
    case "wordcloud": return <ColumnWordcloud block={b} />;
    case "trainer_pattern_table": return <ColumnTrainerPattern block={b} />;
    case "accordion": return <><p className="col-h3">{b.title}</p><ColumnBody blocks={b.blocks} /></>;
    case "link": return <Link href={b.href} className="col-link">{b.desc ?? b.label} →</Link>;
    case "table": return <Table columns={b.columns} rows={b.rows} caption={b.caption} />;
  }
}

export function ColumnBody({ blocks }: { blocks: ColumnBlock[] }) {
  return <>{blocks.map((b, i) => <div key={i}><Block b={b} /></div>)}</>;
}
