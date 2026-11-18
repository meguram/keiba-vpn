import "@/styles/column.css";
import { PageShell } from "@/components/PageShell";
import { ColumnBrowse } from "@/components/column/ColumnParts";
import { ColumnFilterList } from "@/components/column/ColumnFilterList";
import { summarize } from "@/components/column/ColumnCard";
import { COLUMNS } from "@/content/columns";

export default function ColumnPage() {
  return (
    <PageShell title="めぐコラム" description="競馬の通説を、実データで確かめた記事です。" note={false}>
      <ColumnBrowse columns={COLUMNS}>
        <ColumnFilterList columns={COLUMNS.map(summarize)} empty="コラムは準備中です。" />
      </ColumnBrowse>
    </PageShell>
  );
}
