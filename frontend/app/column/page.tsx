import { PageShell } from "@/components/PageShell";
import { CategoryGrid, ColumnLayout, ColumnList } from "@/components/column/ColumnParts";
import { COLUMNS } from "@/content/columns";
import { sortByDate } from "@/lib/columns";

export default function ColumnPage() {
  return (
    <PageShell title="めぐコラム" description="巷の通説を実データで検証してみました。気になるテーマをカテゴリからたどれます。" note={false}>
      <ColumnLayout columns={COLUMNS}>
        <section className="space-y-2">
          <h2 className="font-bold" style={{ fontSize: 15 }}>カテゴリから探す</h2>
          <CategoryGrid columns={COLUMNS} />
        </section>
        <section className="space-y-2">
          <h2 className="font-bold" style={{ fontSize: 15 }}>新着のコラム</h2>
          <ColumnList columns={sortByDate(COLUMNS)} empty="コラムは準備中です。公開までしばらくお待ちください。" />
        </section>
      </ColumnLayout>
    </PageShell>
  );
}
