import { PageShell } from "@/components/PageShell";
import { ColumnLayout, ColumnList } from "@/components/column/ColumnParts";
import { COLUMNS } from "@/content/columns";
import { sortByDate } from "@/lib/columns";

export default function ColumnPage() {
  return (
    <PageShell title="めぐコラム" description="巷の通説を実データで検証してみました。" note={false}>
      <ColumnLayout columns={COLUMNS}>
        <ColumnList columns={sortByDate(COLUMNS)} empty="コラムは準備中です。公開までしばらくお待ちください。" />
      </ColumnLayout>
    </PageShell>
  );
}
